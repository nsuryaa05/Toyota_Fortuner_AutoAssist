"""Retrieval evaluation on training/eval_questions.json (hand-checked ground-truth pages).

Run from the project folder (no MySQL, Flask or API key needed):
    python eval_retrieval.py                  # 1 baseline | 2 FAISS+BM25+RRF | 3 + pretrained CrossEncoder | 4 + fine-tuned CrossEncoder
    python eval_retrieval.py --split test     # only the held-out questions (never used for training the reranker)
    python eval_retrieval.py --keyword        # force keyword-only mode (BM25 + RRF, no FAISS)
    python eval_retrieval.py --rechunk        # also test the current chunker on the PDFs (keyword-only)
    python eval_retrieval.py --sweep          # threshold sweep (BM25 / dense / CrossEncoder / evidence cut)
    python eval_retrieval.py --llm-rewrite    # use the OpenRouter LLM to rewrite follow-ups (needs OPENROUTER_API_KEY)

A configuration whose model cannot be loaded (no internet / no weights) is reported as NOT RUN, never estimated.

Metrics use the top-K distinct PDF pages, K = 5:
  Recall@K      share of the relevant pages that were retrieved
  Precision@K   share of the retrieved pages that are relevant
  MRR           1 / rank of the first relevant page
  Hit@1, Hit@K  a relevant page is first / is among the first K
  Citation correctness  share of the 3 best pages (the ones shown as evidence) that are relevant
  Wrong-model retrieval rate  chunks returned that belong to a different model/generation than the selected one
  False no-match rate   an answerable question for which nothing was retrieved
  Unanswerable handling a question NOT covered by the documents for which the system correctly reports NO strong evidence
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import rag  # noqa: E402

K = 5
QUESTIONS = BASE / "training" / "eval_questions.json"
RESULTS = BASE / "training" / "retrieval_eval_results.json"


def assign_split(question_id):
    """Stable train / validation / test split by question id (about 60 / 15 / 25 %).
    train_retrieval.py uses the same function, so reranker training never sees a test question."""
    bucket = int(hashlib.sha256(str(question_id).encode("utf-8")).hexdigest(), 16) % 100
    return "train" if bucket < 60 else "validation" if bucket < 75 else "test"


def load_cross_encoder(reference):
    """A CrossEncoder object, or (None, reason) when it cannot be loaded here."""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(reference, max_length=512), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:100]}"

# ---------------------------------------------------------------------------------------------------
# The OLD retrieval, copied from the original rag.py so the comparison is against what really ran before.
# ---------------------------------------------------------------------------------------------------
OLD_STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on", "at", "for",
            "from", "with", "by", "and", "or", "what", "how", "why", "when", "where", "which", "who", "can", "could",
            "should", "would", "do", "does", "did", "my", "me", "i", "it", "this", "that", "about", "tell", "please",
            "explain", "fortuner", "toyota"}
_old_tok_cache = {}


def old_tokenize(text):
    return [t.casefold() for t in re.findall(r"[a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)*", text)
            if t.casefold() not in OLD_STOP and len(t) > 1]


def legacy_retrieve(question, model, generation, chunks, dense_fn, top_k=K):
    dtc = rag.is_dtc_question(question)
    eligible = [i for i, c in enumerate(chunks) if rag.chunk_allowed(c, model, generation, dtc)]
    if not eligible:
        return []
    subset = [chunks[i] for i in eligible]
    for i in eligible:
        if i not in _old_tok_cache:
            _old_tok_cache[i] = old_tokenize(chunks[i].get("text", ""))
    terms = old_tokenize(question)
    unique = list(dict.fromkeys(terms))
    bm25, matched = rag.bm25_scores(terms, [_old_tok_cache[i] for i in eligible])
    bm25_ranking = [p for p in sorted(range(len(subset)), key=lambda p: bm25[p], reverse=True)
                    if bm25[p] > 0 and matched[p] / max(1, len(unique)) >= 0.30][:30]      # old coverage gate
    dense = dense_fn(question, eligible)
    dense_ranking = []
    if dense is not None:
        dense_ranking = [p for p in sorted(range(len(subset)), key=lambda p: dense[p], reverse=True)
                         if dense[p] >= 0.28][:30]                                         # old semantic gate
    rankings = [bm25_ranking, dense_ranking]
    codes = set(re.findall(r"\b[A-Za-z]{0,3}\d{2,}[A-Za-z0-9-]*\b", question.upper()))     # old code rule
    if codes:
        code_ranking = [p for p in bm25_ranking if any(c in subset[p].get("text", "").upper() for c in codes)]
        if code_ranking:
            rankings.append(code_ranking)
    fused = rag.reciprocal_rank_fusion(rankings)
    ordered = sorted(fused, key=lambda p: fused[p], reverse=True)[:top_k]
    return [dict(subset[p], neighbour=False) for p in ordered]


# ---------------------------------------------------------------------------------------------------
def make_dense_fn(chunks):
    """Cosine scores from the stored FAISS vectors, or None when the embedding model cannot be loaded."""
    try:
        rag.encode(["test"])
        matrix = rag.load_index().reconstruct_n(0, rag.load_index().ntotal)
    except Exception as exc:
        print(f"[note] semantic search unavailable here ({type(exc).__name__}); keyword-only mode.")
        return None
    cache = {}

    def dense(question, eligible):
        if question not in cache:
            cache[question] = rag.encode([question])[0]
        return [max(0.0, float(matrix[i] @ cache[question])) for i in eligible]
    return dense


def pages_of(results):
    seen, pages = set(), []
    for r in results:
        if r.get("neighbour"):
            continue
        key = (r.get("source"), r.get("page"))
        if key not in seen:
            seen.add(key)
            pages.append(key)
    return pages[:K]


def score(questions, run):
    """run(q) -> (results, strong: bool). Returns the metric dictionary."""
    rec, prec, mrr, hit1, hitk, cite, nomatch, answerable = [], [], [], 0, 0, 0, 0, 0
    wrong_chunks, total_chunks, false_evidence, unanswerable = 0, 0, 0, 0
    misses, details = [], []
    for q in questions:
        results, strong = run(q)
        for r in results:
            total_chunks += 1
            if not rag.chunk_allowed(r, q["model"], q["generation"], rag.is_dtc_question(q["question"])):
                wrong_chunks += 1
        if not q["relevant"]:
            unanswerable += 1
            false_evidence += 1 if strong else 0
            continue
        answerable += 1
        relevant = {tuple(x) for x in q["relevant"]}
        pages = pages_of(results)
        if not results:
            nomatch += 1
        got = [p for p in pages if p in relevant]
        rec.append(len(set(got)) / len(relevant))
        prec.append(len(got) / len(pages) if pages else 0.0)
        first = next((n for n, p in enumerate(pages, 1) if p in relevant), None)
        mrr.append(1 / first if first else 0.0)
        hit1 += 1 if first == 1 else 0
        hitk += 1 if first else 0
        shown = pages[:3]                                  # the pages that can appear as evidence chips
        cite += (sum(1 for x in shown if x in relevant) / len(shown)) if shown else 0.0
        if not first:
            misses.append(q["id"])
            details.append({"id": q["id"], "type": q["type"], "question": q["question"],
                            "expected": sorted(map(list, relevant))[:4], "got": [list(p) for p in pages[:5]]})
    n = max(1, answerable)
    return {"answerable": answerable, "unanswerable": unanswerable,
            f"Recall@{K}": round(sum(rec) / n, 3), f"Precision@{K}": round(sum(prec) / n, 3),
            "MRR": round(sum(mrr) / n, 3), "Hit@1": round(hit1 / n, 3), f"Hit@{K} (retrieval accuracy)": round(hitk / n, 3),
            "Citation correctness (top-3 pages)": round(cite / n, 3), "False no-match rate": round(nomatch / n, 3),
            "Wrong-model retrieval rate": round(wrong_chunks / max(1, total_chunks), 3),
            "False-evidence rate (not-in-docs questions)": round(false_evidence / max(1, unanswerable), 3),
            "Unanswerable handled correctly": round(1 - false_evidence / max(1, unanswerable), 3),
            "missed": misses, "failure_details": details}


def followup_query(q, mode):
    if not q.get("history"):
        return q["question"]
    import llm
    if mode == "llm":
        return llm.rewrite_query(q["question"], q["history"], q["model"], q["generation"])[0]
    return llm.heuristic_rewrite(q["question"], q["history"])


def wrong_generation_probe(questions, chunks, use_dense):
    """Ask every question under a DIFFERENT vehicle and count chunks that come from the asked-about manual."""
    pairs = sorted({(q["model"], q["generation"]) for q in questions})
    leaked = total = 0
    for q in questions:
        for pair in pairs:
            if pair == (q["model"], q["generation"]):
                continue
            res = rag.retrieve(q["question"], pair[0], pair[1], K, chunks=None if use_dense else chunks,
                               use_dense=use_dense, use_rerank=False)["results"]
            for r in res:
                total += 1
                leaked += 0 if (rag.norm_key(r.get("model")) == rag.norm_key(pair[0])
                                and rag.norm_key(r.get("generation")) == rag.norm_key(pair[1])) else 1
    return leaked, total


def rebuild_chunks_in_memory(old_chunks):
    """Chunks as the NEW chunker would produce them (keyword-only testing; no embeddings)."""
    first = {}
    for c in old_chunks:
        first.setdefault(c["source"], c)
    out = []
    for source, c in first.items():
        pdf = BASE / "uploads" / source
        if not pdf.is_file():
            out.extend(x for x in old_chunks if x["source"] == source)
            continue
        meta = {k: c.get(k) for k in ("model", "generation", "document_type", "year_range", "applicability", "synthetic")}
        pages, state, _ = rag.extract_pages(pdf, use_ocr=False)
        out.extend(rag.build_chunks(source, c.get("doc_id"), meta, pages) if state == "ok" else
                   [x for x in old_chunks if x["source"] == source])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword", action="store_true")
    ap.add_argument("--rechunk", action="store_true")
    ap.add_argument("--chunks-file", help="JSON file with already re-chunked chunks (used with --rechunk to skip PDF re-reading)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--llm-rewrite", action="store_true")
    ap.add_argument("--split", choices=["all", "train", "validation", "test"], default="all")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if args.split != "all":
        questions = [q for q in questions if assign_split(q["id"]) == args.split]
    chunks = rag.load_chunks()
    dense_fn = None if args.keyword else make_dense_fn(chunks)
    use_dense = dense_fn is not None
    mode = "FAISS + BM25 + RRF" if use_dense else "KEYWORD-ONLY (BM25 + RRF; FAISS not tested)"
    rewrite_mode = "llm" if args.llm_rewrite else "heuristic"
    print(f"\n{len(questions)} questions (split: {args.split}) | {sum(1 for q in questions if q['relevant'])} answerable | retrieval: {mode}")
    print(f"follow-up rewriting for the improved system: {rewrite_mode}\n")

    pretrained, why_pre = (None, "keyword-only mode") if args.keyword else load_cross_encoder(rag.PRETRAINED_RERANKER)
    tuned_dir = BASE / "trained_models" / "reranker"
    tuned, why_tuned = (load_cross_encoder(str(tuned_dir)) if tuned_dir.is_dir() and not args.keyword
                        else (None, "no fine-tuned reranker saved in trained_models/reranker"))

    def run_baseline(q):
        res = legacy_retrieve(q["question"], q["model"], q["generation"], chunks, dense_fn or (lambda *_: None))
        return res, bool(res)                           # the old system treated ANY result as usable evidence

    def make_run(reranker, chunk_store=None):
        def run(q):
            query = followup_query(q, rewrite_mode)
            out = rag.retrieve(query, q["model"], q["generation"], K, chunks=chunk_store, use_dense=use_dense and chunk_store is None,
                               use_rerank=reranker is not None, reranker=reranker)
            return out["results"], out["strength"] == "strong"
        return run

    if args.sweep:
        sweep(questions, use_dense, rewrite_mode, pretrained or tuned)
        return

    configs = [("1 Baseline (old retrieval)", run_baseline, ""),
               ("2 BM25+FAISS+RRF" if use_dense else "2 BM25+RRF (no FAISS)", make_run(None), ""),
               ("3 RRF + pretrained CE", make_run(pretrained) if pretrained else None, why_pre),
               ("4 RRF + fine-tuned CE", make_run(tuned) if tuned else None, why_tuned)]
    report = {"retrieval_mode": mode, "split": args.split, "questions": len(questions),
              "components_run": {"faiss": use_dense, "pretrained_crossencoder": pretrained is not None,
                                 "fine_tuned_crossencoder": tuned is not None}, "configs": {}, "not_run": {}}
    for name, run, reason in configs:
        if run is None:
            report["not_run"][name] = reason
            continue
        report["configs"][name] = score(questions, run)
    if args.rechunk:
        if args.chunks_file:
            new_chunks = json.loads(Path(args.chunks_file).read_text(encoding="utf-8"))
        else:
            new_chunks = rebuild_chunks_in_memory(chunks)
        report["rechunked_keyword_only"] = score(questions, make_run(None, new_chunks))
        report["chunks_old_vs_rechunked"] = [len(chunks), len(new_chunks)]
    leaked, total = wrong_generation_probe(questions, chunks, use_dense)
    report["cross_vehicle_probe"] = {"chunks_returned_for_other_vehicles": total, "chunks_from_the_wrong_vehicle": leaked}

    by_type = {}
    best_name = list(report["configs"])[-1]
    best_run = dict((n, r) for n, r, _ in configs)[best_name]
    for t in sorted({q["type"] for q in questions if q["relevant"]}):
        subset = [q for q in questions if q["type"] == t]
        by_type[t] = {"baseline": score(subset, run_baseline)[f"Hit@{K} (retrieval accuracy)"],
                      best_name: score(subset, best_run)[f"Hit@{K} (retrieval accuracy)"], "n": len(subset)}
    report["hit_by_question_type"] = by_type

    cols = list(report["configs"]) + (["rechunked (keyword)"] if args.rechunk else [])
    data = dict(report["configs"]); 
    if args.rechunk:
        data["rechunked (keyword)"] = report["rechunked_keyword_only"]
    keys = [k for k in next(iter(data.values())) if k not in ("missed", "failure_details")]
    print(f"{'Metric':<46}" + "".join(f"{c[:20]:>22}" for c in cols))
    for k in keys:
        print(f"{k:<46}" + "".join(f"{data[c][k]:>22}" for c in cols))
    for name, reason in report["not_run"].items():
        print(f"\nNOT RUN: {name}  ({reason})")
    print()
    for name, result in report["configs"].items():
        print(f"Missed in [{name}]: {result['missed']}")
    print(f"\nHit@{K} by question type: {json.dumps(by_type)}")
    print(f"Cross-vehicle probe: {leaked} wrong-vehicle chunks out of {total} returned")
    report["retrieval_failures"] = report["configs"][best_name]["failure_details"]
    for f in report["retrieval_failures"]:
        print(f"  FAIL {f['id']} [{f['type']}] {f['question']!r}\n       expected {f['expected']}  got {f['got']}")
    RESULTS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {RESULTS.relative_to(BASE)}")


def sweep(questions, use_dense, rewrite_mode, reranker=None):
    """Try threshold values on the labelled questions and print how each setting behaves."""
    cache = {}
    orig_encode = rag.encode
    rag.encode = lambda texts: cache.setdefault(tuple(texts), orig_encode(texts)) if use_dense else orig_encode(texts)
    answerable = [q for q in questions if q["relevant"]]
    others = [q for q in questions if not q["relevant"]]
    grids = [("BM25_MIN_COVERAGE", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), ("EVIDENCE_REL_CUT", [0.0, 0.2, 0.3, 0.4, 0.5, 0.6])]
    if use_dense:
        grids.append(("DENSE_FLOOR", [0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.32, 0.36]))
        grids.append(("DENSE_STRONG", [0.30, 0.36, 0.42, 0.48, 0.55]))
    if reranker is not None:
        grids += [("RERANK_DROP", [0.0, 0.005, 0.01, 0.03, 0.1]), ("RERANK_STRONG", [0.1, 0.2, 0.3, 0.5, 0.7]),
                  ("RERANK_WEIGHT", [0.5, 0.7, 0.85, 1.0])]
    print(f"{'setting':<22}{'value':>7}{'Hit@5':>9}{'MRR':>8}{'Prec@5':>8}{'no-match':>10}{'strong(ans)':>13}{'strong(not in docs)':>21}")

    def go(q, text):
        return rag.retrieve(text, q["model"], q["generation"], K, use_dense=use_dense, use_rerank=reranker is not None, reranker=reranker)

    for name, values in grids:
        original = getattr(rag, name)
        for v in values:
            setattr(rag, name, v)
            hits = mrr = nomatch = strong_ans = prec = 0
            for q in answerable:
                out = go(q, followup_query(q, rewrite_mode))
                pages = pages_of(out["results"])
                rel = {tuple(x) for x in q["relevant"]}
                first = next((n for n, p in enumerate(pages, 1) if p in rel), None)
                hits += 1 if first else 0
                mrr += 1 / first if first else 0
                prec += (sum(1 for p in pages if p in rel) / len(pages)) if pages else 0
                nomatch += 0 if out["results"] else 1
                strong_ans += 1 if out["strength"] == "strong" else 0
            strong_no = sum(1 for q in others if go(q, q["question"])["strength"] == "strong")
            n = len(answerable)
            print(f"{name:<22}{v:>7}{hits / n:>9.3f}{mrr / n:>8.3f}{prec / n:>8.3f}{nomatch / n:>10.3f}{strong_ans / n:>13.3f}{strong_no / max(1, len(others)):>21.3f}")
        setattr(rag, name, original)
        print()
    rag.encode = orig_encode


if __name__ == "__main__":
    main()
