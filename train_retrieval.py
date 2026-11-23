"""Prepare data for, fine-tune and compare the retrieval models (CrossEncoder reranker; embeddings only on request).

Run from the project folder, in this order:
  python train_retrieval.py export                 # list every chunk with a stable id (training/chunk_catalog.csv)
  python train_retrieval.py build-data             # query / positive / hard-negative rows -> training/triplets.csv
  python train_retrieval.py validate               # check the rows (vehicle match, no question in two splits)
  python train_retrieval.py train                  # fine-tune the CrossEncoder, compare it with the pretrained one, keep the better
  python train_retrieval.py evaluate               # pretrained vs fine-tuned on the held-out TEST questions
  python train_retrieval.py train --embedding      # ONLY if evaluation shows the embedding model is the bottleneck

Labels come from the hand-checked pages in training/eval_questions.json; this script never invents a label.
The train / validation / test split is fixed per question id (eval_retrieval.assign_split), so a test question is
never used for training. Nothing is reported as trained unless the command really ran and printed its results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

import eval_retrieval as ev
import rag

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "training"
CATALOG_FILE = TRAINING_DIR / "chunk_catalog.csv"
TRIPLETS_FILE = TRAINING_DIR / "triplets.csv"
QUESTIONS_FILE = TRAINING_DIR / "eval_questions.json"
COMPARISON_FILE = TRAINING_DIR / "reranker_comparison.json"
TRAINED_DIR = BASE_DIR / "trained_models"
EMBEDDING_DIR = TRAINED_DIR / "embedding"
RERANKER_DIR = TRAINED_DIR / "reranker"
CROSS_ENCODER_BASE = rag.PRETRAINED_RERANKER
DATA_COLUMNS = ["question", "model", "generation", "positive_chunk_id", "negative_chunk_id", "split"]
NEGATIVES_PER_POSITIVE = 3
MIN_GAIN = 0.01                      # a fine-tuned model must beat the pretrained MRR by at least this much to be kept
log = logging.getLogger("autoassist.training")


# ----------------------------------------------------------------------------------------------------------------------
# Catalog and labelled data
# ----------------------------------------------------------------------------------------------------------------------
def chunk_id(chunk: dict) -> str:
    """Stable ID tied to source/page/chunk/vehicle and the chunk text."""
    text_hash = hashlib.sha256(str(chunk.get("text", "")).encode("utf-8")).hexdigest()
    parts = [chunk.get(k, "") for k in ("source", "page", "chunk", "doc_id", "model", "generation")]
    return hashlib.sha256(("|".join(map(str, parts)) + "|" + text_hash).encode("utf-8")).hexdigest()[:20]


def export_training_files() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    chunks = rag.load_chunks()
    if not chunks:
        raise SystemExit("No document chunks found. Upload and index PDFs before exporting the training catalog.")
    with CATALOG_FILE.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["chunk_id", "source", "page", "chunk", "model", "generation", "document_type", "applicability", "synthetic", "text"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for chunk in chunks:
            writer.writerow({"chunk_id": chunk_id(chunk), **{key: chunk.get(key, "") for key in fields if key != "chunk_id"}})
    if not TRIPLETS_FILE.exists():
        with TRIPLETS_FILE.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=DATA_COLUMNS).writeheader()
    print(f"Exported {len(chunks)} chunks to {CATALOG_FILE}")


def read_catalog() -> tuple[list[dict], dict[str, dict]]:
    chunks = rag.load_chunks()
    return chunks, {chunk_id(chunk): chunk for chunk in chunks}


def read_triplets(require_rows=True) -> list[dict]:
    if not TRIPLETS_FILE.exists():
        raise SystemExit("Training file missing. Run: python train_retrieval.py export")
    rows = []
    with TRIPLETS_FILE.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(DATA_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit("Missing CSV columns: " + ", ".join(sorted(missing)))
        for row in reader:
            if any((value or "").strip() for value in row.values()):
                rows.append({key: (row.get(key) or "").strip() for key in DATA_COLUMNS})
    if require_rows and not rows:
        raise SystemExit("training/triplets.csv is empty. Run: python train_retrieval.py build-data")
    return rows


def build_data() -> None:
    """query + positive + hard negatives from the hand-checked eval questions.

    positive: the chunk on a labelled relevant page that best matches the question (BM25), at most one per page.
    hard negative: a chunk of the SAME vehicle that the keyword/hybrid search ranks high but that is NOT on a labelled page,
    and is at least 2 pages away from every labelled page of the same PDF (neighbouring pages may well hold the answer,
    so they are never used as negatives). Follow-up questions are rewritten first, as the application does."""
    import llm
    chunks = rag.load_chunks()
    if not chunks:
        raise SystemExit("No chunks. Index the PDFs first.")
    ids = {id(c): chunk_id(c) for c in chunks}
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    rows, per_split, skipped = [], Counter(), []
    for q in questions:
        if not q["relevant"]:
            continue                                         # unanswerable questions are test material, not training pairs
        split = ev.assign_split(q["id"])
        query = llm.heuristic_rewrite(q["question"], q["history"]) if q.get("history") else q["question"]
        relevant = {tuple(x) for x in q["relevant"]}
        dtc = rag.is_dtc_question(query)
        eligible = [c for c in chunks if rag.chunk_allowed(c, q["model"], q["generation"], dtc)]
        terms = rag.query_terms(query)
        scores, _, _ = rag._bm25(terms, [rag.tokenize(f"{c.get('section', '')} {c.get('text', '')}") for c in eligible])
        score_of = {id(c): s for c, s in zip(eligible, scores)}
        positives = []
        for source, page in sorted(relevant):
            on_page = [c for c in eligible if c.get("source") == source and c.get("page") == page]
            if on_page:
                positives.append(max(on_page, key=lambda c: score_of[id(c)]))
        positives = sorted(positives, key=lambda c: -score_of[id(c)])[:2]
        if not positives:
            skipped.append(q["id"])
            continue
        near = {(s, p + d) for s, p in relevant for d in (-1, 0, 1)}
        found = rag.retrieve(query, q["model"], q["generation"], 15, use_dense=False, use_rerank=False)["results"]
        negatives = [r for r in found if not r.get("neighbour") and (r.get("source"), r.get("page")) not in near
                     and rag.normalize_text(r["text"])[:200] not in {rag.normalize_text(p["text"])[:200] for p in positives}]
        negatives = negatives[:NEGATIVES_PER_POSITIVE * len(positives)]
        by_key = {(c.get("source"), c.get("page"), c.get("chunk")): c for c in chunks}
        for n, neg in enumerate(negatives):
            positive = positives[n % len(positives)]
            negative = by_key[(neg["source"], neg["page"], neg["chunk"])]
            rows.append({"question": query, "model": q["model"], "generation": q["generation"],
                         "positive_chunk_id": ids[id(positive)], "negative_chunk_id": ids[id(negative)], "split": split})
            per_split[split] += 1
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    with TRIPLETS_FILE.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    by_question = {s: len({r["question"] for r in rows if r["split"] == s}) for s in ("train", "validation", "test")}
    print(f"Wrote {len(rows)} query/positive/negative rows to {TRIPLETS_FILE}")
    print("rows per split:", dict(per_split), "| questions per split:", by_question)
    if skipped:
        print("skipped (no chunk on a labelled page):", skipped)


def validate_dataset() -> list[dict]:
    chunks, catalog = read_catalog()
    rows = read_triplets()
    errors, question_splits = [], {}
    for line_number, row in enumerate(rows, start=2):
        if not row["question"] or not row["model"] or not row["generation"]:
            errors.append(f"Line {line_number}: question, model and generation are required.")
            continue
        if row["split"] not in {"train", "validation", "test"}:
            errors.append(f"Line {line_number}: split must be train, validation or test.")
        qkey = re.sub(r"\s+", " ", row["question"].casefold()).strip()
        if question_splits.setdefault(qkey, row["split"]) != row["split"]:
            errors.append(f"Line {line_number}: the same question appears in more than one split (data leakage).")
        positive, negative = catalog.get(row["positive_chunk_id"]), catalog.get(row["negative_chunk_id"])
        if not positive or not negative:
            errors.append(f"Line {line_number}: positive/negative chunk ID not found. Re-run export and build-data after re-indexing PDFs.")
            continue
        if row["positive_chunk_id"] == row["negative_chunk_id"]:
            errors.append(f"Line {line_number}: positive and negative passages must be different.")
        dtc = rag.is_dtc_question(row["question"])
        if not rag.chunk_allowed(positive, row["model"], row["generation"], dtc):
            errors.append(f"Line {line_number}: positive passage is not applicable to the selected model/generation.")
        if not rag.chunk_allowed(negative, row["model"], row["generation"], dtc):
            errors.append(f"Line {line_number}: negative passage is not applicable to the selected model/generation.")
    counts = Counter(row["split"] for row in rows)
    questions = {s: len({r["question"] for r in rows if r["split"] == s}) for s in ("train", "validation", "test")}
    print("Rows per split:", dict(counts), "| distinct questions per split:", questions, "| chunks:", len(chunks))
    if errors:
        print("\nDataset validation failed:")
        for error in errors[:30]:
            print(" -", error)
        raise SystemExit(f"Found {len(errors)} validation issue(s). Fix them before training.")
    if questions["train"] < 10 or questions["validation"] < 2 or questions["test"] < 2:
        raise SystemExit("Provide at least 10 train, 2 validation and 2 test questions.")
    print("Dataset validation passed (vehicle applicability, split separation).")
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# Reranker: train, compare, keep the better one
# ----------------------------------------------------------------------------------------------------------------------
def safe_replace_dir(temp_dir: Path, destination: Path) -> None:
    backup = destination.with_name(destination.name + "_previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        temp_dir.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def test_questions(split="test"):
    return [q for q in json.loads(QUESTIONS_FILE.read_text(encoding="utf-8")) if ev.assign_split(q["id"]) == split]


def compare_rerankers(contenders: dict, split="test") -> dict:
    """Page-level retrieval metrics of the same hybrid pipeline with each reranker (None = RRF only)."""
    questions = test_questions(split)
    out = {}
    for name, reranker in contenders.items():
        def run(q, reranker=reranker):
            query = ev.followup_query(q, "heuristic")
            found = rag.retrieve(query, q["model"], q["generation"], ev.K, use_rerank=reranker is not None, reranker=reranker)
            return found["results"], found["strength"] == "strong"
        out[name] = ev.score(questions, run)
        out[name].pop("failure_details", None)
    return out


def _load_ce(reference):
    model, why = ev.load_cross_encoder(reference)
    if model is None:
        raise SystemExit(f"Could not load CrossEncoder '{reference}': {why}\nThe first run needs internet access to download it.")
    return model


def train_reranker(epochs: int, batch_size: int) -> None:
    rows = validate_dataset()
    _, catalog = read_catalog()
    train_rows = [r for r in rows if r["split"] == "train"]
    validation_rows = [r for r in rows if r["split"] == "validation"]
    try:
        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Training dependencies are missing. Run: pip install -r requirements.txt") from exc

    examples, seen = [], set()
    for row in train_rows:                                   # one (question, passage, label) pair per distinct passage
        for chunk_key, label in (("positive_chunk_id", 1.0), ("negative_chunk_id", 0.0)):
            key = (row["question"], row[chunk_key])
            if key not in seen:
                seen.add(key)
                examples.append(InputExample(texts=[row["question"], rag.rerank_text(catalog[row[chunk_key]])], label=label))
    print(f"Fine-tuning the CrossEncoder '{CROSS_ENCODER_BASE}' on {len(examples)} pairs "
          f"({sum(1 for e in examples if e.label == 1.0)} positive) from {len({r['question'] for r in train_rows})} questions.")
    candidate_dir = TRAINED_DIR / "reranker_candidate"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(CROSS_ENCODER_BASE, num_labels=1, max_length=512)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    model.fit(train_dataloader=loader, epochs=epochs, warmup_steps=max(1, int(len(loader) * epochs * 0.1)),
              output_path=str(candidate_dir), show_progress_bar=True)
    if not (candidate_dir / "config.json").exists():
        model.save(str(candidate_dir))
    if validation_rows:
        pairs, groups = [], []
        for r in validation_rows:
            groups.append(len(pairs))
            pairs += [[r["question"], rag.rerank_text(catalog[r["positive_chunk_id"]])],
                      [r["question"], rag.rerank_text(catalog[r["negative_chunk_id"]])]]
        scores = model.predict(pairs, show_progress_bar=False)
        print(f"Validation pairwise accuracy (positive scored above its hard negative): "
              f"{sum(float(scores[i]) > float(scores[i + 1]) for i in groups) / len(groups):.1%}")
    del model

    # --- keep only the better model -------------------------------------------------------------------------
    contenders = {"pretrained": _load_ce(CROSS_ENCODER_BASE), "fine_tuned_candidate": _load_ce(str(candidate_dir))}
    if RERANKER_DIR.is_dir():
        contenders["fine_tuned_current"] = _load_ce(str(RERANKER_DIR))
    results = compare_rerankers(contenders, "test")
    report_comparison(results)
    ranked = sorted(results, key=lambda n: (results[n]["MRR"], results[n]["Hit@1"], results[n][f"Hit@{ev.K} (retrieval accuracy)"]), reverse=True)
    best = ranked[0]
    safe = results[best]["Wrong-model retrieval rate"] == 0.0
    gain = results[best]["MRR"] - results["pretrained"]["MRR"]
    keep_fine_tuned = best != "pretrained" and safe and gain >= MIN_GAIN
    decision = {"best": best if keep_fine_tuned else "pretrained", "mrr_gain_over_pretrained": round(gain, 4),
                "test_questions": len(test_questions("test")), "epochs": epochs, "batch_size": batch_size, "results": results}
    COMPARISON_FILE.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    if keep_fine_tuned and best == "fine_tuned_candidate":
        safe_replace_dir(candidate_dir, RERANKER_DIR)
        print(f"\nKEPT the fine-tuned reranker (MRR +{gain:.3f} on held-out questions): {RERANKER_DIR}")
    elif keep_fine_tuned:
        shutil.rmtree(candidate_dir, ignore_errors=True)
        print("\nThe existing fine-tuned reranker is still the best one; the new candidate was discarded.")
    else:
        shutil.rmtree(candidate_dir, ignore_errors=True)
        shutil.rmtree(RERANKER_DIR, ignore_errors=True)
        print("\nThe fine-tuned reranker did NOT beat the pretrained one by the required margin on the held-out questions: "
              "it was discarded and the application keeps using the pretrained CrossEncoder.")
    print(f"Saved the comparison to {COMPARISON_FILE}")


def report_comparison(results: dict) -> None:
    keys = [k for k in next(iter(results.values())) if k != "missed"]
    print("\nHeld-out TEST questions, same hybrid retrieval, different reranker:")
    print(f"{'Metric':<46}" + "".join(f"{n[:22]:>24}" for n in results))
    for k in keys:
        print(f"{k:<46}" + "".join(f"{results[n][k]:>24}" for n in results))


def evaluate_models() -> None:
    validate_dataset()
    contenders = {"pretrained": _load_ce(CROSS_ENCODER_BASE)}
    if RERANKER_DIR.is_dir():
        contenders["fine_tuned"] = _load_ce(str(RERANKER_DIR))
    else:
        print("No fine-tuned reranker is saved (none trained yet, or it did not beat the pretrained one).")
    results = compare_rerankers(contenders, "test")
    report_comparison(results)
    COMPARISON_FILE.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    if EMBEDDING_DIR.is_dir():
        print("\nEmbedding model (chunk-level, held-out test rows):")
        rows = [r for r in validate_dataset() if r["split"] == "test"]
        for name, ref in (("base", rag.BASE_EMBEDDING_MODEL), ("fine-tuned", str(EMBEDDING_DIR))):
            print(f"  {name}: {compute_retrieval_metrics(rows, ref)}")


# ----------------------------------------------------------------------------------------------------------------------
# Embedding model (opt-in: only when the evaluation shows that retrieval, not ranking, is the problem)
# ----------------------------------------------------------------------------------------------------------------------
def compute_retrieval_metrics(rows: list[dict], model_reference: str) -> dict:
    """Chunk-level Recall@5 / MRR@5 of BM25 + embeddings + RRF for the given embedding model (no reranker)."""
    from sentence_transformers import SentenceTransformer

    chunks, _ = read_catalog()
    encoder = SentenceTransformer(model_reference)
    encoder.max_seq_length = 256
    vectors = encoder.encode([rag.embed_text(c) for c in chunks], normalize_embeddings=True, convert_to_numpy=True,
                             batch_size=64, show_progress_bar=True).astype("float32")
    tokenized = [rag.tokenize(f"{c.get('section', '')} {c.get('text', '')}") for c in chunks]
    ids = [chunk_id(c) for c in chunks]
    ranks, wrong, returned = [], 0, 0
    for row in rows:
        dtc = rag.is_dtc_question(row["question"])
        eligible = [i for i, c in enumerate(chunks) if rag.chunk_allowed(c, row["model"], row["generation"], dtc)]
        if not eligible:
            ranks.append(0.0)
            continue
        scores, _, coverage = rag._bm25(rag.query_terms(row["question"]), [tokenized[i] for i in eligible])
        bm25_rank = [p for p in sorted(range(len(eligible)), key=lambda p: scores[p], reverse=True)
                     if scores[p] > 0 and coverage[p] >= rag.BM25_MIN_COVERAGE][:rag.CANDIDATE_POOL]
        query_vector = encoder.encode([row["question"]], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]
        dense = vectors[eligible] @ query_vector
        dense_rank = [p for p in np.argsort(-dense).tolist() if float(dense[p]) >= rag.DENSE_FLOOR][:rag.CANDIDATE_POOL]
        fused = rag.reciprocal_rank_fusion([bm25_rank, dense_rank])
        top = sorted(fused, key=lambda p: fused[p], reverse=True)[:5]
        for p in top:
            returned += 1
            wrong += 0 if rag.chunk_allowed(chunks[eligible[p]], row["model"], row["generation"], dtc) else 1
        top_ids = [ids[eligible[p]] for p in top]
        ranks.append(1.0 / (top_ids.index(row["positive_chunk_id"]) + 1) if row["positive_chunk_id"] in top_ids else 0.0)
    return {"questions": len(ranks), "recall_at_5": round(sum(1 for r in ranks if r) / max(1, len(ranks)), 4),
            "mrr_at_5": round(sum(ranks) / max(1, len(ranks)), 4), "wrong_model_rate": round(wrong / max(1, returned), 4)}


def train_embedding(epochs: int, batch_size: int) -> None:
    rows = validate_dataset()
    _, catalog = read_catalog()
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Training dependencies are missing. Run: pip install -r requirements.txt") from exc
    train_rows = [r for r in rows if r["split"] == "train"]
    candidate = TRAINED_DIR / "embedding_candidate"
    if candidate.exists():
        shutil.rmtree(candidate)
    model = SentenceTransformer(rag.BASE_EMBEDDING_MODEL)
    model.max_seq_length = 256
    examples = [InputExample(texts=[r["question"], rag.embed_text(catalog[r["positive_chunk_id"]]),
                                    rag.embed_text(catalog[r["negative_chunk_id"]])]) for r in train_rows]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs, warmup_steps=max(1, int(len(loader) * epochs * 0.1)),
              output_path=str(candidate), show_progress_bar=True)
    del model
    test_rows = [r for r in rows if r["split"] == "test"]
    base, tuned = compute_retrieval_metrics(test_rows, rag.BASE_EMBEDDING_MODEL), compute_retrieval_metrics(test_rows, str(candidate))
    print("base      :", base, "\nfine-tuned:", tuned)
    if tuned["mrr_at_5"] >= base["mrr_at_5"] + MIN_GAIN and tuned["recall_at_5"] >= base["recall_at_5"] and tuned["wrong_model_rate"] == 0:
        safe_replace_dir(candidate, EMBEDDING_DIR)
        rebuild_index()
        print(f"KEPT the fine-tuned embedding model: {EMBEDDING_DIR}")
    else:
        shutil.rmtree(candidate, ignore_errors=True)
        print("The fine-tuned embedding model did not clearly beat the base model on held-out questions: discarded.")


def rebuild_index() -> None:
    chunks = rag.load_chunks()
    if not chunks:
        raise SystemExit("No RAG chunks exist. Upload/index PDFs first.")
    rag._embedding_model = None
    rag._embedding_model_loaded_id = None
    rag.save_knowledge_base(chunks)
    print(f"Rebuilt FAISS with {len(chunks)} chunks using: {rag.embedding_model_id()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, train and compare the Toyota Fortuner retrieval models.")
    parser.add_argument("command", choices=["export", "build-data", "validate", "train", "evaluate", "rebuild-index"])
    parser.add_argument("--epochs", type=int, default=3, help="training epochs (default 3)")
    parser.add_argument("--batch-size", type=int, default=8, help="training batch size (default 8; use 4 if memory is short)")
    parser.add_argument("--embedding", action="store_true", help="also fine-tune the embedding model (only if evaluation justifies it)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive integers.")
    if args.command == "export":
        export_training_files()
    elif args.command == "build-data":
        build_data()
    elif args.command == "validate":
        validate_dataset()
    elif args.command == "train":
        train_reranker(args.epochs, args.batch_size)
        if args.embedding:
            train_embedding(args.epochs, args.batch_size)
    elif args.command == "evaluate":
        evaluate_models()
    elif args.command == "rebuild-index":
        rebuild_index()


if __name__ == "__main__":
    main()
