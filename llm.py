"""Chat answering: rewrite follow-ups, retrieve evidence (BM25 + FAISS + RRF), ask the LLM, attach the cited pages."""
import logging
import os
import re

from rag import DOCUMENT_TYPES, available_pairs, find_codes, log_retrieval_debug, retrieve, tokenize

PLACEHOLDER_KEYS = {"", "your-openrouter-api-key-here", "your-openrouter-api-key", "your_api_key_here", "replace_me"}
log = logging.getLogger("autoassist.llm")
MAX_EVIDENCE = 3          # up to three evidence references (PDF + page) are shown under an answer
CITATION_PATTERN = re.compile(r"[ \t]*\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]", re.IGNORECASE)
DTC_PATTERN = re.compile(r"\b[PBCU][0-9A-F]{4}\b", re.IGNORECASE)
STATUS_PATTERN = re.compile(r"^\s*\[\[\s*STATUS\s*:\s*(FULL|PARTIAL|NONE)\s*\]\]\s*", re.IGNORECASE)
GENERAL_LABEL = "" \
""

# Work where a wrong or unsupported instruction can injure someone or damage the vehicle.
SAFETY_CRITICAL = re.compile(
    r"\b(brake|braking|abs|airbag|srs|seat ?belt pretension|high[- ]voltage|hv battery|hybrid|48 ?v|12 ?v|"
    r"hydrogen|fuel (?:line|pump|rail|system|pressure)|steering|suspension|jack|lift|tow|torque|tighten|"
    r"timing (?:chain|belt)|cylinder head|camshaft|crankshaft|fire|wiring|short circuit|disconnect|"
    r"coolant (?:cap|hot)|radiator cap|exhaust|carbon monoxide|adblue|def)\b", re.IGNORECASE)

REFERENCE_WORDS = re.compile(
    r"\b(it|this|that|these|those|they|them|its|same|above|previous|earlier|else|more|again|next|then|"
    r"also|another|other|one)\b", re.IGNORECASE)

REWRITE_PROMPT = """You rewrite a technician's latest chat message into ONE complete, standalone question that can be used to search Toyota vehicle manuals.

Rules:
- Use the recent conversation ONLY to resolve references such as "it", "this", "that", "the problem", "next".
- Do not answer the question. Do not add causes, parts, fault codes, numbers or facts that the technician did not mention or that are not in the conversation.
- Keep the technician's intent (why / how to check / what to do next / is it dangerous ...).
- If the latest message is already complete and clear, return it unchanged.
- Output ONLY the rewritten question on a single line, 40 words maximum.

Example
Conversation: Technician: My engine is overheating.
Latest message: Why does it happen?
Output: Why does the Toyota Fortuner engine overheat?"""

SYSTEM_PROMPT = """You are Toyota Fortuner AutoAssist, a careful workshop assistant for technicians. You answer from the uploaded Toyota documents that are supplied to you as numbered passages such as [S1].

FIRST LINE (mandatory, machine-read, never shown to the technician)
Start your reply with exactly one of:
[[STATUS: FULL]]     the passages answer the question
[[STATUS: PARTIAL]]  the passages answer only part of it
[[STATUS: NONE]]     the passages do not contain relevant information for this question (or no passages were supplied)
Judge relevance yourself. Retrieved passages can be about a different topic; ignore passages that do not help.

ACCURACY AND SAFETY
- The technician selected one vehicle. Answer only for that vehicle. Never mix other models, generations, engines or markets.
- Facts, values, procedures and causes must come from the passages. Never invent specifications, torque values, wiring details, fault-code meanings, part numbers, fluid types or repair steps. Never invent citations.
- Passages labelled GENERAL TOYOTA DTC REFERENCE only explain what a fault code generally means; never present them as a repair procedure for this vehicle.
- Passages labelled SYNTHETIC, UNVERIFIED are not verified Toyota data: say clearly that this part is synthetic and unverified.
- An owner's manual supports owner-level checks and maintenance; do not present it as a workshop repair manual.
- Passages marked [Table] list one "label | value" row per line: read each value together with the label on its own row. Never take a value from a different row.
- Always carry over any WARNING, CAUTION or NOTICE text in the passages that applies to the steps you give, under "Safety precautions:". Do not shorten it until its meaning is lost.
- If passages from different documents disagree (the line "Evidence conflict possible: YES" tells you when to check), state both versions and say which kind of document each comes from. Never merge them and never pick one silently.
- Never advise bypassing a safety system. For dangerous work (high voltage / 48 V, airbags, brakes, fuel system, lifting, towing, hot cooling system) give only what the passages support and recommend a qualified technician.
- Workshop safety rules set by the administrator:
{safety_rules}

HOW TO ANSWER BY STATUS
FULL: answer mainly from the passages.
PARTIAL: explain what the passages support, then add a short line "Not found in the documents:" listing what is missing, and suggest a work order for engineer review.
NONE: do NOT show only a "no match" message. Write:
  Line 1: "The uploaded Toyota documents for this vehicle do not contain enough information about this."
  Then, only if it is useful and safe, a section titled exactly:
  {general_label}
  with a SHORT list of general automotive pointers (simple causes or checks). In that section: never present anything as Toyota guidance, never give torque values, part numbers, capacities or step-by-step repair procedures, and add no [S#] citations. If the question is about safety-critical work (see above) or is not about vehicles, give no general instructions: say a qualified technician or the official Toyota service information is needed.
  End with: "You can create a work order for an engineer to review."

STYLE (FULL and PARTIAL)
- Simple English, short numbered steps, no long paragraphs, no repeated points, no markdown symbols (no **, no #). Headings are plain text ending with a colon.
- For a symptom, fault or "how do I fix / check" question use these headings in order, and leave out a heading only when the passages give nothing for it:
  Problem understanding:
  Possible causes:
  Recommended inspection steps:
  Troubleshooting procedure:
  Safety precautions:
  Supporting evidence:
- For a simple fact question (a value, a location, what a light means) give the direct answer in the first line, add only what is needed, then "Supporting evidence:".
- Supporting evidence: one to three short lines, each ending with the source ids it uses, for example "Check the coolant level when the engine is cold [S2]." Also cite [S#] after statements in the other sections that come from a passage. Cite only ids that were supplied and that you used. Never write file names, page numbers or scores yourself.
- The "Conversation so far" is only to understand what the technician means; do not take facts from it.
- Do NOT say "according to the PDF". Do not repeat raw passages."""


def _client(api_key):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
                  default_headers={"HTTP-Referer": "http://localhost:5000", "X-Title": "Toyota Fortuner AutoAssist"},
                  timeout=120.0, max_retries=2)


def _api_key():
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return "" if key.lower() in PLACEHOLDER_KEYS else key


def _short(text, limit):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def format_history(history, answer_chars=250, turns=3):
    """The last few exchanges as plain text (answers are shortened)."""
    lines = []
    for item in (history or [])[-turns:]:
        lines.append(f"Technician: {_short(item.get('question'), 300)}")
        if item.get("answer"):
            lines.append(f"Assistant: {_short(item.get('answer'), answer_chars)}")
    return "\n".join(lines)


def needs_rewrite(question, history):
    """Follow-ups are short or point back at the earlier chat ('Why does it happen?', 'Is it dangerous?')."""
    if not history:
        return False
    return len(question.split()) <= 9 or bool(REFERENCE_WORDS.search(question))


def heuristic_rewrite(question, history):
    """Offline fallback: put the most recent self-contained earlier question in front of the follow-up
    ('The AdBlue warning light is on. Can I fill it myself'), so its symptom / component words are searched too."""
    for item in reversed(history or []):
        previous = (item.get("question") or "").strip()
        if len(previous.split()) >= 4:
            return f"{previous.rstrip('?. ')}. {question.rstrip('?. ')}"
    return question


def specifics(text):
    """Fault codes and identifiers that must survive rewriting: P0171, 1GR-FE, 2GD-FTV, 48V, 4WD ..."""
    return {c.upper() for c in DTC_PATTERN.findall(text or "")} | find_codes(text or "")


def preserve_specifics(original, rewritten, history):
    """A rewrite may never lose a fault code or identifier. Codes of the original question are re-added when missing;
    a follow-up that points back at the chat ('what does it mean?') also gets the codes of the last question."""
    missing = [c for c in sorted(specifics(original)) if c not in specifics(rewritten)]
    if not missing and history and REFERENCE_WORDS.search(original) and not specifics(original):
        last = next((specifics(h.get("question")) for h in reversed(history) if specifics(h.get("question"))), set())
        missing = [c for c in sorted(last) if c not in specifics(rewritten)]
    return (f"{rewritten} ({', '.join(missing)})" if missing else rewritten), missing


def understand_query(question, history, model, generation, api_key=None):
    """Context-aware rewriting BEFORE retrieval. Returns a dict that keeps both texts for debugging:
    {"original", "rewritten", "method": none|llm|heuristic, "restored_codes": [...]}"""
    if not needs_rewrite(question, history):
        rewritten, how = question, "none"
    else:
        api_key = api_key if api_key is not None else _api_key()
        rewritten, how = None, "heuristic"
        if api_key:
            try:
                response = _client(api_key).chat.completions.create(
                    model=os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"), temperature=0, max_tokens=90,
                    messages=[{"role": "system", "content": REWRITE_PROMPT},
                              {"role": "user", "content": f"Selected vehicle: {model}, {generation}\n\n"
                                                          f"Conversation:\n{format_history(history)}\n\n"
                                                          f"Latest message: {question}\nOutput:"}])
                text = (response.choices[0].message.content or "").strip().splitlines()[0:1]
                text = re.sub(r"^(output|rewritten question)\s*:\s*", "", text[0].strip().strip('"\''), flags=re.IGNORECASE) if text else ""
                if 3 <= len(text.split()) <= 45:
                    rewritten, how = text, "llm"
            except Exception as exc:
                log.warning("Query rewrite failed (%s); using the offline fallback.", type(exc).__name__)
        if rewritten is None:
            rewritten = heuristic_rewrite(question, history)
    rewritten, restored = preserve_specifics(question, rewritten, history)
    return {"original": question, "rewritten": rewritten, "method": how, "restored_codes": restored}


def rewrite_query(question, history, model, generation, api_key=None):
    """Compatibility wrapper used by eval_retrieval.py: (query, method)."""
    info = understand_query(question, history, model, generation, api_key)
    return info["rewritten"], info["method"]


def top_sources(ranked_sources, limit=1):
    """The best-ranked retrieved pages (document + page), used when nothing valid is cited."""
    found, seen = [], set()
    for item in ranked_sources:
        key = (item.get("source"), item.get("page"))
        if item.get("neighbour") or not item.get("source") or item.get("page") is None or key in seen:
            continue
        seen.add(key)
        found.append({"document": key[0], "page": key[1]})
        if len(found) >= limit:
            break
    return found


def attach_source(answer, source_info, ranked_sources):
    """Remove the model's [S#] markers and return the evidence pages the answer really cites.

    Returns (clean_answer, [{"document":..., "page":...}, ...]). Evidence is ordered by retrieval rank
    (S1 = top-ranked passage). If nothing valid is cited, the top retrieved passage is used.
    """
    cited = []

    def collect(match):
        for sid in re.split(r"\s*[,;]\s*", match.group(1).upper()):
            if sid in source_info and sid not in cited:
                cited.append(sid)
        return ""

    answer = CITATION_PATTERN.sub(collect, answer)
    answer = re.sub(r"[ \t]*\[\s*S\d+\s*\]", "", answer)       # unknown ids
    answer = re.sub(r"[ \t]+([.,;:!?])", r"\1", answer)        # tidy space left before punctuation
    answer = re.sub(r"[ \t]{2,}", " ", answer).strip()

    if cited:
        cited.sort(key=lambda sid: int(sid[1:]))
        return answer, [dict(source_info[sid]) for sid in cited[:MAX_EVIDENCE]]
    return answer, top_sources(ranked_sources, 1)


def ensure_general_label(answer):
    """General knowledge must always sit under the exact 'not verified' label (the model sometimes forgets it).
    The first paragraph is the 'documents do not contain enough information' line; anything after it that is
    not the closing work-order hint is general advice and gets the label placed above it."""
    if GENERAL_LABEL.lower() in answer.lower():
        return answer
    paragraphs = [p for p in re.split(r"\n\s*\n", answer) if p.strip()]
    for n, text in enumerate(paragraphs[1:], start=1):
        if not text.lower().startswith("you can create a work order"):
            return "\n\n".join(paragraphs[:n] + [f"{GENERAL_LABEL}:"] + paragraphs[n:])
    return answer


def no_documents_note(model, generation):
    """Explains a missing exact-generation manual (and never uses another vehicle's pages)."""
    others = [f"{m}, {g}" for m, g in available_pairs()
              if (m or "").casefold() != model.casefold() or (g or "").casefold() != generation.casefold()]
    note = f"No indexed manual was found for exactly {model}, {generation}, so no manual pages could be used."
    if others:
        note += " Documents for other vehicles exist, but they are not used for this vehicle."
    return note


def build_evidence(sources):
    """Number each distinct (document, page) S1, S2... and build the passage text for the model."""
    source_ids, source_info, parts = {}, {}, []
    uses_synthetic, remaining = False, 40000
    for item in sources[:10]:
        excerpt = (item.get("text") or "").strip()
        if not excerpt:
            continue
        key = (item.get("source"), item.get("page"))
        if key not in source_ids:
            source_ids[key] = f"S{len(source_ids) + 1}"
            source_info[source_ids[key]] = {"document": key[0], "page": key[1]}
        if item.get("applicability") == "general_dtc_reference":
            label = "GENERAL TOYOTA DTC REFERENCE (not vehicle-specific)"
        else:
            label = f"VEHICLE-SPECIFIC: {item.get('model')} {item.get('generation')}, {DOCUMENT_TYPES.get(item.get('document_type'), 'document')}"
            if item.get("year_range") and item["year_range"] != "Not specified":
                label += f", {item['year_range']}"
        if item.get("section"):
            label += f" | section: {item['section']}"
        if item.get("synthetic"):
            label += " | SYNTHETIC, UNVERIFIED"
            uses_synthetic = True
        block = f"[{source_ids[key]} | {label}]\n{excerpt}"[:remaining]
        parts.append(block)
        remaining -= len(block)
        if remaining <= 0:
            break
    return "\n\n---\n\n".join(parts), source_info, uses_synthetic


_UNIT = (r"mm|cm|km/h|km|kg|mg|ml|l|litres?|liters?|n[·.]?m|kpa|mpa|bar|psi|rpm|°\s?[cf]|volts?|v|amps?|a|ah|kw|hp|ps|%|"
         r"minutes?|mins?|seconds?|hours?|hrs?|qt|gal|lb|in|inch(?:es)?|mph|nm")
_QUANTITY_RE = re.compile(rf"(?<![\w.])(\d+(?:[.,]\d+)?)\s?({_UNIT})(?![a-z])", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")


def passage_texts(sources, source_info):
    """{S-id: all passage text shown to the model under that id} (same numbering as build_evidence)."""
    ids = {(info["document"], info["page"]): sid for sid, info in source_info.items()}
    out = {}
    for item in sources[:10]:
        sid = ids.get((item.get("source"), item.get("page")))
        if sid and (item.get("text") or "").strip():
            out[sid] = out.get(sid, "") + "\n" + item["text"]
    return out


def _content_tokens(text):
    return {t for t in tokenize(text) if len(t) > 2}


def verify_citations(answer, texts):
    """Check every [S#] marker against the passage it points to. A statement whose words and numbers are not in the cited
    passage is re-pointed to the passage that really contains them, or loses the marker when none does.
    Returns (answer, notes). The notes are for the debug log."""
    if not texts:
        return answer, []
    tokens = {sid: _content_tokens(t) | set(_NUMBER_RE.findall(t)) for sid, t in texts.items()}
    notes, lines = [], []
    for line in answer.split("\n"):
        match = CITATION_PATTERN.search(line)
        if not match:
            lines.append(line)
            continue
        cited = [x.upper() for x in re.split(r"\s*[,;]\s*", match.group(1)) if x.upper() in tokens]
        claim = CITATION_PATTERN.sub("", line)
        claim_tokens = _content_tokens(claim) | set(_NUMBER_RE.findall(claim))
        if not claim_tokens or not cited:
            lines.append(line)
            continue
        share = {sid: len(claim_tokens & toks) / len(claim_tokens) for sid, toks in tokens.items()}
        best = max(share, key=share.get)
        if max(share[c] for c in cited) >= 0.30 or share[best] < 0.40:
            if max(share[c] for c in cited) < 0.15:
                lines.append(CITATION_PATTERN.sub("", line))           # nothing supports this statement
                notes.append(f"citation removed (no passage supports it): {claim.strip()[:80]}")
            else:
                lines.append(line)
        else:
            lines.append(CITATION_PATTERN.sub(f" [{best}]", line, count=1))
            notes.append(f"citation moved {','.join(cited)} -> {best}: {claim.strip()[:80]}")
    return "\n".join(lines), notes


def unsupported_specifics(answer, texts):
    """Quantities (value + unit) and fault codes in the answer that appear in NONE of the passages."""
    evidence = "\n".join(texts.values())
    if not evidence:
        return []
    body = CITATION_PATTERN.sub("", answer)
    known_numbers = {n.replace(",", ".") for n in _NUMBER_RE.findall(evidence)}
    known_codes = specifics(evidence)
    bad = []
    for value, unit in _QUANTITY_RE.findall(body):
        if value.replace(",", ".") not in known_numbers and f"{value} {unit}" not in bad:
            bad.append(f"{value} {unit}")
    for code in sorted(specifics(body)):
        if code not in known_codes and not re.fullmatch(r"\d+V|\dWD", code):
            bad.append(code)
    return bad[:6]


def _complete(client, llm_model, max_tokens, system, user):
    """One model call; if the provider cut the answer at its token limit, ask it to continue (max 2 times)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    response = client.chat.completions.create(model=llm_model, temperature=0.1, max_tokens=max_tokens, messages=messages)
    choice = response.choices[0]
    answer, finish = (choice.message.content or "").strip(), getattr(choice, "finish_reason", None)
    for _ in range(2):
        if finish != "length" or not answer:
            break
        more = client.chat.completions.create(
            model=llm_model, temperature=0.1, max_tokens=max_tokens,
            messages=messages + [{"role": "assistant", "content": answer},
                                 {"role": "user", "content": "Continue exactly where you stopped. Do not repeat previous text."}])
        finish = getattr(more.choices[0], "finish_reason", None)
        extra = (more.choices[0].message.content or "").strip()
        if extra:
            answer += "\n\n" + extra
    return answer


def ask_assistant(question, model, generation, safety_rules, top_k=5, history=None, allowed_doc_ids=None):
    """Full chat turn. Returns {"answer", "sources", "mode", "query"}.

    mode: ok (documents answer it) | partial (documents answer part) | general_only (no document support,
    clearly labelled general knowledge) | configuration_error | api_error
    history: [{"question":..., "answer":...}, ...] - the recent chat for this vehicle, oldest first.
    A full diagnostic record of every turn goes to logs/retrieval_debug.jsonl (never to the technician).
    """
    api_key = _api_key()
    understood = understand_query(question, history or [], model, generation, api_key)
    query = understood["rewritten"]
    if understood["method"] != "none":
        log.info("Query rewritten (%s): %r -> %r", understood["method"], question, query)

    found = retrieve(query, model, generation, top_k=top_k, allowed_doc_ids=allowed_doc_ids)
    sources = [s for s in found["results"] if s.get("applicability") in ("vehicle_specific", "general_dtc_reference")]
    strength = found["strength"] if sources else "none"

    record = {"original_question": question, "rewritten_question": query, "rewrite_method": understood["method"],
              "restored_codes": understood["restored_codes"], "vehicle": f"{model} | {generation}",
              "retrieval": found.get("debug", {}), "strength": strength, "reranked": found.get("reranked"),
              "errors": list(found.get("debug", {}).get("errors", []))}

    def finish(result, answer_text=""):
        record.update(answer=answer_text or result["answer"], mode=result["mode"],
                      evidence_shown=result["sources"])
        log_retrieval_debug(record)
        return result

    if not api_key:
        record["errors"].append("no OpenRouter API key configured")
        return finish({"answer": ("The manual was searched, but no OpenRouter API key is configured. An administrator must set "
                                  "OPENROUTER_API_KEY in the .env file and restart Flask."),
                       "sources": [], "mode": "configuration_error", "query": query})

    evidence, source_info, uses_synthetic = build_evidence(sources) if sources else ("", {}, False)
    texts = passage_texts(sources, source_info)
    critical = bool(SAFETY_CRITICAL.search(question) or SAFETY_CRITICAL.search(query))
    system = SYSTEM_PROMPT.format(safety_rules=safety_rules or "None set.", general_label=GENERAL_LABEL)
    llm_model = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1800"))
    conversation = format_history(history)
    user = (f"Selected vehicle: {model}, {generation}\n"
            f"Safety-critical topic: {'YES' if critical else 'no'}\n"
            f"Retrieval confidence: {strength}\n"
            f"Evidence conflict possible: {'YES' if found.get('conflict_possible') else 'no'}\n"
            + (f"\nConversation so far:\n{conversation}\n" if conversation else "")
            + f"\nTechnician's latest message: {question}\n"
            + (f"Understood as: {query}\n" if query != question else "")
            + "\nDocument passages (cite with the [S#] ids):\n"
            + (evidence if evidence else "(none - no passage matched this question for the selected vehicle)"))

    try:
        answer = _complete(_client(api_key), llm_model, max_tokens, system, user)
        if not answer:
            answer = "I couldn't prepare an answer from the available manual information. Please try a more specific question."
        record["raw_llm_answer"] = answer
        match = STATUS_PATTERN.match(answer)
        status = match.group(1).upper() if match else {"strong": "FULL", "weak": "PARTIAL"}.get(strength, "NONE")
        answer = STATUS_PATTERN.sub("", answer, count=1).strip()
        if not sources:
            status = "NONE"
        record["llm_status"] = status

        if status == "NONE":
            # Never show a citation for an answer the documents do not support.
            answer = CITATION_PATTERN.sub("", answer).strip()
            answer = ensure_general_label(answer)
            if not found["eligible"]:
                answer = f"{no_documents_note(model, generation)}\n\n{answer}"
            return finish({"answer": answer, "sources": [], "mode": "general_only", "query": query})

        if uses_synthetic and "synthetic" not in answer.lower():
            answer = "Note: part of this answer comes from synthetic, unverified information.\n\n" + answer
        answer, citation_notes = verify_citations(answer, texts)         # every [S#] must point at the passage that holds the claim
        unsupported = unsupported_specifics(answer, texts)               # values / codes that are in no passage
        record["citation_checks"], record["unsupported_specifics"] = citation_notes, unsupported
        answer, evidence_pages = attach_source(answer, source_info, sources if strength == "strong" else [])
        if unsupported:
            answer += ("\n\nCheck before use: " + ", ".join(unsupported) + " could not be found in the manual pages used for this "
                       "answer. Confirm in the official Toyota documents before relying on it.")
        return finish({"answer": answer, "sources": evidence_pages, "mode": "ok" if status == "FULL" else "partial", "query": query})
    except Exception as exc:
        # Details go to the log only; the technician sees a safe, generic message (never the key).
        detail = f"{type(exc).__name__}: {str(exc).replace(api_key, '[hidden]')[:300]}"
        log.error("LLM request failed: %s", detail)
        record["errors"].append("LLM request failed: " + detail)
        return finish({
            "answer": ("Relevant manual pages were found, but the AI answer could not be generated right now "
                       "(connection, credits or API key problem). The pages below are the evidence that was found; "
                       "please try again, or create a work order for an engineer. An administrator can check the log."
                       if sources else
                       "The AI answer could not be generated right now (connection, credits or API key problem). "
                       "Please try again, or create a work order for an engineer."),
            "sources": top_sources(sources, 1), "mode": "api_error", "query": query,
        })
