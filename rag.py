"""RAG engine: PDF ingestion (with OCR fallback), section-aware chunking, metadata-filtered hybrid retrieval (BM25 + FAISS + RRF).

MySQL only keeps document records. The page-wise chunks (with PDF name + page) live in rag_store/chunks.json
and the FAISS vectors in rag_store/manuals.faiss, so they survive a restart.
"""
import os

# Same thread limit here, in case rag.py is imported on its own.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import json
import logging
import math
import re
import shutil
import threading
import time
from datetime import datetime
from collections import Counter
from logging.handlers import RotatingFileHandler
from pathlib import Path

import faiss
import numpy as np
import pymupdf

try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

log = logging.getLogger("autoassist.rag")

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "rag_store"
CHUNKS_FILE = STORE_DIR / "chunks.json"
INDEX_FILE = STORE_DIR / "manuals.faiss"
INDEX_META_FILE = STORE_DIR / "index_manifest.json"
BASE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = BASE_EMBEDDING_MODEL  # kept as a compatibility name for older project code
_embedding_model = None
_embedding_model_loaded_id = None
_cross_encoder = None
_cross_encoder_loaded_id = None
PRETRAINED_RERANKER = os.getenv("PRETRAINED_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_model_failures = {}                 # model kind -> time of the last failed load (no repeated slow retries)
MODEL_RETRY_SECONDS = 300
_LOCK = threading.RLock()            # one writer at a time (uploads are processed one after another)
_cache = {"chunks": (None, []), "index": (None, None)}

# Only question filler is removed. Technical words, part names and codes are never treated as stop words.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "from", "with", "by", "and", "or",
    "what", "how", "why", "when", "where", "which", "who", "can", "could",
    "should", "would", "do", "does", "did", "my", "me", "i", "it", "this",
    "that", "about", "tell", "please", "explain", "fortuner", "toyota",
}
# Extra filler that is dropped from the QUESTION only (the manuals keep every word).
QUERY_FILLER = {
    "happen", "happens", "happening", "problem", "problems", "issue", "issues", "help", "need", "want",
    "just", "also", "really", "much", "many", "get", "gets", "getting", "go", "goes", "going", "make",
    "makes", "way", "thing", "things", "there", "they", "them", "these", "those", "some", "any", "have",
    "has", "had", "will", "may", "might", "must", "if", "then", "than", "so", "because", "but", "not",
    "no", "yes", "vehicle", "car", "after", "before", "now", "next", "again", "possible", "possibly",
    "okay", "ok", "myself", "yourself", "ourselves", "regarding", "anything", "something", "everything",
}
# Technician wording -> manual wording. Used for the keyword (BM25) search only. Keys are stemmed words.
QUERY_SYNONYMS = {
    "hot": ["overheat", "temperature", "coolant"], "overheat": ["temperature", "coolant", "radiator"],
    "tyre": ["tire"], "puncture": ["flat", "tire", "spare"], "flat": ["tire", "spare"],
    "gearbox": ["transmission"], "petrol": ["fuel", "gasoline"], "diesel": ["fuel"],
    "bonnet": ["hood"], "boot": ["luggage"], "stall": ["engine", "stop"],
    "leak": ["leakage", "fluid"], "noise": ["sound", "buzzer"], "vibration": ["vibrate"],
    "ac": ["air", "conditioning"], "mil": ["malfunction", "indicator", "lamp"],
    "dead": ["discharged", "battery"], "jumpstart": ["jump", "start", "battery"],
    "brake": ["braking"], "smoke": ["steam"], "flash": ["blink"], "sluggish": ["power", "acceleration"],
    "handbrake": ["parking", "brake"], "hold": ["capacity"], "inspect": ["check"], "check": ["inspect"],
    "burn": ["fire", "extinguish"], "stay": ["remain"], "refill": ["fill", "add"], "topup": ["add", "fill"],
}
QUERY_PHRASES = [  # regex on the lower-cased question -> extra words for BM25
    (r"check engine|engine light|engine warning", ["malfunction", "indicator", "warning", "light"]),
    (r"(won'?t|wont|not|fails? to|unable to|cannot|can'?t) start|no start|cranks? but", ["engine", "start", "battery"]),
    (r"warning light|dash ?board light|indicator light", ["warning", "lights", "indicators"]),
    (r"12 ?-?v(olt)?s? battery", ["12v", "battery"]),
]


_SYNONYMS_BY_STEM = {}                                          # filled lazily: keys stemmed like the tokens


def normalize_units(text):
    """'12-volt' / '12 volt' / '12 V' -> '12v' so the question and the manual use the same token."""
    return re.sub(r"\b(\d+)\s*-?\s*(?:volts?|v)\b", r"\1v", text, flags=re.IGNORECASE)


def stem(token):
    """Tiny stemmer (same rule for questions and manuals): overheating/overheats/overheated -> overheat.
    Codes and part numbers that contain digits are never changed."""
    if any(ch.isdigit() for ch in token) or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    elif token.endswith(("ches", "shes", "sses", "xes", "zes")):
        token = token[:-2]
    elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        token = token[:-1]
    if len(token) > 3 and token[-1] == token[-2] and token[-1] not in "lsz":     # stopp -> stop
        token = token[:-1]
    if len(token) > 3 and token.endswith("e"):                                   # drive / driving -> driv
        token = token[:-1]
    return token


def embedding_model_reference():
    """Prefer the locally fine-tuned model when it exists; otherwise use the original model."""
    configured = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        if candidate.is_dir():
            return str(candidate.resolve())
        log.warning("Configured embedding model folder does not exist; using the base model.")
    trained = BASE_DIR / "trained_models" / "embedding"
    if (trained / "modules.json").exists() or (trained / "config.json").exists():
        return str(trained.resolve())
    return BASE_EMBEDDING_MODEL


def embedding_model_id():
    reference = embedding_model_reference()
    path = Path(reference)
    return str(path.resolve()) if path.exists() else reference


def get_encoder():
    global _embedding_model, _embedding_model_loaded_id
    active_id = embedding_model_id()
    if _embedding_model is None or _embedding_model_loaded_id != active_id:
        from sentence_transformers import SentenceTransformer  # loaded only when first needed
        _embedding_model = SentenceTransformer(embedding_model_reference())
        _embedding_model_loaded_id = active_id
        log.info("Loaded embedding model: %s", active_id)
    return _embedding_model


def _recently_failed(kind):
    return time.time() - _model_failures.get(kind, 0.0) < MODEL_RETRY_SECONDS


def reranker_reference():
    """Which CrossEncoder is used: (reference, kind).

    1. RERANKER_MODEL_PATH, or trained_models/reranker. That folder only exists when train_retrieval.py found
       the fine-tuned model BETTER than the pretrained one on the held-out test questions.
    2. Otherwise the pretrained CrossEncoder (PRETRAINED_RERANKER). RERANKER_ENABLED=0 switches reranking off.
    """
    if os.getenv("RERANKER_ENABLED", "1").strip().lower() in ("0", "false", "no", "off"):
        return None, "disabled"
    configured = os.getenv("RERANKER_MODEL_PATH", "").strip()
    candidate = Path(configured) if configured else (BASE_DIR / "trained_models" / "reranker")
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    if candidate.is_dir():
        return str(candidate.resolve()), "fine_tuned"
    return PRETRAINED_RERANKER, "pretrained"


def get_reranker():
    """Load the CrossEncoder once. Returns None (and keeps working with RRF order) if it cannot be loaded."""
    global _cross_encoder, _cross_encoder_loaded_id
    reference, kind = reranker_reference()
    if reference is None:
        return None
    if _cross_encoder is not None and _cross_encoder_loaded_id == reference:
        return _cross_encoder
    if _recently_failed("reranker:" + reference):
        return None
    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(reference, max_length=512)
        _cross_encoder_loaded_id = reference
        log.info("Loaded %s CrossEncoder reranker: %s", kind, reference)
    except Exception as exc:
        log.warning("Could not load the %s CrossEncoder (%s); using RRF order.", kind, type(exc).__name__)
        _model_failures["reranker:" + reference] = time.time()
        _cross_encoder, _cross_encoder_loaded_id = None, None
    return _cross_encoder


def index_embedding_compatible():
    """Detect stale FAISS vectors after an embedding model is fine-tuned or changed."""
    try:
        manifest = json.loads(INDEX_META_FILE.read_text(encoding="utf-8"))
        return manifest.get("embedding_model") == embedding_model_id()
    except (OSError, json.JSONDecodeError):
        # The supplied original FAISS index predates manifests and was built with this base model.
        return embedding_model_id() == BASE_EMBEDDING_MODEL


def encode(texts):
    return get_encoder().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    ).astype("float32")


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)*")


def tokenize(text):
    """Words + codes, stemmed. Identifiers such as P0300, 4WD, 2GD-FTV and 48V are kept whole;
    compounds such as 'high-voltage' also add their parts so 'high voltage' matches them."""
    tokens = []
    for raw in _TOKEN_RE.findall(normalize_units(text)):
        word = raw.casefold()
        parts = re.split(r"[-_/]", word)
        for piece in ([word] + parts if len(parts) > 1 else [word]):
            if len(piece) > 1 and piece not in STOP_WORDS:
                tokens.append(stem(piece))
    return tokens


def query_terms(question):
    """Search terms of a question: filler removed, stemmed, technician words mapped to manual words."""
    terms = [t for t in tokenize(question) if t not in QUERY_FILLER and stem(t) not in QUERY_FILLER]
    if not _SYNONYMS_BY_STEM:
        _SYNONYMS_BY_STEM.update({stem(k): v for k, v in QUERY_SYNONYMS.items()})
    extra = []
    for t in terms:
        extra += [stem(w) for w in _SYNONYMS_BY_STEM.get(t, [])]
    low = question.lower()
    for pattern, words in QUERY_PHRASES:
        if re.search(pattern, low):
            extra += [stem(w) for w in words]
    return list(dict.fromkeys(terms + extra))


def split_into_chunks(text, chunk_words=300, overlap_words=60):
    """Fallback splitter: overlapping word windows (used only for one very long block of text)."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        part = words[start:start + chunk_words]
        if part:
            chunks.append(" ".join(part))
        if start + chunk_words >= len(words):
            break
    return chunks


# --- Section-aware chunking ---------------------------------------------------------------------
CHUNK_MAX_WORDS = 260
CHUNK_MIN_WORDS = 25
_STEP_RE = re.compile(r"^\s*(?:\d{1,2}[.)]\s|\(\w\)\s|step\s*\d|[a-z][.)]\s)", re.IGNORECASE)
_MARKER_RE = re.compile(r"^\s*[■●▲◆▶]\s*\S")
_PAGE_PREFIX_RE = re.compile(r"^\s*(?:[A-Z]{1,3}\s?[–-]\s?)?\d{1,4}\s{2,}")
_ALLCAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-–/&,().]{7,80}$")


def clean_page_text(text, repeated=frozenset()):
    """Join words broken at a line end, drop bare page numbers and running footers, keep the line structure."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in repeated or re.fullmatch(r"[\d\s]*", stripped) and stripped:
            continue                                         # footer / "1 2 3 ... 10" tab bars / page numbers
        lines.append(re.sub(r"[ \t]+", " ", stripped))
    return re.sub(r"\n{2,}", "\n\n", "\n".join(lines)).strip()


def page_heading(text):
    """The page's running title (for example '6-3. Do-it-yourself maintenance'), or ''."""
    for line in text.splitlines()[:4]:
        line = re.sub(r"^\s*(?:[A-Z]{1,3}\s?[–-]\s?)?\d{1,4}\s+(?=\d+-\d+\.|[A-Z0-9])", "", line).strip()
        if 4 <= len(line) <= 100 and not re.fullmatch(r"[\d\s.]*", line):
            return line
    return ""


def _blocks(text):
    """Split page text into blocks at blank lines and ■ sub-headings, but never inside a numbered procedure."""
    lines = text.splitlines()
    blocks, current = [], []
    for i, line in enumerate(lines):
        starts_new = (not line.strip()) or _MARKER_RE.match(line) or _ALLCAPS_RE.match(line.strip())
        nxt = next((l for l in lines[i + 1:] if l.strip()), "")
        if starts_new and current and not _STEP_RE.match(nxt if not line.strip() else line):
            if len(" ".join(current).split()) >= CHUNK_MIN_WORDS or _MARKER_RE.match(line):
                blocks.append("\n".join(current).strip())
                current = []
        if line.strip():
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


_SAFETY_RE = re.compile(r"^\s*[▲■●◆▶]?\s*(?:WARNING|CAUTION|NOTICE|DANGER|NOTE)\b", re.IGNORECASE)
_SAFETY_ANYWHERE_RE = re.compile(r"\b(?:WARNING|CAUTION|NOTICE|DANGER)\b")
TABLE_MARK = "[Table]"
CHUNK_OVERLAP_WORDS = 45


def has_warning(text):
    """True when the text carries a WARNING / CAUTION / NOTICE / DANGER paragraph (upper-case, as printed in manuals)."""
    return bool(_SAFETY_ANYWHERE_RE.search(text or ""))


def has_table(text):
    return TABLE_MARK in (text or "")


def _split_long_block(block, max_words, overlap_words=CHUNK_OVERLAP_WORDS):
    """Split one block that is longer than max_words WITHOUT cutting a numbered step or a table row in half.

    Lines are packed greedily; the last line(s) of a part are repeated at the start of the next part (overlap) so a
    step is never separated from its predecessor. A table repeats its header rows in every part."""
    lines = [l for l in block.splitlines() if l.strip()]
    if len(lines) < 2:
        return split_into_chunks(block, max_words, overlap_words)       # one giant line: word windows
    header = lines[:2] if lines[0].startswith(TABLE_MARK) else []
    parts, current, count = [], [], 0
    for line in lines:
        words = len(line.split())
        if current and count + words > max_words:
            parts.append(current)
            carry, carried = [], 0
            for prev in reversed(current):                            # overlap: whole trailing lines, max ~45 words
                w = len(prev.split())
                if carried + w > overlap_words and carry:
                    break
                if w <= overlap_words * 2 and prev not in header:
                    carry.insert(0, prev); carried += w
            current = header + carry
            count = sum(len(l.split()) for l in current)
        current.append(line); count += words
    if current:
        parts.append(current)
    return ["\n".join(part) for part in parts]


def build_page_chunks(page_text, max_words=CHUNK_MAX_WORDS):
    """Meaningful chunks for one page.

    * whole sub-sections are packed together, numbered procedures are kept in one piece
    * a WARNING / CAUTION / NOTICE paragraph is glued to the block that follows it (or precedes it, at the page end),
      so a safety statement is never separated from the work it belongs to
    * a table stays together with its heading and surrounding text; when it is too long it is split by rows with the header repeated
    * a block that must be split is cut at step / line borders and the neighbouring parts overlap
    """
    blocks = _blocks(page_text)
    glued, i = [], 0
    while i < len(blocks):
        block = blocks[i]
        if _SAFETY_RE.match(block) and len(block.split()) < 90 and i + 1 < len(blocks) and not blocks[i + 1].startswith(TABLE_MARK):
            block = block + "\n" + blocks[i + 1]
            i += 1
        elif _SAFETY_RE.match(block) and glued and len(block.split()) < 90 and not glued[-1].startswith(TABLE_MARK):
            glued[-1] = glued[-1] + "\n" + block
            i += 1
            continue
        glued.append(block)
        i += 1

    packed, current, count = [], [], 0
    for block in glued:
        words = len(block.split())
        if words > max_words * 1.25:                             # one huge block: step-aware split with overlap
            if current:
                packed.append("\n".join(current)); current, count = [], 0
            packed.extend(_split_long_block(block, max_words))
        elif count + words > max_words and current:
            packed.append("\n".join(current)); current, count = [block], words
        else:
            current.append(block); count += words
    if current:
        packed.append("\n".join(current))
    if len(packed) > 1 and len(packed[-1].split()) < CHUNK_MIN_WORDS:
        tail = packed.pop()                                     # do not leave a tiny tail chunk
        packed[-1] += "\n" + tail
    return packed


def embed_text(chunk):
    """What is embedded for a chunk: the section title gives the vector context the page text may lack."""
    section = chunk.get("section")
    return f"{section}. {chunk['text']}" if section else chunk["text"]


# ---------------------------------------------------------------------------
# Storage on disk (chunks.json + manuals.faiss)
# ---------------------------------------------------------------------------
def _stamp(path):
    try:
        info = path.stat()
        return (str(path), info.st_mtime_ns, info.st_size)
    except OSError:
        return None


def load_chunks():
    stamp = _stamp(CHUNKS_FILE)
    if stamp is None:
        return []
    if _cache["chunks"][0] == stamp:
        return _cache["chunks"][1]
    try:
        data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        data = data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        log.error("chunks.json could not be read")
        return []
    _cache["chunks"] = (stamp, data)
    return data


def load_index():
    stamp = _stamp(INDEX_FILE)
    if stamp is None:
        return None
    if _cache["index"][0] == stamp:
        return _cache["index"][1]
    try:
        index = faiss.read_index(str(INDEX_FILE))
    except Exception:
        log.error("manuals.faiss could not be read")
        return None
    _cache["index"] = (stamp, index)
    return index


def _write_store(chunks, vectors):
    """Write chunks + index through temporary files so a crash never leaves half a file."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_json, tmp_faiss = CHUNKS_FILE.with_suffix(".json.tmp"), INDEX_FILE.with_suffix(".faiss.tmp")
    if len(chunks) == 0:
        INDEX_FILE.unlink(missing_ok=True)
    else:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(tmp_faiss))
        os.replace(tmp_faiss, INDEX_FILE)
    tmp_json.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_json, CHUNKS_FILE)
    tmp_meta = INDEX_META_FILE.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps({"embedding_model": embedding_model_id(), "chunk_count": len(chunks)}, indent=2), encoding="utf-8")
    os.replace(tmp_meta, INDEX_META_FILE)
    _cache["chunks"] = (None, [])
    _cache["index"] = (None, None)


def save_knowledge_base(chunks):
    """Rebuild the whole index by embedding every chunk (only used if the index is out of sync)."""
    with _LOCK:
        vectors = encode([embed_text(c) for c in chunks]) if chunks else None
        _write_store(chunks, vectors)


def update_knowledge_base(all_chunks, old_count):
    """Add only the new chunks to the FAISS index (rebuild if the index is out of sync)."""
    index = load_index()
    if index is None or index.ntotal != old_count or not index_embedding_compatible():
        save_knowledge_base(all_chunks)
        return
    old = np.vstack([index.reconstruct(i) for i in range(old_count)]).astype("float32") if old_count else None
    new = encode([embed_text(c) for c in all_chunks[old_count:]])
    _write_store(all_chunks, new if old is None else np.vstack([old, new]))


def keep_chunks(should_keep):
    """Keep only the chunks for which should_keep(chunk) is True. Vectors of kept chunks are copied (no re-embedding)."""
    with _LOCK:
        chunks = load_chunks()
        keep = [i for i, c in enumerate(chunks) if should_keep(c)]
        removed = len(chunks) - len(keep)
        if not removed:
            return 0
        kept = [chunks[i] for i in keep]
        index = load_index()
        if not kept:
            _write_store([], None)
        elif index is not None and index.ntotal == len(chunks) and index_embedding_compatible():
            _write_store(kept, np.vstack([index.reconstruct(i) for i in keep]).astype("float32"))
        else:
            save_knowledge_base(kept)
        return removed


def remove_document(doc_id):
    """Drop one PDF's chunks (fast: other PDFs' vectors are copied, not re-embedded)."""
    return keep_chunks(lambda c: c.get("doc_id") != doc_id)


def tag_chunks(source_to_doc_id, valid):
    """Point chunks whose doc_id does not match a MySQL record at the record for their PDF.
    valid = {document id: filename} of the records that exist."""
    with _LOCK:
        chunks = load_chunks()
        changed = False
        for c in chunks:
            if valid.get(c.get("doc_id")) != c.get("source") and c.get("source") in source_to_doc_id:
                c["doc_id"] = source_to_doc_id[c["source"]]
                changed = True
        if changed:
            tmp = CHUNKS_FILE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, CHUNKS_FILE)
            _cache["chunks"] = (None, [])


def rag_status():
    chunks, index = load_chunks(), load_index()
    reference, kind = reranker_reference()
    return {"chunks": len(chunks), "vectors": index.ntotal if index else 0,
            "in_sync": (index.ntotal if index else 0) == len(chunks) and index_embedding_compatible(),
            "embedding_model": embedding_model_id(), "reranker_ready": reference is not None,
            "reranker": reference if reference is None or kind == "pretrained" else Path(reference).name,
            "reranker_kind": kind}


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
DOCUMENT_TYPES = {
    "owner_manual": "Owner's manual",
    "service_manual": "Service / repair manual",
    "wiring_diagram": "Wiring diagram",
    "dtc_vehicle": "Vehicle-specific DTC list",
    "general_dtc": "General Toyota DTC reference",
    "other": "Other",
}
GENERAL_LABEL = "General Toyota"
SYNTHETIC_MARKERS = ("SYNTHETIC / NOT OEM-VERIFIED", "SYNTHETIC DATASET", "SYNTHETIC DATA")


def norm_key(value):
    """Normalise a model/generation name so 'Toyota Fortuner ' == 'fortuner'."""
    value = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    return re.sub(r"^toyota\s+", "", value)


def clean_metadata(raw):
    """Validate one PDF's metadata. Returns (metadata, error_message)."""
    doc_type = (raw.get("document_type") or "").strip()
    if doc_type not in DOCUMENT_TYPES:
        return None, "please choose a document type."
    model = re.sub(r"\s+", " ", raw.get("model") or "").strip()
    generation = re.sub(r"\s+", " ", raw.get("generation") or "").strip()
    years = re.sub(r"\s+", " ", raw.get("year_range") or "").strip() or "Not specified"

    # Applicability is decided by the document type: only a general DTC reference is non-vehicle-specific.
    if doc_type == "general_dtc":
        return {
            "model": GENERAL_LABEL, "generation": GENERAL_LABEL, "document_type": doc_type,
            "year_range": years, "applicability": "general_dtc_reference",
            "synthetic": bool(raw.get("synthetic")),
        }, None

    if not model or not generation:
        return None, "vehicle model and generation are required for this document type."
    return {
        "model": model, "generation": generation, "document_type": doc_type,
        "year_range": years, "applicability": "vehicle_specific",
        "synthetic": bool(raw.get("synthetic")),
    }, None


def is_dtc_question(question):
    q = question.upper()
    if re.search(r"\b[PBCU][0-9A-F]{4}\b", q):
        return True
    return bool(re.search(r"\b(DTC|DTCS|FAULT CODE|TROUBLE CODE|DIAGNOSTIC CODE|ERROR CODE)\b", q))


def chunk_allowed(chunk, model, generation, dtc_question):
    """The single rule that decides whether a chunk may be used as evidence."""
    if chunk.get("applicability") == "general_dtc_reference":
        return bool(dtc_question)          # general DTC references: DTC questions only
    if chunk.get("applicability") != "vehicle_specific":
        return False                       # legacy / unknown chunks are never used
    return (norm_key(chunk.get("model")) == norm_key(model)
            and norm_key(chunk.get("generation")) == norm_key(generation))


# ---------------------------------------------------------------------------
# Ingestion: extract text (OCR for scanned PDFs), chunk page by page, embed, add to FAISS
# ---------------------------------------------------------------------------
def tesseract_available():
    return shutil.which("tesseract") is not None


def _ocr_page(page):
    try:
        textpage = page.get_textpage_ocr(language="eng", dpi=200, full=True)
        return page.get_text("text", textpage=textpage, sort=True).strip()
    except Exception as exc:
        log.warning("OCR failed on a page: %s", exc)
        return ""


TABLE_EXTRACTION = os.getenv("TABLE_EXTRACTION", "1").strip().lower() not in ("0", "false", "no", "off")


def _table_rows(page, table):
    """One 'cell | cell | cell' line per table row. Each cell is read with a clipped get_text(), which keeps the
    original spacing and decimal points (table.extract() was found to drop them, so it is not used)."""
    rows = []
    for row in table.rows:
        cells = []
        for box in row.cells:
            text = page.get_text("text", clip=pymupdf.Rect(box), sort=True) if box else ""
            cells.append(re.sub(r"\s+", " ", text).strip())
        if any(cells):
            rows.append(" | ".join(cells))
    return rows


def _page_text(page):
    """Page text in reading order. Real tables (specifications, capacities, fault-code lists) are written as
    '[Table]' + one 'cell | cell | cell' line per row, so a value stays next to its label instead of being scattered."""
    plain = page.get_text("text", sort=True).strip()
    if not TABLE_EXTRACTION or not plain:
        return plain
    try:
        tables = []
        for t in page.find_tables().tables:
            rows = _table_rows(page, t)
            filled = sum(1 for r in rows for c in r.split(" | ") if c)
            leaders = re.search(r"\.{4,}|(?:\. ){4,}", " ".join(rows))          # table-of-contents / index lines are not tables
            if len(rows) >= 2 and t.col_count >= 2 and filled >= 0.5 * len(rows) * t.col_count and not leaders:
                tables.append((t.bbox, rows))
        if not tables:
            return plain
        items, pending = [], sorted(tables, key=lambda x: x[0][1])
        for block in page.get_text("blocks", sort=True):
            if block[6] != 0:
                continue
            cx, cy = (block[0] + block[2]) / 2, (block[1] + block[3]) / 2
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b, _ in tables):
                continue                                         # this text is a table cell: written as a row below
            while pending and pending[0][0][1] <= block[1]:
                items.append(f"\n{TABLE_MARK}\n" + "\n".join(pending.pop(0)[1]) + "\n")
            items.append(block[4].strip())
        for _, rows in pending:
            items.append(f"\n{TABLE_MARK}\n" + "\n".join(rows) + "\n")
        text = "\n".join(items).strip()
        # Safety net: every number of the page (values, capacities, page numbers) must survive, else keep the plain text.
        numbers = Counter(re.findall(r"\d+(?:\.\d+)?", plain))
        kept = Counter(re.findall(r"\d+(?:\.\d+)?", text))
        if sum((numbers - kept).values()) > 0.02 * max(1, sum(numbers.values())) or len(text.split()) < 0.6 * len(plain.split()):
            return plain
        return text
    except Exception as exc:
        log.debug("Table extraction skipped on a page: %s", exc)
        return plain


def extract_pages(pdf_path, use_ocr=True):
    """Return (page_texts, status, message). page_texts = [(page_number, text)].

    status is "ok", "needs_ocr" (scanned PDF and OCR is not available) or "empty".
    Scanned PDFs are read with OCR. In a normal PDF, image-only pages are OCR'd too when Tesseract exists.
    """
    with pymupdf.open(str(pdf_path)) as document:
        if document.needs_pass:
            raise ValueError("the PDF is password protected.")
        total = len(document)
        if total == 0:
            raise ValueError("the PDF has no pages.")
        texts = [_page_text(page) for page in document]
        with_text = sum(1 for t in texts if t)
        scanned = with_text < max(1, int(total * 0.3))       # mostly image pages
        ocr_ready = use_ocr and tesseract_available()
        if scanned and not ocr_ready:
            return [], "needs_ocr", ("this PDF looks scanned (no selectable text). Install Tesseract OCR "
                                     "and press Reprocess.")
        if ocr_ready:
            blank = [n for n, t in enumerate(texts) if len(t) < 20 and (scanned or document[n].get_images())]
            if scanned or len(blank) <= 60:                    # mixed PDFs: OCR the few image-only pages
                for n in blank:
                    texts[n] = _ocr_page(document[n])
        # Running footers (the same line on many pages) are noise, not evidence.
        line_counts = Counter(l.strip() for t in texts for l in set(t.splitlines()) if 8 < len(l.strip()) < 80)
        repeated = frozenset(l for l, c in line_counts.items() if c > max(5, total * 0.4))
        pages = []
        for n, t in enumerate(texts, start=1):
            t = clean_page_text(t, repeated)
            if t:
                pages.append((n, t))
    if not pages:
        return [], "empty", "no readable text was found in this PDF, even after OCR."
    return pages, "ok", ""


def looks_synthetic(pages):
    sample = "\n".join(text for _, text in pages[:12]).upper()
    return any(marker in sample for marker in SYNTHETIC_MARKERS)


def _subheading(text):
    """The sub-heading a chunk starts with (a ■ title or an upper-case title line), or ''."""
    first = next((l.strip() for l in text.splitlines() if l.strip() and not l.startswith(TABLE_MARK)), "")
    if first and len(first.split()) <= 14 and (_MARKER_RE.match(first) or _ALLCAPS_RE.match(first)) and not _SAFETY_RE.match(first):
        return re.sub(r"^[■●▲◆▶]\s*", "", first)
    return ""


def build_chunks(source, doc_id, meta, pages):
    """Chunks for one PDF. Every chunk keeps PDF name, page, section, sub-section, warning/table flags and the vehicle metadata."""
    out, last_section = [], ""
    for page_number, page_text in pages:
        section = page_heading(page_text) or last_section
        last_section = section
        for chunk_number, chunk_text in enumerate(build_page_chunks(page_text), start=1):
            out.append({"text": chunk_text, "source": source, "page": page_number, "chunk": chunk_number,
                        "section": section, "subsection": _subheading(chunk_text),
                        "has_warning": has_warning(chunk_text), "has_table": has_table(chunk_text),
                        "doc_id": doc_id, **meta})
    return out


def index_document(doc_id, pdf_path, raw_meta, use_ocr=True):
    """Process one PDF end to end and add it to the search index.

    Returns {"status": "Indexed" | "Needs OCR" | "Failed", "pages": n, "chunks": n, "message": str}.
    Never raises for a bad PDF; the caller stores the status and message in MySQL.
    """
    meta, error = clean_metadata(raw_meta)
    if error:
        return {"status": "Failed", "pages": 0, "chunks": 0, "message": error}
    try:
        pages, state, message = extract_pages(pdf_path, use_ocr)
    except Exception as exc:                                  # corrupt / encrypted / unreadable PDF
        log.warning("Could not read %s: %s", pdf_path, exc)
        return {"status": "Failed", "pages": 0, "chunks": 0, "message": f"Invalid PDF: {exc}"}
    if state == "needs_ocr":
        return {"status": "Needs OCR", "pages": 0, "chunks": 0, "message": message}
    if state == "empty":
        return {"status": "Failed", "pages": 0, "chunks": 0, "message": message}
    if looks_synthetic(pages):
        meta["synthetic"] = True

    new_chunks = build_chunks(Path(pdf_path).name, doc_id, meta, pages)
    try:
        with _LOCK:
            remove_document(doc_id)                           # reprocessing replaces the old chunks
            all_chunks = list(load_chunks())
            old_count = len(all_chunks)
            all_chunks.extend(new_chunks)
            update_knowledge_base(all_chunks, old_count)
    except Exception as exc:
        log.exception("Indexing failed for %s", pdf_path)
        return {"status": "Failed", "pages": len(pages), "chunks": 0,
                "message": f"Indexing failed ({type(exc).__name__}). See the log."}
    return {"status": "Indexed", "pages": len(pages), "chunks": len(new_chunks), "message": ""}


def rechunk_all(upload_dir=None):
    """Re-read every stored PDF with the current chunker and rebuild the index (keeps doc ids and metadata).
    Run once after upgrading:  python rag.py rechunk"""
    upload_dir = Path(upload_dir or BASE_DIR / "uploads")
    with _LOCK:
        old = load_chunks()
        first = {}
        for c in old:
            first.setdefault(c.get("source"), c)
        rebuilt = []
        for source, c in first.items():
            pdf = upload_dir / (source or "")
            if not pdf.is_file():
                log.warning("PDF missing, keeping its old chunks: %s", source)
                rebuilt.extend(x for x in old if x.get("source") == source)
                continue
            meta = {k: c.get(k) for k in ("model", "generation", "document_type", "year_range", "applicability", "synthetic")}
            pages, state, message = extract_pages(pdf, True)
            if state != "ok":
                log.warning("%s: %s (old chunks kept)", source, message)
                rebuilt.extend(x for x in old if x.get("source") == source)
                continue
            rebuilt.extend(build_chunks(source, c.get("doc_id"), meta, pages))
            print(f"{source}: {len(pages)} pages")
        save_knowledge_base(rebuilt)
        print(f"Rebuilt {len(rebuilt)} chunks (was {len(old)}).")


# ---------------------------------------------------------------------------
# Retrieval: BM25 (keyword) + FAISS (semantic), merged with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
BM25_K1 = 1.5          # term-frequency saturation
BM25_B = 0.75          # document-length normalisation
RRF_K = 60             # standard RRF constant: score = 1 / (RRF_K + rank)
CANDIDATE_POOL = 30    # how many top results each ranker contributes to the fusion
MIN_QUERY_COVERAGE = 0.30   # (old count-based gate, kept only because train_retrieval.py still imports it)
MIN_SEMANTIC_SCORE = 0.28   # (old semantic gate, kept only because train_retrieval.py still imports it)

# Current relevance rules (checked with eval_retrieval.py --sweep):
BM25_MIN_COVERAGE = float(os.getenv("BM25_MIN_COVERAGE", "0.20"))   # share of the question's IDF weight a keyword hit must match
DENSE_FLOOR = float(os.getenv("DENSE_FLOOR", "0.22"))               # lowest cosine similarity that may enter the fusion
DENSE_BAND = 0.20                                                    # ... and it must be within this of the best semantic score
DENSE_STRONG = float(os.getenv("DENSE_STRONG", "0.42"))             # cosine at/above which a chunk alone counts as strong evidence
SYNTHETIC_WEIGHT = 0.6                                               # unverified synthetic chunks rank below real manual pages

# Evidence selection. The BM25/dense numbers above were checked with eval_retrieval.py --sweep. The three RERANK_*
# numbers are STARTING VALUES: they must be re-checked with `python eval_retrieval.py --sweep` on a machine that
# has the CrossEncoder model, because they depend on that model's score scale.
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "20"))                  # how many fused candidates the CrossEncoder scores
RERANK_WEIGHT = float(os.getenv("RERANK_WEIGHT", "0.7"))             # final order = W * CrossEncoder + (1 - W) * RRF
RERANK_DROP = float(os.getenv("RERANK_DROP", "0.01"))                # CrossEncoder probability below this: not evidence
RERANK_STRONG = float(os.getenv("RERANK_STRONG", "0.30"))            # probability at/above which a passage is strong evidence
EVIDENCE_REL_CUT = float(os.getenv("EVIDENCE_REL_CUT", "0.0"))      # drop passages scoring below this share of the best one
NEAR_DUPLICATE = 0.85                                                # shingle containment above which two passages are duplicates
DEBUG_LOG_FILE = BASE_DIR / "logs" / "retrieval_debug.jsonl"


_debug_logger = None


def log_retrieval_debug(record):
    """Append one retrieval record (JSON line) to logs/retrieval_debug.jsonl. Internal only: this file is never sent
    to the technician's browser. RETRIEVAL_DEBUG=0 switches it off."""
    global _debug_logger
    if os.getenv("RETRIEVAL_DEBUG", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    try:
        if _debug_logger is None:
            DEBUG_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            logger = logging.getLogger("autoassist.retrieval_debug")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            if not logger.handlers:
                handler = RotatingFileHandler(DEBUG_LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)
            _debug_logger = logger
        _debug_logger.info(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **record},
                                      ensure_ascii=False, default=str))
    except Exception as exc:                                  # diagnostics must never break a chat
        log.warning("Could not write the retrieval debug log: %s", exc)


def _bm25(query_terms, documents_tokens, k1=BM25_K1, b=BM25_B):
    """Okapi BM25. Returns (scores, matched_term_counts, idf_coverage) aligned with documents_tokens.
    idf_coverage = share of the question's total IDF weight found in the chunk, so a hit on a rare
    term ('overheat') counts for much more than a hit on a common one ('engine')."""
    n_docs = len(documents_tokens)
    if not query_terms or n_docs == 0:
        return [0.0] * n_docs, [0] * n_docs, [0.0] * n_docs
    unique_terms = list(dict.fromkeys(query_terms))
    term_counts = [Counter(tokens) for tokens in documents_tokens]
    doc_lengths = [len(tokens) for tokens in documents_tokens]
    avg_length = (sum(doc_lengths) / n_docs) or 1.0
    df = {t: sum(1 for counts in term_counts if t in counts) for t in unique_terms}
    idf = {t: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    total_idf = sum(idf.values()) or 1.0
    scores, matched, coverage = [], [], []
    for counts, length in zip(term_counts, doc_lengths):
        score, hits, hit_idf = 0.0, 0, 0.0
        for t in unique_terms:
            tf = counts.get(t, 0)
            if not tf:
                continue
            hits += 1
            hit_idf += idf[t]
            score += idf[t] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avg_length))
        scores.append(score); matched.append(hits); coverage.append(hit_idf / total_idf)
    return scores, matched, coverage


def bm25_scores(query_terms, documents_tokens, k1=BM25_K1, b=BM25_B):
    """Compatibility wrapper: (scores, matched_term_counts)."""
    scores, matched, _ = _bm25(query_terms, documents_tokens, k1, b)
    return scores, matched


def reciprocal_rank_fusion(rankings, k=RRF_K, weights=None):
    """Merge several ranked lists of item ids (best first): score = sum of weight / (k + rank).
    Items ranked well by more than one retriever rise to the top; no score scaling is needed."""
    fused = {}
    for n, ranking in enumerate(rankings):
        weight = weights[n] if weights else 1.0
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + weight / (k + rank)
    return fused


_CODE_RE = re.compile(r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9]{2,}(?:-[A-Z0-9]+)*\b")


def find_codes(text):
    """Identifiers that mix letters and digits: P0300, 1GR-FE, 2GD-FTV, 48V, 4WD. Plain numbers are not codes."""
    return set(_CODE_RE.findall(normalize_units(text).upper()))


_token_cache = {"stamp": None, "tokens": []}
_dense_cache = {"index": None, "matrix": None, "subs": {}}


def _chunk_tokens(chunks):
    stamp = _cache["chunks"][0]
    if _token_cache["stamp"] != stamp or len(_token_cache["tokens"]) != len(chunks):
        _token_cache["tokens"] = [tokenize(f"{c.get('section', '')} {c.get('text', '')}") for c in chunks]
        _token_cache["stamp"] = stamp
    return _token_cache["tokens"]


def _dense_scores(chunks, eligible, query):
    """Cosine scores of the question against the eligible chunks, from a cached per-vehicle FAISS index.
    Returns a list aligned with eligible, or None when semantic search is unavailable."""
    if _recently_failed("embedding"):
        return None
    try:
        full = load_index()
        if full is None or full.ntotal != len(chunks) or not index_embedding_compatible():
            log.info("Rebuilding FAISS because its vectors do not match the chunks or embedding model.")
            save_knowledge_base(chunks)
            full = load_index()
        if _dense_cache["index"] is not full:
            _dense_cache.update(index=full, matrix=full.reconstruct_n(0, full.ntotal).astype("float32"), subs={})
        key = hash(tuple(eligible))
        sub = _dense_cache["subs"].get(key)
        if sub is None:
            sub = faiss.IndexFlatIP(_dense_cache["matrix"].shape[1])
            sub.add(np.ascontiguousarray(_dense_cache["matrix"][eligible]))
            if len(_dense_cache["subs"]) > 8:
                _dense_cache["subs"].clear()
            _dense_cache["subs"][key] = sub
        scores, positions = sub.search(encode([query]), len(eligible))
        dense = [0.0] * len(eligible)
        for score, position in zip(scores[0], positions[0]):
            if 0 <= int(position) < len(eligible):
                dense[int(position)] = max(0.0, float(score))
        return dense
    except Exception as exc:
        log.warning("Semantic search unavailable, using BM25 only: %s", exc)
        _model_failures["embedding"] = time.time()
        return None


def _shingles(text, size=5):
    tokens = normalize_text(text).split()
    return {" ".join(tokens[i:i + size]) for i in range(max(1, len(tokens) - size + 1))}


def _is_near_duplicate(shingles, others):
    """True when one passage is (almost) contained in a passage that was already chosen."""
    for other in others:
        smaller = min(len(shingles), len(other)) or 1
        if len(shingles & other) / smaller >= NEAR_DUPLICATE:
            return True
    return False


def _probability(scores):
    """CrossEncoder outputs -> 0..1. Models that already return probabilities are left alone; logits go through a sigmoid."""
    values = [float(x) for x in scores]
    if values and min(values) >= 0.0 and max(values) <= 1.0:
        return values
    return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, v)))) for v in values]


def rerank_text(chunk, limit=1800):
    section = chunk.get("section")
    text = f"{section}. {chunk.get('text', '')}" if section else chunk.get("text", "")
    return text[:limit]


def _ref(chunk, position, **extra):
    """Compact description of a chunk for the debug record."""
    return {"pos": position, "source": chunk.get("source"), "page": chunk.get("page"), "chunk": chunk.get("chunk"),
            "section": (chunk.get("section") or "")[:60], **extra}


def _other_vehicle_mentioned(question, chunks, model, generation):
    """Names of other vehicles the question mentions (diagnostics only; the vehicle filter is never relaxed)."""
    own = set(norm_key(model).split()) | set(norm_key(generation).split()) | {"fortuner", "gen"}
    names = set()
    for c in chunks:
        if c.get("applicability") == "vehicle_specific" and (norm_key(c.get("model")), norm_key(c.get("generation"))) != (norm_key(model), norm_key(generation)):
            names |= set(norm_key(c.get("model")).split())
    low = question.lower()
    return sorted(n for n in names - own if len(n) > 3 and re.search(r"\b" + re.escape(n) + r"\b", low))


def retrieve(question, model, generation, top_k=5, allowed_doc_ids=None, chunks=None, use_dense=True,
             use_rerank=None, reranker=None):
    """Filter by vehicle FIRST, then BM25 + FAISS fused by RRF, then (optionally) a CrossEncoder, then evidence selection.

    allowed_doc_ids: only chunks of these (indexed) documents may be used - the document-status filter.
    chunks / use_dense: used by the evaluation script to test other chunk stores or keyword-only mode.
    use_rerank: None = rerank when a CrossEncoder can be loaded, False = never (RRF order only), True = same as None.
    reranker: a CrossEncoder object to use instead of the configured one (evaluation of pretrained vs fine-tuned).
    Returns {"results", "strength": "strong" | "weak" | "none", "eligible", "dense_used", "reranked", "documents",
             "conflict_possible", "debug"}.  Every result keeps its source document name and page number.
    "debug" is for logs/retrieval_debug.jsonl only and is never shown to the technician.
    """
    custom = chunks is not None
    chunks = chunks if custom else load_chunks()
    dtc = is_dtc_question(question)
    eligible = [i for i, c in enumerate(chunks)
                if chunk_allowed(c, model, generation, dtc)
                and (allowed_doc_ids is None or c.get("doc_id") in allowed_doc_ids)]
    debug = {"search_query": question,
             "vehicle_filter": {"model": model, "generation": generation, "dtc_question": dtc,
                                "eligible_chunks": len(eligible), "total_chunks": len(chunks),
                                "documents": sorted(allowed_doc_ids) if allowed_doc_ids is not None else "all"},
             "other_vehicle_mentioned": _other_vehicle_mentioned(question, chunks, model, generation),
             "errors": []}
    empty = {"results": [], "strength": "none", "eligible": len(eligible), "dense_used": False, "reranked": False,
             "documents": [], "conflict_possible": False, "debug": debug}
    if not eligible:
        debug["errors"].append("no chunk is allowed for this vehicle / document selection")
        return empty
    subset = [chunks[i] for i in eligible]

    # --- Ranker 1: BM25 over the filtered chunks only -----------------------
    if custom:
        tokens_all = [tokenize(f"{c.get('section', '')} {c.get('text', '')}") for c in chunks]
    else:
        tokens_all = _chunk_tokens(chunks)
    terms = query_terms(question)
    bm25, _matched, coverage = _bm25(terms, [tokens_all[i] for i in eligible])
    bm25_all = sorted(range(len(subset)), key=lambda p: bm25[p], reverse=True)
    bm25_ranking = [p for p in bm25_all if bm25[p] > 0 and coverage[p] >= BM25_MIN_COVERAGE][:CANDIDATE_POOL]
    debug["query_terms"] = terms
    debug["bm25"] = [_ref(subset[p], p, rank=r, score=round(bm25[p], 3), coverage=round(coverage[p], 3))
                     for r, p in enumerate(bm25_ranking[:10], start=1)]

    # --- Ranker 2: semantic search (FAISS) over the filtered chunks only ----
    dense = _dense_scores(chunks, eligible, question) if (use_dense and not custom) else None
    dense_ranking = []
    if dense is not None:
        best = max(dense) if dense else 0.0
        dense_ranking = [p for p in sorted(range(len(subset)), key=lambda p: dense[p], reverse=True)
                         if dense[p] >= DENSE_FLOOR and dense[p] >= best - DENSE_BAND][:CANDIDATE_POOL]
        debug["faiss"] = [_ref(subset[p], p, rank=r, score=round(dense[p], 3)) for r, p in enumerate(dense_ranking[:10], start=1)]
    else:
        debug["faiss"] = []
        if use_dense and not custom:
            debug["errors"].append("semantic search (FAISS / embedding model) unavailable: BM25 only")

    # --- Ranker 3 (only when the question has a code such as P0300 / 1GR-FE) --
    rankings, weights = [bm25_ranking, dense_ranking], [1.0, 1.0]
    codes = find_codes(question)
    code_hit = {}
    if codes:
        for p in range(len(subset)):
            if codes & find_codes(subset[p].get("text", "")):
                code_hit[p] = True
        code_ranking = sorted(code_hit, key=lambda p: bm25[p], reverse=True)
        if code_ranking:
            rankings.append(code_ranking[:CANDIDATE_POOL]); weights.append(1.0)

    # --- Reciprocal Rank Fusion --------------------------------------------
    fused = reciprocal_rank_fusion(rankings, weights=weights)
    for p in fused:
        if subset[p].get("synthetic"):
            fused[p] *= SYNTHETIC_WEIGHT
    if not fused:
        debug["errors"].append("no passage passed the keyword / semantic relevance floors")
        return {**empty, "dense_used": dense is not None}
    ordered = sorted(fused, key=lambda p: fused[p], reverse=True)
    bm25_rank = {p: r for r, p in enumerate(bm25_ranking, start=1)}
    dense_rank = {p: r for r, p in enumerate(dense_ranking, start=1)}
    debug["rrf"] = [_ref(subset[p], p, rank=r, score=round(fused[p], 5), bm25_rank=bm25_rank.get(p), faiss_rank=dense_rank.get(p),
                         code_match=p in code_hit) for r, p in enumerate(ordered[:15], start=1)]

    # --- Second stage: CrossEncoder over the already filtered, fused candidates ------------------
    rerank_prob, order_score = {}, dict(fused)
    if reranker is None and use_rerank is not False and not custom:
        reranker = get_reranker()
    elif use_rerank is False:
        reranker = None
    if reranker is not None:
        candidates = ordered[:max(RERANK_TOP_N, top_k)]
        try:
            raw = reranker.predict([(question, rerank_text(subset[p])) for p in candidates], show_progress_bar=False)
            probs = _probability(raw)
            lo, hi = min(float(x) for x in raw), max(float(x) for x in raw)
            top_rrf = max(fused[p] for p in candidates) or 1.0
            for p, prob, value in zip(candidates, probs, raw):
                rerank_prob[p] = prob
                normal = (float(value) - lo) / (hi - lo) if hi > lo else 1.0
                order_score[p] = RERANK_WEIGHT * normal + (1.0 - RERANK_WEIGHT) * (fused[p] / top_rrf)
            ordered = sorted(candidates, key=lambda p: order_score[p], reverse=True)
            rrf_position = {p: n for n, p in enumerate(sorted(candidates, key=lambda x: fused[x], reverse=True), start=1)}
            debug["rerank"] = [_ref(subset[p], p, rank=r, probability=round(rerank_prob[p], 4), final=round(order_score[p], 4),
                                    rrf_rank=rrf_position[p]) for r, p in enumerate(ordered, start=1)]
        except Exception as exc:
            log.exception("CrossEncoder reranking failed; falling back to RRF order.")
            debug["errors"].append(f"CrossEncoder failed ({type(exc).__name__}): RRF order used")
            rerank_prob, order_score = {}, dict(fused)
    elif use_rerank is not False and not custom:
        debug["errors"].append("CrossEncoder unavailable: RRF order used")
    reranked = bool(rerank_prob)

    # --- Evidence selection: thresholds, duplicates, relative cut ------------------------------
    if reranked:
        dropped = [p for p in ordered if rerank_prob[p] < RERANK_DROP]
        ordered = [p for p in ordered if rerank_prob[p] >= RERANK_DROP]
        debug["dropped_below_rerank_threshold"] = [_ref(subset[p], p, probability=round(rerank_prob[p], 4)) for p in dropped[:10]]
    if ordered and EVIDENCE_REL_CUT > 0:
        best_score = order_score[ordered[0]]
        ordered = [p for p in ordered if order_score[p] >= EVIDENCE_REL_CUT * best_score]
    if not ordered:
        debug["errors"].append("every candidate was below the evidence threshold")
        return {**empty, "dense_used": dense is not None, "reranked": reranked}

    selected, seen_text, seen_shingles = [], set(), []
    for p in ordered:
        text = subset[p].get("text", "")
        fingerprint = normalize_text(text)[:300]
        shingles = _shingles(text)
        if fingerprint in seen_text or _is_near_duplicate(shingles, seen_shingles):
            continue                                   # same text already chosen (duplicate page / chunk)
        seen_text.add(fingerprint); seen_shingles.append(shingles)
        selected.append(p)
        if len(selected) >= max(1, top_k):
            break

    # Neighbouring chunk for the two best hits (a procedure may continue in the next chunk or on the next page),
    # and the WARNING / CAUTION chunk of the same page for the three best hits (safety text must not be lost).
    order = sorted(range(len(subset)), key=lambda p: (subset[p].get("source") or "", subset[p].get("page") or 0, subset[p].get("chunk") or 0))
    where = {p: n for n, p in enumerate(order)}
    by_page = {}
    for p in order:
        by_page.setdefault((subset[p].get("source"), subset[p].get("page")), []).append(p)
    chosen, companion = list(selected), {}
    for p in selected[:2]:
        a = subset[p]
        for delta in (-1, 1):
            n = where[p] + delta
            if not 0 <= n < len(order):
                continue
            q = order[n]
            b = subset[q]
            same_page = a.get("page") == b.get("page")
            continues = same_page or (abs((a.get("page") or 0) - (b.get("page") or 0)) == 1 and (delta == 1 or a.get("chunk") == 1))
            if (a.get("source") == b.get("source") and continues and q not in chosen
                    and normalize_text(b.get("text", ""))[:300] not in seen_text):
                chosen.append(q); companion[q] = "neighbour"
    for p in selected[:3]:
        for q in by_page.get((subset[p].get("source"), subset[p].get("page")), []):
            warn = subset[q].get("has_warning")
            warn = has_warning(subset[q].get("text", "")) if warn is None else warn
            if warn and q not in chosen and len([1 for v in companion.values() if v == "warning"]) < 2 \
                    and normalize_text(subset[q].get("text", ""))[:300] not in seen_text:
                chosen.append(q); companion[q] = "warning"

    def legacy_strong(p):
        return ((bm25_rank.get(p, 99) <= 5 and dense_rank.get(p, 99) <= 5)                   # both rankers agree
                or (dense is not None and dense[p] >= DENSE_STRONG)                           # clearly similar in meaning
                or (bm25_rank.get(p, 99) <= 3 and coverage[p] >= 0.6)                         # most key terms matched
                or (p in code_hit and bm25_rank.get(p, 99) <= 5))                             # exact code / part number

    results, strong_found = [], False
    for p in chosen:
        item = dict(subset[p])
        item["score"] = round(fused.get(p, 0.0), 5)
        item["bm25_rank"], item["semantic_rank"] = bm25_rank.get(p), dense_rank.get(p)
        item["bm25_coverage"] = round(coverage[p], 3)
        item["semantic_score"] = round(dense[p], 3) if dense is not None else None
        if p in rerank_prob:
            item["rerank_score"] = round(rerank_prob[p], 5)
        item["neighbour"] = p not in selected
        item["companion"] = companion.get(p)
        if item["neighbour"]:
            strong = False
        elif reranked:
            strong = rerank_prob.get(p, 0.0) >= RERANK_STRONG or (legacy_strong(p) and rerank_prob.get(p, 0.0) >= 0.10)
        else:
            strong = legacy_strong(p)
        item["relevance"] = "strong" if strong else "related"
        strong_found = strong_found or strong
        results.append(item)

    primary = [r for r in results if not r["neighbour"]]
    documents = sorted({r.get("source") for r in primary if r.get("source")})
    # Two different documents (e.g. owner's manual and service manual, or two editions) can disagree: flag it so the answer
    # states both instead of silently merging them.
    conflict = len({(r.get("source"), r.get("document_type"), r.get("year_range")) for r in primary if r["relevance"] == "strong"}) > 1
    debug["selected"] = [_ref(r, i, relevance=r["relevance"], neighbour=r["neighbour"], companion=r["companion"],
                              rrf=r["score"], rerank=r.get("rerank_score")) for i, r in enumerate(results)]
    debug["conflict_possible"] = conflict
    return {"results": results, "strength": "strong" if strong_found else "weak", "eligible": len(eligible),
            "dense_used": dense is not None, "reranked": reranked, "documents": documents,
            "conflict_possible": conflict, "debug": debug}


def search_manuals(question, model, generation, top_k=5, allowed_doc_ids=None):
    """Compatibility wrapper: just the list of retrieved chunks."""
    return retrieve(question, model, generation, top_k, allowed_doc_ids)["results"]


def available_pairs():
    """Distinct (model, generation) pairs that have vehicle-specific chunks."""
    return sorted({(c.get("model"), c.get("generation")) for c in load_chunks() if c.get("applicability") == "vehicle_specific"})


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == "rechunk":
        rechunk_all()
    else:
        print("Usage: python rag.py rechunk   (re-read every PDF with the current chunker and rebuild the index)")
