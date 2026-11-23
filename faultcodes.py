"""Fault-code (DTC) lists: read a CSV or PDF the administrator uploads and turn it into rows for the fault_codes table.

CSV: one row per code. The header names are flexible (see ALIASES); only a code column and a description column are required.
PDF: every line that starts with a code such as P0171, B1A23, C1201 or U0100 begins an entry; the lines below it belong
to that entry until the next code.
"""
import csv
import io
import re

import pymupdf

CODE_RE = re.compile(r"[PBCU][0-9][0-9A-F]{3}(?:[-\s]?[0-9A-F]{2})?", re.IGNORECASE)   # P0171, B1A23 or P0171-00
LINE_START_RE = re.compile(r"^\s*([PBCU][0-9][0-9A-F]{3}(?:[-\s]?[0-9A-F]{2})?)\b[\s:.\-–—|]*(.*)$", re.IGNORECASE)
MAX_ROWS = 50000

ALIASES = {
    "code": {"code", "dtc", "dtccode", "faultcode", "troublecode", "errorcode", "diagnosticcode", "dtcno", "codeno"},
    "description": {"description", "meaning", "title", "faultdescription", "dtcdescription", "name", "definition", "detail", "details", "fault"},
    "system": {"system", "category", "component", "area", "module", "subsystem"},
    "causes": {"causes", "cause", "possiblecauses", "possiblecause", "probablecause", "probablecauses", "detectioncondition", "troubleareas", "troublearea"},
    "action": {"action", "actions", "solution", "remedy", "fix", "recommendedaction", "recommendation", "repair", "procedure", "checks", "nextstep", "nextsteps"},
    "model": {"model", "vehicle", "vehiclemodel"},
    "generation": {"generation", "gen"},
}


def normalize_code(value):
    """'p0171', 'P 0171' and 'P0171-00' all become 'P0171' (the manufacturer sub-code after the dash is dropped)."""
    text = re.sub(r"\s+", "", str(value or "")).upper()
    match = re.match(r"^([PBCU][0-9][0-9A-F]{3})(?:-?[0-9A-F]{2})?$", text)
    return match.group(1) if match else ""


def _key(header):
    return re.sub(r"[^a-z]", "", (header or "").lower())


def _decode(data):
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def parse_csv(data):
    """Returns (rows, error). Each row is a dict with code, description, system, causes, action, model, generation."""
    text = _decode(data)
    if not text.strip():
        return [], "The CSV file is empty."
    first_line = text.splitlines()[0]
    delimiter = max((",", ";", "\t", "|"), key=first_line.count)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        return [], "The CSV file is empty."
    columns = {}
    for index, name in enumerate(header):
        for field, names in ALIASES.items():
            if _key(name) in names and field not in columns:
                columns[field] = index
    if "code" not in columns or "description" not in columns:
        return [], ("The first row must be a header with at least a 'code' column and a 'description' column "
                    "(optional: system, causes, action, model, generation).")
    rows, skipped = [], 0
    for line in reader:
        if not any((cell or "").strip() for cell in line):
            continue
        pick = lambda field: (line[columns[field]].strip() if field in columns and columns[field] < len(line) else "")
        code = normalize_code(pick("code"))
        description = pick("description")
        if not code or not description:
            skipped += 1
            continue
        rows.append({"code": code, "description": description[:4000], "system": pick("system")[:120],
                     "causes": pick("causes")[:4000], "action": pick("action")[:4000],
                     "model": pick("model")[:80], "generation": pick("generation")[:40], "page": None})
        if len(rows) >= MAX_ROWS:
            break
    if not rows:
        return [], "No valid rows found. Each row needs a code such as P0171 and a description."
    return rows, ("" if not skipped else f"{skipped} row(s) without a valid code or description were skipped.")


def parse_pdf(data):
    """Returns (rows, error). Reads the text of every page and splits it at the lines that start with a code."""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        return [], "The PDF could not be opened (corrupt or password-protected)."
    entries, current = [], None
    with doc:
        for page in doc:
            for line in page.get_text("text").splitlines():
                match = LINE_START_RE.match(line)
                if match and normalize_code(match.group(1)):
                    current = {"code": normalize_code(match.group(1)), "lines": [match.group(2).strip()], "page": page.number + 1}
                    entries.append(current)
                elif current is not None and line.strip():
                    current["lines"].append(line.strip())
    rows = []
    for entry in entries:
        text = re.sub(r"\s+", " ", " ".join(part for part in entry["lines"] if part)).strip()
        if len(text) < 3:
            continue
        rows.append({"code": entry["code"], "description": text[:2000], "system": "", "causes": "", "action": "",
                     "model": "", "generation": "", "page": entry["page"]})
        if len(rows) >= MAX_ROWS:
            break
    if not rows:
        return [], ("No fault codes were found. The PDF needs lines that start with a code such as P0171. "
                    "A scanned PDF has no readable text; use a CSV instead.")
    return rows, ""


SAMPLE_CSV = (
    "code,description,system,causes,action,model,generation\n"
    "P0171,System too lean (Bank 1),Fuel and air,\"Vacuum leak; dirty MAF sensor; weak fuel pump\","
    "\"Check intake hoses for leaks; inspect the MAF sensor; test fuel pressure\",,\n"
    "P0300,Random or multiple cylinder misfire detected,Ignition,\"Worn spark plugs; faulty ignition coil\","
    "\"Inspect spark plugs and coils; check compression\",,\n"
)
