# Toyota Fortuner AutoAssist: RAG-based vehicle troubleshooting assistant

A Flask web app where technicians describe a vehicle problem and get an AI answer built **only from the manuals of the
selected vehicle**, with the PDF name and page number shown as evidence. Engineers review and approve work, administrators
manage vehicles, manuals and users.

**Stack:** Python, Flask, Bootstrap 5, MySQL (SQLAlchemy), PyMuPDF, Sentence Transformers, FAISS, BM25 + Reciprocal Rank
Fusion, Llama 3.3 70B through OpenRouter.

---

## 1. What you need installed

| Program | Why |
|---|---|
| Python 3.10+ | runs the app |
| MySQL 8 (or MariaDB 10.5+) | stores users, vehicles, work orders, chat history, document records |
| Tesseract OCR *(optional)* | reads **scanned** PDFs. Without it a scanned PDF shows the status "Needs OCR" |
| Internet | Bootstrap loads from a CDN, the embedding model downloads once (~90 MB), and AI answers use OpenRouter |

## 2. Create the MySQL database

Open the MySQL command line (or MySQL Workbench) and run:

```sql
CREATE DATABASE autoassist CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'autoassist'@'localhost' IDENTIFIED BY 'choose-a-strong-password';
GRANT ALL PRIVILEGES ON autoassist.* TO 'autoassist'@'localhost';
FLUSH PRIVILEGES;
```

You do **not** create tables by hand. The app creates the required MySQL tables the first time it starts, including `otp_challenges`, and adds supported new columns
automatically if you already have a database from an earlier version.

## 3. Install the project

```bash
cd Toyota
python -m venv venv
venv\Scripts\activate          # Windows      (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## 4. Configure `.env`

Copy `.env.example` to a new file named `.env` and fill it in:

```
DB_HOST=localhost
DB_PORT=3306
DB_USER=autoassist
DB_PASSWORD=the-password-you-chose-above
DB_NAME=autoassist
SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
OPENROUTER_API_KEY=<your NEW key from openrouter.ai/keys>

# Gmail OTP (Google App Password; do not use your normal Gmail password)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=465
MAIL_USE_SSL=1
MAIL_USERNAME=<your Gmail address>
MAIL_PASSWORD=<your 16-character Google App Password>
MAIL_DEFAULT_SENDER=<your Gmail address>

# Optional: the app automatically detects these local folders after training
EMBEDDING_MODEL_PATH=
RERANKER_MODEL_PATH=
```

For Gmail, enable 2-Step Verification on the Google account and create an **App Password** for this application. Put that App Password in `MAIL_PASSWORD`. Never put your normal Gmail password in `.env`. Gmail OTP is sent after a correct password at login and during password reset. Each code is six digits, expires after **120 seconds**, is one-use, and allows at most five verification attempts. Resending invalidates the previous code and has a 30-second cooldown.

`.env` is in `.gitignore`. Never upload it or share it. If a real API key or mail App Password was included in an older ZIP, revoke it and create a new one.

## 5. Create the first administrator

**Easiest way:** start the app (step 6) and open **Create account**. While the system has no administrator, the page shows
"First-time setup" and the first account you create becomes the **active Administrator**. Sign-in still requires Gmail OTP verification.
After that, this shortcut is gone for good: every new account is pending until an administrator approves it.

**Command-line way (optional):** `flask --app app create-admin` asks for name, email, username and password.

## 6. Run

```bash
python app.py
```

Open http://127.0.0.1:5000 and sign in.

On the **first start**, the app finds the PDFs in `uploads/` and the ready-made index in `rag_store/`, links them to
MySQL records, and creates one vehicle per manual so you can try the chat immediately.

---

## 7. How people use it

* **New user:** *Create account* on the login page and choose a requested role (Technician, Maintenance Engineer or
  Administrator). This is only a **request**: the account stays **pending** and cannot sign in until an administrator opens
  *User Management*, sees the requested role, and presses *Approve* (the administrator can change the role first).
* **Existing user:** enter username/email and password. After the password is correct, AutoAssist sends a six-digit OTP to
  the account's registered Gmail address. Enter it within **2 minutes** to finish signing in. The OTP is one-use.
* **Forgot password:** enter the registered email, verify the emailed OTP within **2 minutes**, then set and confirm a new
  password. The new password is stored as a Werkzeug hash; the old password is not recoverable.
* **Administrator:** *Document Management* → choose the PDF(s) → for each one type the **car model**, **generation**, pick the
  **document type** and enter the **year** → *Upload and process*. Processing starts automatically and the status changes
  Pending → Processing → Indexed. Failed or scanned PDFs show the reason and can be reprocessed. The administrator also
  manages users, safety rules and settings, and can edit or delete registered vehicles.
* **Technician:** open **AutoAssist Chat** → pick the **car model**, then the **generation** from the two dropdowns (they list
  exactly the models/generations that have an indexed manual) → describe the problem → read the answer and its **evidence
  (one PDF name + page number)** → *Create work order*. Only the manuals of that exact model and generation are searched.
  **Every new login starts a fresh chat**: earlier questions are not shown again (they stay stored in the database).
* **Vehicle Registry** is a plain list of the vehicles the workshop has registered. Technicians and administrators can
  register a new vehicle there. It is **not** linked to the chat or to the manuals; a work order can optionally name a registered vehicle.
* **Engineer:** *Work Order Review* → open a case (the AI answer is attached) → set the risk level → for High risk, record the
  decision in *High-Risk Approval* → add diagnosis and verification → *Closed*. A closed case is added to repair history.

## 8. Project structure (and why each file exists)

```
Toyota/
├── app.py              Flask routes: login, roles, work orders, documents, users, start-up, "create-admin" command
├── models.py           SQLAlchemy models for users, OTP challenges, vehicles and workshop records
├── rag.py              PDF reading (+OCR, tables), chunking, FAISS index, BM25 + FAISS + RRF, CrossEncoder reranking, evidence selection, debug log
├── train_retrieval.py  Build training data from the eval labels, fine-tune the CrossEncoder, keep it only if it wins
├── llm.py              Follow-up rewriting, prompt, OpenRouter call, citation + value checks, full/partial/no-evidence answers
├── eval_retrieval.py   Baseline / RRF / +CrossEncoder / +fine-tuned evaluation on training/eval_questions.json
├── requirements.txt    Python packages
├── .env.example        Template for MySQL, Gmail OTP and OpenRouter settings
├── training/           eval_questions.json (hand-checked), triplets.csv (generated), chunk_catalog.csv, results
├── templates/          15 authentication, dashboard and role/workflow pages
├── static/style.css    Theme on top of Bootstrap 5
├── uploads/            ORIGINAL PDFs (kept after restart)
├── logs/               retrieval_debug.jsonl (created on first question; internal, not shown to technicians)
└── rag_store/          chunks.json (text + PDF name + page) and manuals.faiss (vectors)
```

The application keeps the original retrieval design (BM25 + FAISS + RRF) and adds two optional trained models: a fine-tuned
Sentence Transformer embedding model and a CrossEncoder reranker. Gmail OTP challenges are stored as hashes in MySQL;
raw OTP values are never saved.

**Where data lives:** MySQL = records (users, vehicles, document status/paths, work orders, chats, approvals, history,
settings). Disk = PDFs (`uploads/`) and search data (`rag_store/`). No PDFs or vectors are stored in MySQL.

## 9. Database tables

`users` · `otp_challenges` (hashed one-use OTP, expiry and attempt count) · `vehicles` · `documents` (path, SHA-256 hash, status, page and chunk counts) · `work_orders` · `order_log`
(activity trail) · `approvals` · `repair_history` · `chat_history` · `system_settings` · `fault_code_files` · `fault_codes`.
Foreign keys: work orders → vehicles/users/chat; approvals → work orders/engineers; history → vehicles/work orders;
chats → users/vehicles; documents → uploader.

A document is tied to the **model + generation** typed at upload (spelling differences like `fortuner hilux` / `Fortuner Hilux`
are merged into one dropdown entry). Chats, work orders and repair history store model + generation too; a registered vehicle is optional.

## 10. Safety and retrieval rules (unchanged in spirit)

* The vehicle's exact **model and generation filter the chunks before any ranking**. Another vehicle's manual is never used.
* A *general DTC reference* is used only for fault-code questions, and never on its own as an answer.
* If nothing matches, the app says so instead of guessing.
* Every chunk keeps its PDF filename and page number; the single best-supporting one is shown under the answer.
* Safety rules edited by the administrator are sent to the AI with every question.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| "Could not connect to MySQL" at start | Is MySQL running? Check `DB_*` values in `.env`; the database must exist |
| Chat says no API key | Set `OPENROUTER_API_KEY` in `.env`, restart Flask |
| PDF shows **Needs OCR** | It is a scan. Install Tesseract, restart, press *Reprocess* |
| PDF shows **Failed** | Read the reason under the status. Corrupt, password-protected or empty PDFs cannot be indexed |
| "already uploaded as …" | The same file (same content) exists. Duplicates are blocked by hash |
| Everyone is logged out after restart | `SECRET_KEY` is missing from `.env` |
| Index and database disagree (e.g. you recreated the database) | Automatic: on start-up the app links existing PDFs back to MySQL, removes search data of missing PDFs and never keeps duplicates |
| App log | `app.log` in the project folder |

## 12. Known limitations (be honest with your guide)

* While no administrator exists, anyone who can reach the sign-up page can claim the first administrator account. Create yours
  right after installing, before sharing the address.
* Gmail OTP requires a working Gmail App Password in `.env` and internet access from the Flask server. Email delivery and the real MySQL database must be tested in the local environment.
* Training does not create labels automatically. A person must review `training/chunk_catalog.csv` and add truthful question/positive/negative rows to `training/triplets.csv` before model fine-tuning.
* PDF processing runs in a background thread of the Flask process, not a job queue. Fine for one admin, not for many.
* If a restart happens mid-processing, that document is marked Failed and must be reprocessed.
* No Content-Security-Policy header. Use HTTPS and set `SESSION_COOKIE_SECURE=1` before any real deployment.
* `Fortuner_Hilux_2026.pdf` is Toyota's **Emergency Response Guide** (first responders), not a repair manual, so troubleshooting
  answers for that vehicle will be limited. Upload the real service manual for better results.


## 13. Retrieval pipeline and training

**Pipeline for every question:** (1) follow-ups are rewritten with the recent chat, keeping fault codes and engine codes;
(2) chunks are filtered by the selected vehicle model + generation and the documents indexed for it; (3) BM25 and FAISS
each give up to 30 candidates, fused by RRF (an exact fault code / part number adds a third ranker); (4) the top 20 are scored by a
CrossEncoder (the pretrained `cross-encoder/ms-marco-MiniLM-L-6-v2`, or your fine-tuned one if it beat it) and blended with the RRF
rank; (5) weak and duplicate passages are removed, the neighbouring chunk and the page's WARNING/CAUTION chunk are attached;
(6) the LLM answers only from those passages; every `[S#]` citation and every number/code in the answer is checked against them.
If the CrossEncoder or the embedding model cannot be loaded (no internet on first run), the app keeps working with the remaining
rankers and says so in the debug log. The CrossEncoder model downloads once (~90 MB).

**Training the reranker** (needs `pip install -r requirements.txt` and internet for the first download):

```bash
python train_retrieval.py export        # chunk catalog with stable ids
python train_retrieval.py build-data    # query / positive / hard-negative rows from training/eval_questions.json
python train_retrieval.py validate      # vehicle match + no question in two splits
python train_retrieval.py train         # fine-tune, compare with the pretrained model on TEST questions, keep the better one
python train_retrieval.py evaluate      # pretrained vs fine-tuned on the held-out test questions
```

Splits are fixed per question id (about 60/15/25 %), so a test question is never trained on. A fine-tuned reranker is kept only if
its MRR beats the pretrained one by at least 0.01 on the test questions with no wrong-vehicle retrieval; otherwise it is discarded.
The data is small (40 train questions), so treat the result as indicative. Fine-tune the embedding model (`train --embedding`) only if
`python eval_retrieval.py` shows that the right page is missing from the candidates, not just ranked low. If you re-chunk the PDFs
(`python rag.py rechunk`), run `export` and `build-data` again. After retraining, run `python eval_retrieval.py`.

## 14. Evaluation and diagnostics

```
python eval_retrieval.py                 # baseline | BM25+FAISS+RRF | + pretrained CE | + fine-tuned CE
python eval_retrieval.py --split test    # only held-out questions
python eval_retrieval.py --sweep         # thresholds (re-check RERANK_DROP / RERANK_STRONG here with the real model)
python eval_retrieval.py --keyword --rechunk   # compare the current chunker with the shipped index (no models needed)
```

A configuration whose model cannot be loaded is printed as NOT RUN. Results and the list of retrieval failures are saved to
`training/retrieval_eval_results.json`. **Check on your machine:** the shipped results were produced without FAISS and CrossEncoder
(keyword-only), so run the command once with internet access.

**Debug log:** each chat turn appends one JSON line to `logs/retrieval_debug.jsonl` (original and rewritten question, vehicle filter,
BM25 / FAISS / RRF ranks, reranker scores, selected evidence, citation checks, answer, errors). It rotates at 5 MB and is
never sent to the technician's browser. Set `RETRIEVAL_DEBUG=0` to switch it off.

**Chunking:** new uploads keep tables as `label | value` rows, keep WARNING/CAUTION text with its steps, and split long
procedures at step borders with overlap. In a keyword-only test this did not beat the shipped index (Hit@5 0.973 vs 0.986), so the
shipped `rag_store/` was **not** rebuilt. Rebuild (`python rag.py rechunk`, several minutes) only if
`python eval_retrieval.py --rechunk` with FAISS shows it wins.

## Updates: fault codes and the new work order flow

**Technician**
- *Fault Codes* (sidebar): type a code such as `P0171` (or a keyword like `misfire`), optionally pick a vehicle, and get the meaning, possible causes and recommended checks from the lists the administrator uploaded.
- *New work order / Update your findings*: Case status (Open, Under review, In progress, On hold, Ready for engineering review), Findings, Actions taken, Parts used, Hours spent. A technician cannot approve or close a work order. A High-risk case cannot go to *In progress* until it is approved.

**Maintenance Engineer**
- *Work Order Review*: open a case to see the technician's entry, then set the **Risk classification** (Low / Medium / High) and write the **Review findings / Evidence gap**. Low and Medium go to *Diagnosis and Closure*; High goes to *High-Risk Approval*.
- *High-Risk Approval*: click a work order to open the pop-up with **Action scope**, **Review notes** and the buttons *Record approval*, *Reject*, *Return for evidence* (the case goes back to the technician, who adds evidence and resubmits).
- *Diagnosis and Closure*: click a work order to open the pop-up with **Final diagnosis**, **Verification / observed result**, **Outcome / additional notes** and the buttons *Save diagnosis* and *Close work order*. Closing needs a diagnosis and verification, and a High-risk case must be approved first. Closed cases are added to repair history.

**Administrator**
- *Fault Code Management*: upload CSV or PDF fault code lists (a CSV template can be downloaded on that page). CSV columns: `code`, `description`, optional `system`, `causes`, `action`, `model`, `generation`. In a PDF, each entry starts on a line with the code. Lists can apply to all vehicles or to one model/generation.

The database upgrades itself on start: two new columns on `work_orders` (`actions_taken`, `review_gap`) and two new tables (`fault_code_files`, `fault_codes`).


## 15. Earlier retrieval changes (conversation-aware RAG)

**What changed and why**

| File | Change | Why |
|---|---|---|
| `rag.py` | Stemming + technician synonyms for BM25 (`overheating` = `overheats`, `tyre` = `tire`, `handbrake` = `parking brake`); codes (`P0300`, `1GR-FE`, `12V`) kept whole | Keyword search no longer misses a page because of word endings or British wording |
| `rag.py` | The old hard gates (30 % of words must match, cosine >= 0.28) are replaced by an IDF-weighted coverage floor (0.20) and a cosine floor (0.22) plus a "within 0.20 of the best" band | Conversational questions no longer return "no match" when the manual has the answer. `python eval_retrieval.py --sweep` shows how each value behaves |
| `rag.py` | Per-vehicle FAISS index is cached; weighted RRF; duplicate chunks removed; neighbour chunk (same or next page) added for the two best hits | Faster, no duplicates, procedures that continue on the next page are not cut off |
| `rag.py` | `retrieve()` returns each chunk's relevance (`strong` / `related`) and the overall strength | `llm.py` and the evaluation use the same rule |
| `rag.py` | Section-aware chunking (title of the section is stored in every chunk, numbered steps are not split), image-only pages are OCR'd, running footers removed | Better chunks and metadata. Applies to new uploads and to `python rag.py rechunk` |
| `llm.py` | Follow-up questions are rewritten with the last 3 exchanges BEFORE retrieval (LLM first, offline fallback if the API fails) | "Why does it happen?" becomes "Why does the Toyota Fortuner engine overheat?" |
| `llm.py` | New prompt: Problem understanding, Possible causes, Inspection steps, Troubleshooting procedure, Safety precautions, Supporting evidence. The model also reports FULL / PARTIAL / NONE | Case 1 (documents answer it), Case 2 (partly, with "Not found in the documents"), Case 3 (not at all) |
| `llm.py` | Case 3 shows "General AI Knowledge - Not verified against the uploaded Toyota documents", no citations, and no procedures for safety-critical work | General knowledge is never presented as Toyota guidance |
| `app.py` | `/api/ask` passes the recent chat history and only the INDEXED documents of the selected model + generation | Follow-up understanding and the document-status filter. No table or route was changed |
| `templates/chat.html` | New answer modes are not shown as errors; up to 3 evidence pages | UI matches the new answers |

**Check the retrieval yourself** (no MySQL or API key needed):

```
python eval_retrieval.py --rechunk      # old vs new retrieval; uses the embedding model if it can be loaded
python eval_retrieval.py --sweep        # test the threshold values
python eval_retrieval.py --llm-rewrite  # also use the LLM to rewrite follow-ups (needs OPENROUTER_API_KEY)
```

The 83 questions in `training/eval_questions.json` have hand-checked pages. Add your own questions in the same format
and re-run. Optional: `python rag.py rechunk` rebuilds the index with the new chunker (re-embeds every chunk, a few
minutes on CPU). Run the evaluation with and without it and keep the better one.
