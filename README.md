# ⚖️ Agentic RAG Legal Assistant

Production-style **Agentic RAG** over a legal PDF (built for the US *Code of Federal Regulations*, e.g. 21 CFR), with a
**Script Studio** (video scripts from the PDF) and a **Voice Studio** (MP3 narration in a woman, man or child voice).

**Stack:** Python 3.11+ · Streamlit · LangGraph · Qdrant (local) · Groq · fastembed (free local embeddings) · PyMuPDF · SQLite · edge-tts

Every answer is grounded in retrieved passages and cited with **document, Title, Part, Section, page and revision date**.
The app never presents the document as current law - the revision date (e.g. *April 1, 1996*) is shown with every answer.

## Dashboard (Streamlit UI)

| Page | What you do there |
|---|---|
| 💬 **Chat** | Ask questions. Live agent steps, answer with `[S1]` markers, source cards, expandable evidence + raw metadata, saved conversations, export to Markdown. |
| 🎬 **Script Studio** | Enter a topic or a section (`§1303.13` or `1303.13`) and get a video script built only from the PDF, with citations and warnings. |
| 🔊 **Voice Studio** | Turn a PDF / TXT file, pasted text or the last script into an MP3. Choose language, speaker (Woman / Man / Child), voice and tone; preview before generating. |
| 📚 **Knowledge Base** | Ingest PDFs (progress bar + report), see the document list, delete a document, test retrieval in the *Retrieval playground*. |

**Sidebar (always visible):** page menu · **🔑 Groq API key** box (a key typed here overrides `.env`; kept only for the browser session) ·
status badges (Qdrant chunks, documents, key set / missing) · **⚙️ Settings** (Groq model, top-k, rewrite retries, restrict to a Part) ·
conversation list with New chat / Clear / delete / export.

## Architecture

````mermaid
flowchart LR
  subgraph Ingestion
    A["Legal PDF"] --> B["PyMuPDF parser: text, tables, images, OCR, layout"]
    B --> C["Legal structure tracker: Title, Chapter, Part, Subpart, Section, (a)(1)(i)"]
    C --> D["Structure-aware chunker"]
    D --> E["fastembed embeddings"]
    E --> F[("Qdrant: vector + rich metadata payload")]
  end
  subgraph Agent ["LangGraph agent"]
    Q["Question"] --> N1["Query analysis"]
    N1 --> N2["Retrieval tools: search_legal_documents, get_legal_section"]
    N2 --> N3{"Evidence valid?"}
    N3 -- "no, retries left" --> N4["Query rewrite"]
    N4 --> N2
    N3 -- "no, retries used" --> X["Insufficient-evidence reply"]
    N3 -- "yes" --> N5["Answer generation - Groq"]
    N5 --> N6["Deterministic citations + checks"]
  end
  subgraph Studio ["Script + Voice Studio"]
    S1["Topic or section"] --> S2["Script from exact section / top passages"]
    S2 --> S3["Cleaned narration text"]
    S4["PDF / TXT / pasted text"] --> S3
    S3 --> S5["Translate with Groq if not English"]
    S5 --> S6["edge-tts: voice + tone"]
    S6 --> S7["MP3"]
  end
  F <--> N2
  F <--> S2
  N6 --> UI["Streamlit dashboard"]
  X --> UI
  S7 --> UI
````

**Anti-hallucination design**
1. The LLM only sees numbered sources `[S1]..[Sn]` and must cite them.
2. Citations are built **in code** from Qdrant payloads (not by the LLM); markers pointing to non-existent sources are stripped.
3. Section numbers mentioned in an answer but absent from the evidence are flagged in the UI.
4. Chapter and Part names are passed to the model with each source, so it does not guess the agency.
5. If evidence is still insufficient after the retry loop, the app says so instead of guessing.
6. **Script Studio topic lock:** a named section is built from the chunks of exactly that section; script length is capped by the
   amount of source text (no padding); abbreviations that are not in the sources (e.g. a wrong agency name) are flagged.

## Metadata

### Chunk metadata (stored in Qdrant with every vector)

| Group | Fields |
|---|---|
| Document | `document_id`, `document_name`, `source_file`, `revision_date` |
| Legal hierarchy | `title`, `title_name`, `chapter`, `chapter_name`, `subchapter`, `part`, `part_name`, `subpart`, `subpart_name`, `section`, `section_heading`, `subsection`, `hierarchy_path` |
| Location / layout | `page`, `page_start`, `page_end`, `bbox` (x0, y0, x1, y1) |
| Content | `element_type` (text, table, image, ocr, footnote), `element_types`, `caption`, `image_paths`, `text`, `char_count` |
| Bookkeeping | `chunk_index`, `ingested_at` |

The embedded text is prefixed with `document | hierarchy path | section heading` (contextual chunk header).
The revision date is read from the PDF (`Revised as of April 1, 1996` or `(4-1-96 Edition)`); set `DEFAULT_REVISION_DATE` only as a fallback.

### Document registry (SQLite, `storage/app.db`)
`document_id`, `document_name`, `source_file`, `revision_date`, `page_count`, `chunk_count`, `ingested_at`, and the PDF's own metadata (title, author, dates).

### Answer metadata (saved with each chat message)
`citations` (marker, label, Title/Part/Section/page, revision date, score, text), `warnings`, `disclaimer`, `trace` (every agent step with timing), `status`.

### Voice metadata
Each MP3 is saved to `storage/audio/voiceover.mp3` and offered as `voiceover_<voice>.mp3`. The voice, speed %, pitch Hz and volume % used are shown in the Voice Studio caption.

## Voice Studio

Free text-to-speech with Microsoft Edge neural voices through `edge-tts` (no API key). Code: `app/voice.py`.

**Controls:** Language / accent · Speaker (**Woman**, **Man**, **Child**) · Voice · Tone · ▶ Preview.

| Tone | Speed | Pitch | Volume |
|---|---|---|---|
| Neutral | 0% | 0 Hz | 0% |
| Calm & soothing | -12% | -2 Hz | -5% |
| Warm & friendly | -4% | +2 Hz | 0% |
| Energetic | +15% | +4 Hz | +5% |
| Serious / authoritative | -8% | -6 Hz | +5% |
| News reader | +8% | 0 Hz | +5% |
| Slow & clear (learners) | -25% | 0 Hz | 0% |
| Cheerful | +10% | +8 Hz | +5% |
| Storyteller | -10% | -3 Hz | 0% |
| Custom | sliders | sliders | sliders |

**How it works:** text from a PDF / TXT file, pasted text or the last script → cleaned for narration (headings, timestamps,
"On screen" cues and `[S1]` markers removed, `§` read as "section") → translated by Groq when the voice language is not English →
split into parts → spoken part by part → joined into one MP3 you can play and download.

**Limits:** `edge-tts` changes only speed, pitch and volume, so tones are approximations, not real emotions. A real child voice exists
only for English (US); other languages imitate it by raising a woman's voice. One voice per file. Internet is required to create audio.

## Project layout

````
streamlit_app.py            dashboard entry point (Chat, Script Studio, Voice Studio, Knowledge Base)
app/
  config.py                 .env-driven settings (no secrets in code)
  models.py                 DocumentInfo / PageElement / Chunk
  embeddings.py             fastembed (+ offline HashingEmbedder for tests)
  vectorstore.py            Qdrant: collection, upsert, filtered search, section lookup
  storage.py                SQLite: conversations, messages, document registry
  llm.py                    Groq client (retry/backoff, JSON parsing, gpt-oss support)
  script_mode.py            Script Studio: topic lock, length cap, term check
  voice.py                  Voice Studio: text cleaning, translation, voices, tones, MP3
  ingestion/
    pdf_parser.py           text, headings, footnotes, captions, tables, images, OCR, layout (bbox)
    legal_structure.py      hierarchy detection incl. long run-in section headings (pure Python)
    chunker.py              structure-aware chunking + metadata
    pipeline.py             PDF -> chunks -> embeddings -> Qdrant
  agent/
    state.py prompts.py tools.py nodes.py graph.py citations.py
  ui/  styles.py components.py
scripts/ingest.py           CLI ingestion
scripts/make_sample_pdf.py  builds a small SYNTHETIC test PDF
data/sample/                synthetic sample PDF (not real law)
tests/                      pytest suite
````

## Setup (VS Code)

````bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env
# set GROQ_API_KEY in .env  (free key: https://console.groq.com/keys)  - or paste it in the sidebar later
streamlit run streamlit_app.py
````

The first start downloads the embedding model (~130 MB, cached afterwards).
Open the project folder that contains `streamlit_app.py` (unzipping can create a nested folder of the same name).

### Ingest your PDF
- **UI:** *Knowledge Base* page → upload the PDF → **Ingest**. Do not click another page until it finishes.
- **CLI (recommended for big PDFs):** `python scripts/ingest.py --pdf "data\pdfs\your_file.pdf" --name "Code of Federal Regulations - Title 21"`
  (add `--ocr` for scanned pages). Re-ingesting the same file replaces the old copy.

> Local Qdrant allows **one process at a time**. Stop Streamlit before using the CLI, or run Qdrant as a server:
> `docker compose up -d` and set `QDRANT_URL=http://localhost:6333` in `.env`.
> Data lives in `storage/qdrant`; delete that folder (and `storage/app.db`) to reset everything.

Quick smoke test without your own PDF: ingest `data/sample/sample_cfr_excerpt.pdf` (synthetic "Title 99") and ask
*"How long must batch records be kept?"* or *"What does § 900.4 say?"*.

### OCR (optional)
Install the Tesseract binary (`sudo apt install tesseract-ocr`, `brew install tesseract`, or the Windows installer), then set
`OCR_ENABLED=true` or tick the box on the Knowledge Base page.

## Groq model tips
- Default model is `openai/gpt-oss-20b` (light, fast). `openai/gpt-oss-120b` follows "use only the sources" more reliably for scripts.
  Groq's Llama models are Enterprise-only for many accounts; check https://console.groq.com/docs/models for the current list.
- gpt-oss models "think" first, so `GROQ_MAX_TOKENS=2000` and a low reasoning effort are used.
- Free tiers have token-per-minute limits; prompts are kept compact (`MAX_CONTEXT_CHARS`, `VALIDATION_SNIPPET_CHARS`, `TOP_K`) and 429s are retried.
- `MIN_SCORE` (default 0.35) is the similarity floor before the validator is asked; tune it if answers are refused too often or too rarely.

## Tests

````bash
pytest -v
````
`test_agent_graph.py` runs the whole LangGraph loop with in-memory Qdrant and a scripted fake LLM (no network, no key).
`test_parser.py` runs the PDF parser on the synthetic sample PDF.

## Known limits
- Table detection uses PyMuPDF `find_tables()`; complex borderless tables may come out as text.
- Two-column reading order is heuristic; check a few pages in the *Retrieval playground* after ingesting a new layout.
- Section headings are detected from font weight + numbering patterns; unusual formatting may need tweaks in `legal_structure.py`.
- Short sections give short scripts on purpose (no invented filler).
- Voice tones are approximations; real child voice is English only; voice creation needs internet.
- Informational research tool, not legal advice. Always verify against the current eCFR / Federal Register.
````
````

The README says the audio is saved at `storage/audio/voiceover.mp3`, and that is where the current `streamlit_app.py` writes it. If your `requirements.txt` doesn't list `edge-tts` yet, add the line `edge-tts>=6.1.0` to it so a fresh install gets the voice feature. The zip I built has all of this applied. 