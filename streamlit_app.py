"""Agentic RAG Legal Assistant - Streamlit UI.   Run:  streamlit run streamlit_app.py"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Legal Agentic RAG", page_icon="\u2696\uFE0F", layout="wide", initial_sidebar_state="expanded")

from app.agent.graph import build_graph, initial_state, stream_agent  # noqa: E402
from app.agent.nodes import NODE_LABELS  # noqa: E402
from app.config import ConfigError, get_settings, setup_logging  # noqa: E402
from app.embeddings import FastEmbedEmbedder  # noqa: E402
from app.ingestion.pipeline import ingest_pdf  # noqa: E402
from app.llm import GroqLLM, LLMError  # noqa: E402
from app.script_mode import generate_script  # noqa: E402
from app.storage import Storage  # noqa: E402
from app.ui.components import ICONS, hero, inject_css, pill, render_answer_extras, render_citations, typewriter  # noqa: E402
from app.vectorstore import LegalVectorStore  # noqa: E402
from app.voice import extract_upload_text, render_voice_controls, synthesize_long, translate_text  # noqa: E402

setup_logging()
inject_css()
SETTINGS = get_settings()


# ------------------------------------------------------------------ cached runtime
@st.cache_resource(show_spinner="Loading embedding model and vector store (first run downloads ~130 MB)...")
def get_core():
    embedder = FastEmbedEmbedder(SETTINGS.embedding_model)
    store = LegalVectorStore(SETTINGS, embedder)
    store.ensure_collection()
    return embedder, store, Storage(SETTINGS.sqlite_path)


@st.cache_resource(show_spinner=False)
def _get_llm(model: str, api_key: str):
    # the key typed in the sidebar (or the one from .env) is used for this connection
    return GroqLLM(replace(SETTINGS, groq_api_key=api_key), model=model)


@st.cache_resource(show_spinner=False)
def _get_graph(model: str, api_key: str):
    _, store, _ = get_core()
    return build_graph(store, _get_llm(model, api_key), SETTINGS)


def get_llm(model: str):
    return _get_llm(model, st.session_state.get("effective_key", ""))


def get_graph(model: str):
    return _get_graph(model, st.session_state.get("effective_key", ""))


try:
    _, STORE, STORAGE = get_core()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not start the vector store: {exc}")
    st.info("If you see a lock error, another process (e.g. the ingest CLI) is using the local Qdrant folder. "
            "Stop it, or run Qdrant via Docker and set QDRANT_URL.")
    st.stop()

ss = st.session_state
ss.setdefault("conv_id", None)
ss.setdefault("messages", [])
ss.setdefault("script_result", None)
ss.setdefault("voice_text", "")
ss.setdefault("voice_audio", None)
ss.setdefault("voice_filename", "voiceover.mp3")
ss.setdefault("effective_key", SETTINGS.groq_api_key)


# ---------------------------------------------------------------------- sidebar
def sidebar() -> tuple[str, str, int, int, str]:
    with st.sidebar:
        st.markdown("### \u2696\uFE0F Legal Agentic RAG")
        page = st.radio(
            "Navigate",
            ["\U0001F4AC Chat", "\U0001F3AC Script Studio", "\U0001F50A Voice Studio", "\U0001F4DA Knowledge Base"],
            label_visibility="collapsed",
        )

        # ---- Groq API key (typed key wins over the .env key)
        with st.expander("\U0001F511 Groq API key", expanded=not SETTINGS.has_groq_key):
            typed = st.text_input(
                "Paste your Groq API key", type="password", key="api_key_input", placeholder="gsk_...",
                help="Kept only in this browser session. It is not saved to disk.",
            ).strip()
            if typed and not typed.startswith("gsk_"):
                st.warning("Groq keys usually start with gsk_ - check that you copied the whole key.")
            if not typed and SETTINGS.has_groq_key:
                st.caption("Using the key from your .env file.")
            st.markdown("[Get a free key](https://console.groq.com/keys)")
        api_key = typed or SETTINGS.groq_api_key
        has_key = bool(api_key) and not api_key.lower().startswith("your_")
        ss.effective_key = api_key if has_key else ""

        n_chunks = STORE.count()
        docs = STORAGE.list_documents()
        st.markdown(
            pill(f"Qdrant: {n_chunks:,} chunks", "ok" if n_chunks else "warn")
            + pill(f"{len(docs)} document(s)", "ok" if docs else "warn")
            + pill("Groq key set" if has_key else "Groq key missing", "ok" if has_key else "bad"),
            unsafe_allow_html=True,
        )

        with st.expander("\u2699\uFE0F Settings", expanded=False):
            # The model box always starts from GROQ_MODEL in .env (e.g. openai/gpt-oss-20b).
            model = st.text_input("Groq model", SETTINGS.groq_model, key="groq_model_input").strip() or SETTINGS.groq_model
            st.caption("Examples: openai/gpt-oss-20b (fast) or openai/gpt-oss-120b (stronger)")
            top_k = st.slider("Passages retrieved (top-k)", 2, 12, SETTINGS.top_k)
            retries = st.slider("Max query-rewrite retries", 0, 3, SETTINGS.max_retries)
            part = st.text_input("Restrict to Part (optional)", placeholder="e.g. 101")

        if page == "\U0001F4AC Chat":
            st.markdown("---")
            c1, c2 = st.columns(2)
            if c1.button("\u2795 New chat", use_container_width=True):
                ss.conv_id, ss.messages = None, []
                st.rerun()
            if c2.button("\U0001F9F9 Clear", use_container_width=True, disabled=not ss.messages):
                if ss.conv_id:
                    STORAGE.delete_conversation(ss.conv_id)
                ss.conv_id, ss.messages = None, []
                st.rerun()
            if ss.messages:
                md = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in ss.messages)
                st.download_button("\u2B07\uFE0F Export chat (.md)", md, file_name="legal-chat.md", use_container_width=True)
            st.caption("Recent conversations")
            for conv in STORAGE.list_conversations(12):
                cols = st.columns([5, 1])
                if cols[0].button(conv["title"] or "Untitled", key=f"open-{conv['id']}"):
                    ss.conv_id, ss.messages = conv["id"], STORAGE.get_messages(conv["id"])
                    st.rerun()
                if cols[1].button("\U0001F5D1", key=f"del-{conv['id']}"):
                    STORAGE.delete_conversation(conv["id"])
                    if ss.conv_id == conv["id"]:
                        ss.conv_id, ss.messages = None, []
                    st.rerun()
    return page, model, top_k, retries, part.strip()


# --------------------------------------------------------------------- chat page
def chat_page(model: str, top_k: int, retries: int, part: str) -> None:
    hero("Agentic RAG Legal Assistant", "Grounded answers with Title / Part / Section / page / revision-date citations. "
         "Every answer is checked against retrieved evidence.")
    if STORE.count() == 0:
        st.info("The knowledge base is empty. Open **\U0001F4DA Knowledge Base** in the sidebar and ingest a PDF first.")

    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"], avatar="\U0001F9D1\u200D\u2696\uFE0F" if m["role"] == "user" else "\u2696\uFE0F"):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_answer_extras(m.get("meta", {}), f"h{i}")

    prompt = st.chat_input("Ask about the regulations, e.g. \"What does \u00a7 101.9 require?\"", disabled=STORE.count() == 0)
    if not prompt:
        return

    try:
        graph = get_graph(model)
    except (ConfigError, LLMError) as exc:
        st.error(f"{exc}  \u2192 Paste your key in the sidebar under \U0001F511 Groq API key.")
        return

    if ss.conv_id is None:
        ss.conv_id = STORAGE.create_conversation(prompt[:60])
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages[-6:]]
    ss.messages.append({"role": "user", "content": prompt, "meta": {}})
    STORAGE.add_message(ss.conv_id, "user", prompt)
    with st.chat_message("user", avatar="\U0001F9D1\u200D\u2696\uFE0F"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="\u2696\uFE0F"):
        final = initial_state(prompt, history, top_k, retries, part)
        try:
            with st.status("\U0001F916 Agent is working...", expanded=True) as status:
                for node, update in stream_agent(graph, final, ss.conv_id):
                    final.update(update)
                    ev = (update.get("trace") or [{}])[-1]
                    icon = ICONS.get(node, "\u2022")
                    label = NODE_LABELS.get(node, node)
                    status.write(f"{icon} **{label}** \u2014 {ev.get('detail', '')}")
                    status.update(label=f"\U0001F916 {label} done...")
                ok = final.get("status") in ("answered", "direct")
                status.update(label="Done" if ok else "Finished with warnings", state="complete" if ok else "error", expanded=False)
        except (LLMError, ConfigError) as exc:
            st.error(f"\u274C {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error("Something went wrong while running the agent.")
            with st.expander("Error details"):
                st.exception(exc)
            return

        answer = final.get("answer", "")
        st.write_stream(typewriter(answer))
        meta = {k: final.get(k) for k in ("citations", "warnings", "disclaimer", "trace", "status")}
        render_answer_extras(meta, "new")

    ss.messages.append({"role": "assistant", "content": answer, "meta": meta})
    STORAGE.add_message(ss.conv_id, "assistant", answer, meta)


# ------------------------------------------------------------------ script studio
def script_page(model: str) -> None:
    hero("Script Studio", "Generate a ~4-minute video script from the document only. Nothing is added from outside the PDF.")
    if STORE.count() == 0:
        st.info("Ingest a PDF first (Knowledge Base page).")
        return
    c1, c2 = st.columns([3, 1])
    topic = c1.text_input("Topic", placeholder="e.g. Nutrition labeling requirements")
    minutes = c2.number_input("Minutes", 1.0, 10.0, 4.0, 0.5)
    tone = st.selectbox("Tone", ["clear and engaging", "formal and precise", "friendly explainer", "news-style"])
    if st.button("\U0001F3AC Generate script", type="primary", disabled=not topic.strip()):
        try:
            llm = get_llm(model)
            with st.spinner("Searching the document and writing the script..."):
                ss.script_result = generate_script(topic.strip(), STORE, llm, SETTINGS, minutes, tone)
        except (ConfigError, LLMError) as exc:
            st.error(f"{exc}  \u2192 Paste your key in the sidebar under \U0001F511 Groq API key.")
    res = ss.script_result
    if res:
        if res.get("error"):
            st.warning(res["error"])
            return
        st.caption(f"{res['words']} spoken words \u2248 {res['est_minutes']} min (target {res['target_words']})")
        for w in res.get("warnings", []):
            st.warning(w)
        st.markdown(res["script"])
        st.download_button("\u2B07\uFE0F Download script (.md)", res["script"], file_name="script.md")
        render_citations(res["citations"], "script")
        if res.get("disclaimer"):
            st.markdown(f'<div class="disclaimer">{res["disclaimer"]}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ voice studio
LANG_NAMES = {
    "en": "English", "ur": "Urdu", "hi": "Hindi", "ar": "Arabic", "es": "Spanish", "fr": "French",
    "de": "German", "pt": "Portuguese", "it": "Italian", "tr": "Turkish", "fa": "Persian",
    "bn": "Bengali", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ru": "Russian",
}


def voice_page(model: str) -> None:
    hero("Voice Studio", "Turn a PDF, a text file, pasted text, or your generated script into an MP3 narration "
         "with a woman, man or child voice and the tone you choose.")

    st.markdown("**1. Choose where the text comes from** (or type directly in the box)")
    t1, t2 = st.tabs(["\U0001F4C4 PDF / TXT file", "\U0001F3AC Last generated script"])

    with t1:
        up = st.file_uploader("PDF or text file", type=["pdf", "txt"], key="voice_pdf")
        p1, p2 = st.columns(2)
        first = p1.number_input("From page (PDF only)", 1, 5000, 1)
        last = p2.number_input("To page (0 = end)", 0, 5000, 0)
        if st.button("Extract text", disabled=up is None):
            try:
                txt = extract_upload_text(up.name, up.getvalue(), int(first), int(last) or None)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read the file: {exc}")
            else:
                if not txt:
                    st.warning("No text found. This PDF may be scanned (images only) and needs OCR first.")
                ss.voice_text = txt

    with t2:
        res = ss.script_result
        if res and res.get("script"):
            if st.button("Use last script"):
                ss.voice_text = res["script"]
        else:
            st.info("No script yet. Generate one in Script Studio first.")

    text = st.text_area("Text to narrate (you can edit it)", key="voice_text", height=260)
    n_words = len(text.split())
    st.caption(f"{n_words:,} words \u2248 {n_words / 150:.1f} min of audio")

    st.markdown("**2. Choose the voice and tone**")
    v = render_voice_controls(key="voice")      # Language, Speaker (Woman / Man / Child), Voice, Tone, Preview
    voice = v["voice"]
    lang = voice.split("-")[0]

    translate = False
    if lang != "en":
        translate = st.checkbox(f"Translate the text to {LANG_NAMES.get(lang, lang)} first (uses the LLM)", value=True)

    st.markdown("**3. Generate**")
    if st.button("\U0001F50A Generate voice", type="primary", disabled=not text.strip()):
        ss.voice_audio = None
        try:
            final_text = text
            if translate:
                llm = get_llm(model)
                bar_t = st.progress(0.0, text="Translating...")
                final_text = translate_text(
                    text, LANG_NAMES.get(lang, lang), llm,
                    progress=lambda d, t: bar_t.progress(d / t, text=f"Translating part {d}/{t}"),
                )
                with st.expander("Translated text"):
                    st.write(final_text)
            bar = st.progress(0.0, text="Creating audio...")
            out_file = SETTINGS.images_dir.parent / "audio" / "voiceover.mp3"
            path = synthesize_long(
                final_text, out_file, voice, v["rate_pct"], v["pitch_hz"],
                progress=lambda d, t: bar.progress(d / t, text=f"Audio part {d}/{t}"),
                volume_pct=v["volume_pct"],
            )
            ss.voice_audio = path.read_bytes()
            ss.voice_filename = f"voiceover_{voice}.mp3"
            bar.progress(1.0, text="Voice is ready")
        except (ConfigError, LLMError) as exc:
            st.error(f"{exc}  \u2192 Paste your key in the sidebar under \U0001F511 Groq API key.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Voice generation failed: {exc}")

    # Shown only after the voice has been generated completely
    if ss.voice_audio:
        st.success("\u2705 Voice is complete. Listen below or download it.")
        st.audio(ss.voice_audio, format="audio/mp3")
        st.download_button(
            "\u2B07\uFE0F Download voice (.mp3)",
            data=ss.voice_audio,
            file_name=ss.voice_filename,
            mime="audio/mpeg",
            type="primary",
            use_container_width=True,
        )


# ---------------------------------------------------------------- knowledge base
def kb_page() -> None:
    hero("Knowledge Base", "Ingest legal PDFs, inspect what was extracted, and test retrieval.")
    docs = STORAGE.list_documents()

    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", len(docs))
    m2.metric("Chunks in Qdrant", f"{STORE.count():,}")
    m3.metric("Embedding model", SETTINGS.embedding_model.split("/")[-1])

    st.subheader("Ingest a PDF")
    up = st.file_uploader("Legal PDF (e.g. CFR Title 21)", type=["pdf"])
    o1, o2 = st.columns([2, 1])
    name = o1.text_input("Display name (optional)", placeholder="Code of Federal Regulations - Title 21")
    ocr = o2.checkbox("Run OCR (needs Tesseract)", value=SETTINGS.ocr_enabled)
    if st.button("\U0001F680 Ingest", type="primary", disabled=up is None):
        SETTINGS.pdf_dir.mkdir(parents=True, exist_ok=True)
        path = SETTINGS.pdf_dir / Path(up.name).name
        path.write_bytes(up.getbuffer())
        bar = st.progress(0.0, text="Starting...")

        def cb(done: int, total: int, msg: str) -> None:
            bar.progress(min(done / total, 1.0) if total else 0.99, text=msg)

        try:
            with st.spinner("Extracting, chunking, embedding..."):
                rep = ingest_pdf(path, STORE, SETTINGS, STORAGE, name, ocr=ocr, progress=cb)
            bar.progress(1.0, text="Finished")
            st.success(f"Ingested **{rep.document_name}** - {rep.pages} pages, {rep.chunks} chunks, "
                       f"{rep.sections} sections, revision date **{rep.revision_date}** ({rep.seconds}s)")
            st.json(rep.by_type)
            time.sleep(0.5)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error("Ingestion failed.")
            st.exception(exc)

    if docs:
        st.subheader("Ingested documents")
        st.dataframe(
            [{k: d[k] for k in ("document_name", "source_file", "revision_date", "page_count", "chunk_count", "ingested_at")} for d in docs],
            use_container_width=True, hide_index=True,
        )
        target = st.selectbox("Delete a document", ["-"] + [f"{d['document_name']} ({d['document_id']})" for d in docs])
        if target != "-" and st.button("\U0001F5D1 Delete selected document"):
            doc_id = target.rsplit("(", 1)[1].rstrip(")")
            STORE.delete_document(doc_id)
            STORAGE.delete_document(doc_id)
            st.rerun()

    st.subheader("Retrieval playground")
    q = st.text_input("Test a search query")
    if q and STORE.count():
        from app.agent.tools import make_tools

        search, section_tool = make_tools(STORE)
        for h in search.invoke({"query": q, "top_k": 5, "part": "", "section": ""}):
            with st.expander(f"{h.get('hierarchy_path') or 'Unstructured'} \u2014 p.{h.get('page')} \u2014 score {h['score']:.3f} ({h['element_type']})"):
                st.write(h["text"][:1500])
                st.json({k: v for k, v in h.items() if k != "text"})


# ------------------------------------------------------------------------- main
page, model, top_k, retries, part = sidebar()
if page.endswith("Chat"):
    chat_page(model, top_k, retries, part)
elif page.endswith("Script Studio"):
    script_page(model)
elif page.endswith("Voice Studio"):
    voice_page(model)
else:
    kb_page()