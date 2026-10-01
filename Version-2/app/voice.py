"""Text-to-speech for generated scripts, PDFs, text and Markdown files (free, no API key). Many languages.

Voices: choose Woman / Man / Child, then a tone. Uses Microsoft Edge neural voices through `edge-tts`.
Note: edge-tts can change speed, pitch and volume only (no emotion/style tags), so "tone" presets are
built from those three controls.
"""
from __future__ import annotations

import asyncio
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import edge_tts
import fitz  # PyMuPDF (already in requirements.txt)

_MARKER_RE = re.compile(r"\[S\d+(?:[,;\s]+S\d+)*\]")
_ONSCREEN_RE = re.compile(r"\(On screen:[^)]*\)", re.I)
_TIME_RE = re.compile(r"\(\s*\d+:\d{2}\s*[-\u2010-\u2015]\s*\d+:\d{2}\s*\)")
_LABEL_RE = re.compile(r"^\s*\*\*[^*]+\*\*", re.M)   # **Hook**, **Segment 1 - Name**, **Title**
_HEADING_RE = re.compile(r"^\s*#+.*$", re.M)          # ## Title, ### Segment 1 ...
_RULE_RE = re.compile(r"^\s*[-*_]{3,}\s*$", re.M)     # --- horizontal lines

DEFAULT_VOICE = "en-US-AriaNeural"

_SECTION_WORD = {
    "en": "section", "ur": "سیکشن", "hi": "धारा", "ar": "المادة",
    "es": "sección", "fr": "section", "de": "Abschnitt",
}

# ------------------------------------------------------------------ tones (speed %, pitch Hz, volume %)
TONES: dict[str, tuple[int, int, int]] = {
    "Neutral":                  (0, 0, 0),
    "Calm & soothing":          (-12, -2, -5),
    "Warm & friendly":          (-4, 2, 0),
    "Energetic":                (15, 4, 5),
    "Serious / authoritative":  (-8, -6, 5),
    "News reader":              (8, 0, 5),
    "Slow & clear (learners)":  (-25, 0, 0),
    "Cheerful":                 (10, 8, 5),
    "Storyteller":              (-10, -3, 0),
}

# Only English has a real child voice. For other languages a female voice is raised to sound child-like.
CHILD_VOICES = {"en-US": "en-US-AnaNeural"}
CHILD_LIKE_ADJUST = (8, 25, 0)  # extra speed %, pitch Hz, volume % when a child voice is imitated

PREFERRED_LOCALES = ["en-US", "en-GB", "en-IN", "ur-PK", "ur-IN", "hi-IN", "ar-SA", "es-ES", "fr-FR", "de-DE"]

# Used if the live voice list cannot be downloaded (e.g. no internet).
_FALLBACK_VOICES = [
    ("en-US-GuyNeural", "Male"), ("en-US-ChristopherNeural", "Male"), ("en-US-EricNeural", "Male"),
    ("en-US-RogerNeural", "Male"), ("en-US-SteffanNeural", "Male"),
    ("en-US-AriaNeural", "Female"), ("en-US-JennyNeural", "Female"), ("en-US-MichelleNeural", "Female"),
    ("en-US-AnaNeural", "Female"),
    ("en-GB-RyanNeural", "Male"), ("en-GB-SoniaNeural", "Female"),
    ("ur-PK-AsadNeural", "Male"), ("ur-PK-UzmaNeural", "Female"),
    ("hi-IN-MadhurNeural", "Male"), ("hi-IN-SwaraNeural", "Female"),
    ("ar-SA-HamedNeural", "Male"), ("ar-SA-ZariyahNeural", "Female"),
]

_SAMPLE_TEXT = {
    "en": "Hello! This is a short sample of how this voice sounds when it reads your script.",
    "ur": "یہ اس آواز کا ایک مختصر نمونہ ہے۔",
    "hi": "यह इस आवाज़ का एक छोटा सा नमूना है।",
    "ar": "هذا نموذج قصير لصوت القراءة.",
}


# ------------------------------------------------------------------ text preparation
def script_to_speech_text(script: str, lang: str = "en") -> str:
    """Strip everything a narrator should NOT read aloud."""
    t = _RULE_RE.sub("", script)
    t = _HEADING_RE.sub("", t)
    t = _LABEL_RE.sub("", t)
    t = _TIME_RE.sub("", t)

    t = _ONSCREEN_RE.sub("", t)
    t = _MARKER_RE.sub("", t)
    t = t.replace("**", "").replace("*", "").replace("§", _SECTION_WORD.get(lang, "section") + " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"[ \t]+([.,;:!?\u06D4\u0964])", r"\1", t)   # no space before punctuation left by removed markers
    t = re.sub(r"\n[ \t]+", "\n", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


def extract_pdf_text(data: bytes, first_page: int = 1, last_page: int | None = None) -> str:
    """Read text from a PDF (1-based page range). Returns '' for scanned PDFs without a text layer."""
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        total = len(doc)
        first = max(first_page, 1)
        last = min(last_page or total, total)
        pages = [doc[i].get_text("text", sort=True) for i in range(first - 1, last)]
    finally:
        doc.close()
    text = "\n\n".join(pages)
    text = re.sub(r"-\n(?=[a-z])", "", text)         # join words hyphenated across lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)      # single line breaks -> space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_upload_text(name: str, data: bytes, first_page: int = 1, last_page: int | None = None) -> str:
    """Text from an uploaded .pdf, .txt or .md file."""
    if name.lower().endswith((".txt", ".md", ".markdown")):
        return data.decode("utf-8-sig", errors="ignore").strip()
    return extract_pdf_text(data, first_page, last_page)


def _split_text(text: str, limit: int = 2500) -> list[str]:
    """Split into chunks of about `limit` characters at sentence ends."""
    parts: list[str] = []
    cur = ""
    for sent in re.split(r"(?<=[.!?\u06D4\u0964])\s+", text):
        while len(sent) > limit:                      # a single very long sentence
            cut = sent.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if cur and len(cur) + len(sent) + 1 > limit:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return [p for p in parts if p]


# ------------------------------------------------------------------ speech
async def _synth(text: str, voice: str, rate: str, pitch: str, volume: str, out: Path) -> None:
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume).save(str(out))


def _run(text: str, voice: str, rate: str, pitch: str, volume: str, out: Path) -> None:
    try:
        asyncio.run(_synth(text, voice, rate, pitch, volume, out))
    except RuntimeError:  # an event loop is already running
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_synth(text, voice, rate, pitch, volume, out))
        finally:
            loop.close()


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def synthesize_long(text: str, out_path: str | Path, voice: str = DEFAULT_VOICE,
                    rate_pct: int = 0, pitch_hz: int = 0, progress=None, volume_pct: int = 0) -> Path:
    """Create one MP3 from long text (split into parts, then joined).
    rate_pct: speed (-50..+50), pitch_hz: (-50..+50), volume_pct: (-50..+50). progress(done, total) is optional."""
    lang = voice.split("-")[0]
    clean = script_to_speech_text(text, lang)
    if not clean:
        raise ValueError("There is no text to read.")
    chunks = _split_text(clean)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp3")
    rate = f"{_clamp(rate_pct, -50, 50):+d}%"
    pitch = f"{_clamp(pitch_hz, -50, 50):+d}Hz"
    volume = f"{_clamp(volume_pct, -50, 50):+d}%"
    try:
        with open(out, "wb") as f:
            for i, chunk in enumerate(chunks, 1):
                _run(chunk, voice, rate, pitch, volume, tmp)
                f.write(tmp.read_bytes())
                if progress:
                    progress(i, len(chunks))
    finally:
        tmp.unlink(missing_ok=True)
    return out


def synthesize(script: str, out_path: str | Path, voice: str = DEFAULT_VOICE,
               rate_pct: int = 0, pitch_hz: int = 0, volume_pct: int = 0) -> Path:
    """Short text -> MP3 (simple use)."""
    return synthesize_long(script, out_path, voice, rate_pct, pitch_hz, volume_pct=volume_pct)


# ------------------------------------------------------------------ translation + voices
def translate_text(text: str, language_name: str, llm, progress=None) -> str:
    """Translate long text with the LLM, part by part.
    The text is cleaned first (no headings, cues, source markers), so only spoken words get translated."""
    system = (
        f"You are a professional translator. Translate the text faithfully into {language_name}. "
        "Keep numbers and section numbers exactly as they are. Do not add, remove or explain anything. "
        "Output only the translation."
    )
    parts = _split_text(script_to_speech_text(text, "en"))
    out: list[str] = []
    for i, part in enumerate(parts, 1):
        out.append(llm.complete(system, part, max_tokens=3000).strip())
        if progress:
            progress(i, len(parts))
    return "\n\n".join(out)


@lru_cache(maxsize=1)
def get_voices() -> tuple[dict, ...]:
    """All voices as dicts with ShortName, Locale, Gender ('Male'/'Female'). Falls back to a built-in list offline."""
    try:
        voices = asyncio.run(edge_tts.list_voices())
        return tuple(
            {"ShortName": v["ShortName"], "Locale": v["Locale"], "Gender": v["Gender"]} for v in voices
        )
    except Exception:  # no internet, or an event loop is already running
        return tuple(
            {"ShortName": n, "Locale": n.rsplit("-", 1)[0], "Gender": g} for n, g in _FALLBACK_VOICES
        )


def list_voices_by_language() -> dict[str, list[str]]:
    """{'ur-PK': ['ur-PK-AsadNeural', ...], 'en-US': [...], ...}"""
    out: dict[str, list[str]] = {}
    for v in get_voices():
        out.setdefault(v["Locale"], []).append(v["ShortName"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def voices_for(locale: str, speaker: str) -> list[str]:
    """Voice names for a locale and speaker type: 'Man', 'Woman' or 'Child'."""
    if speaker == "Child":
        child = CHILD_VOICES.get(locale)
        return [child] if child else []
    gender = "Male" if speaker == "Man" else "Female"
    names = [v["ShortName"] for v in get_voices() if v["Locale"] == locale and v["Gender"] == gender]
    return sorted(n for n in names if n not in CHILD_VOICES.values())


def tone_settings(tone: str, imitate_child: bool = False, custom: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    """(rate %, pitch Hz, volume %) for a tone name; 'Custom' uses the given values."""
    rate, pitch, volume = custom if (tone == "Custom" and custom) else TONES.get(tone, TONES["Neutral"])
    if imitate_child:
        rate, pitch, volume = rate + CHILD_LIKE_ADJUST[0], pitch + CHILD_LIKE_ADJUST[1], volume + CHILD_LIKE_ADJUST[2]
    return _clamp(rate, -50, 50), _clamp(pitch, -50, 50), _clamp(volume, -50, 50)


def preview_voice(voice: str, rate_pct: int, pitch_hz: int, volume_pct: int) -> bytes:
    """Short MP3 sample (bytes) so you can hear a voice and tone before reading a long script."""
    sample = _SAMPLE_TEXT.get(voice.split("-")[0], _SAMPLE_TEXT["en"])
    with tempfile.TemporaryDirectory() as d:
        path = synthesize_long(sample, Path(d) / "preview.mp3", voice, rate_pct, pitch_hz, volume_pct=volume_pct)
        return path.read_bytes()


# ------------------------------------------------------------------ Streamlit voice controls
def render_voice_controls(key: str = "tts") -> dict:
    """Draws Language / Speaker (Woman, Man, Child) / Voice / Tone controls and a preview button.
    Returns {'voice', 'rate_pct', 'pitch_hz', 'volume_pct'} to pass to synthesize_long()."""
    import streamlit as st

    voices = get_voices()
    locales = sorted(
        {v["Locale"] for v in voices},
        key=lambda loc: (PREFERRED_LOCALES.index(loc) if loc in PREFERRED_LOCALES else 99, loc),
    )
    c1, c2, c3 = st.columns(3)
    locale = c1.selectbox("Language / accent", locales, key=f"{key}_locale")
    speaker = c2.selectbox("Speaker", ["Woman", "Man", "Child"], key=f"{key}_speaker")
    tone = c3.selectbox("Tone", list(TONES) + ["Custom"], key=f"{key}_tone")

    imitate_child = False
    names = voices_for(locale, speaker)
    if speaker == "Child" and not names:
        names = voices_for(locale, "Woman")
        imitate_child = True
        st.info("This language has no real child voice, so a woman's voice is made higher and faster to sound child-like. "
                "For a real child voice choose English (US).")
    if not names:
        st.warning("No voice of this type is available for this language. Try another speaker type.")
        names = [DEFAULT_VOICE]
    voice = st.selectbox("Voice", names, key=f"{key}_voice",
                         format_func=lambda n: n.split("-", 2)[-1].replace("Neural", ""))

    custom = None
    if tone == "Custom":
        s1, s2, s3 = st.columns(3)
        custom = (
            s1.slider("Speed %", -50, 50, 0, key=f"{key}_rate"),
            s2.slider("Pitch Hz", -50, 50, 0, key=f"{key}_pitch"),
            s3.slider("Volume %", -50, 50, 0, key=f"{key}_vol"),
        )
    rate, pitch, volume = tone_settings(tone, imitate_child, custom)
    st.caption(f"Speed {rate:+d}% \u00b7 Pitch {pitch:+d} Hz \u00b7 Volume {volume:+d}%")

    if st.button("\u25B6 Preview this voice", key=f"{key}_preview"):
        try:
            with st.spinner("Creating preview..."):
                st.audio(preview_voice(voice, rate, pitch, volume), format="audio/mp3")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Preview failed (internet needed for the voices): {exc}")
    return {"voice": voice, "rate_pct": rate, "pitch_hz": pitch, "volume_pct": volume}