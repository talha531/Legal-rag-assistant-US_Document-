"""
src/youtube_script.py
======================
SINGLE ACTION: given a topic (or a CFR section number), retrieve grounded
source material and write a ~10-minute YouTube video script from it.

Notebook origin: the "YOUTUBE SCRIPT GENERATOR" cell.

Run directly:
    python -m src.youtube_script "driver medical qualification requirements"
"""

import sys
import time

import config
from src.groq_client import get_groq_client
from src.retriever import semantic_search
from src.section_retrieval import detect_cfr_section, section_exact_answer_context

SCRIPT_WRITER_SYSTEM = """
You are a scriptwriter for a YouTube channel that explains real federal
regulations in an engaging but accurate way.

You must write the script using ONLY the supplied CFR source material for
any factual/legal claims.

STRICT RULES:
1. Use ONLY the supplied source material for legal facts.
2. Do NOT invent section numbers, dates, penalties, or procedures.
3. Do NOT invent facts not present in the supplied material.
4. You MAY add narration, hooks, transitions, and delivery notes
   ("(pause)", "(screen: ...)") — that's creative framing, not a legal
   claim, so it doesn't need to come from the source.
5. Cite the document and page for any regulation you quote or paraphrase.
6. If the supplied material does not fully cover the requested topic, say
   so plainly in the script rather than filling gaps with invented detail.
7. Target spoken length: ~10 minutes (~1,400-1,550 words), 150 wpm pace.
"""


def build_script_prompt(topic: str, context: str) -> str:
    return f"""
VIDEO TOPIC:
{topic}

SOURCE MATERIAL (grounded facts — use only this for legal claims):
{context}

Write a complete 10-minute YouTube video script about this topic.

Structure it with timestamp markers, like:
[0:00 - 0:30] HOOK
[0:30 - 1:30] WHAT THIS REGULATION COVERS
[1:30 - 4:00] KEY PROVISIONS (walk through the actual subsections)
[4:00 - 6:00] WHY THIS MATTERS / WHO IT APPLIES TO
[6:00 - 8:00] COMMON MISTAKES OR EDGE CASES (only if supported by the source)
[8:00 - 9:30] PRACTICAL TAKEAWAY / SUMMARY
[9:30 - 10:00] OUTRO / CTA

Include short (screen: ...) direction notes for the editor.
Cite document name and page wherever a specific rule is stated.
Do not use outside legal knowledge — only the supplied material.
"""


def retrieve_topic_context(topic: str):
    """Exact CFR section if named, otherwise semantic search across the collection."""
    detected_section = detect_cfr_section(topic)
    if detected_section:
        results = section_exact_answer_context(detected_section)
        if results:
            return results, detected_section
    results = semantic_search(topic, top_k=8)
    return results, None


def build_script_context(results: list, max_chars: int = 16000) -> str:
    blocks, total_chars = [], 0
    for i, r in enumerate(results, 1):
        text = r.get("text", "").strip()
        if not text:
            continue
        block = f"[SOURCE {i}]\nDocument: {r.get('document', 'Unknown')}\nPage: {r.get('page', 'Unknown')}\nText:\n{text}\n"
        if total_chars + len(block) > max_chars:
            break
        blocks.append(block)
        total_chars += len(block)
    return "\n\n".join(blocks)


def generate_youtube_script(topic: str):
    results, section = retrieve_topic_context(topic)
    if not results:
        print("❌ No relevant material found in the indexed collection for this topic.")
        return None

    context = build_script_context(results)
    prompt = build_script_prompt(topic, context)

    groq_client = get_groq_client()
    start_time = time.time()
    try:
        response = groq_client.chat.completions.create(
            model=config.TEXT_MODEL,
            messages=[
                {"role": "system", "content": SCRIPT_WRITER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="low",
            include_reasoning=False,
            temperature=0.4,
            max_completion_tokens=3000,
            stream=False,
        )
        script = response.choices[0].message.content or "⚠️ Empty response from Groq."
    except Exception as e:
        script = f"❌ Groq error:\n\n{e}"

    print(f"⏱️ Generated in {time.time() - start_time:.2f}s")
    return script


if __name__ == "__main__":
    video_topic = " ".join(sys.argv[1:]) or input("Video topic → ").strip()
    if not video_topic:
        print("❌ No topic entered.")
    else:
        result_script = generate_youtube_script(video_topic)
        if result_script:
            print("\n" + "=" * 70)
            print("📜 GENERATED SCRIPT")
            print("=" * 70)
            print(result_script)
