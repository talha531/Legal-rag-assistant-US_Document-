"""
src/self_rag_answer.py
=======================
SINGLE ACTION: answer a legal question, grounded strictly in retrieved
passages, using a two-round Self-RAG loop (retrieve -> judge -> retrieve
more if needed -> answer).

Notebook origin: Cell 10 ("ROBUST LEGAL SELF-RAG ANSWER GENERATION").

Run directly:
    python -m src.self_rag_answer "What does §386.48 require?"
"""

import re
import sys
import time

import config
from src.groq_client import get_groq_client
from src.retriever import semantic_search
from src.section_retrieval import extract_section_refs
from src.self_rag_judge import self_rag_judge


def normalize_section(section: str) -> str:
    section = str(section).strip()
    if section.startswith("§"):
        section = section[1:]
    return section.replace(" ", "")


def exact_section_retrieval(query: str, limit: int = config.EXACT_MATCH_LIMIT) -> list:
    """Lexically scan every chunk for an exact CFR citation match."""
    from src.retriever import get_qdrant_client

    section_refs = extract_section_refs(query)
    if not section_refs:
        return []
    section_refs = [normalize_section(x) for x in section_refs]

    qdrant = get_qdrant_client()
    matches = []
    offset = None

    while True:
        points, offset = qdrant.scroll(
            collection_name=config.COLLECTION_NAME, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break

        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text", ""))
            if not text:
                continue

            matched_sections = [
                s for s in section_refs
                if re.search(rf'\b{re.escape(s)}\b', text, flags=re.IGNORECASE)
            ]
            if matched_sections:
                score = sum(
                    1.00 if re.search(rf'§\s*{re.escape(s)}\b', text, flags=re.IGNORECASE)
                    else 0.95 if re.search(rf'\bsection\s+{re.escape(s)}\b', text, flags=re.IGNORECASE)
                    else 0.85
                    for s in matched_sections
                )
                matches.append({
                    "rank": 0, "score": score, "semantic_score": 0.0,
                    "lexical_score": score, "rerank_score": score,
                    "chunk_id": payload.get("chunk_id", str(point.id)),
                    "document": payload.get("document", "Unknown"),
                    "page": payload.get("page", "Unknown"),
                    "text": text,
                    "word_count": payload.get("word_count", len(text.split())),
                    "exact_section_match": True,
                    "matched_sections": matched_sections,
                })

        if offset is None:
            break

    matches.sort(key=lambda x: x["rerank_score"], reverse=True)

    unique, seen = [], set()
    for item in matches:
        if item["chunk_id"] in seen:
            continue
        seen.add(item["chunk_id"])
        unique.append(item)
        if len(unique) >= limit:
            break

    for i, item in enumerate(unique, 1):
        item["rank"] = i
    return unique


def semantic_retrieval(query: str, top_k: int = 10) -> list:
    """Thin wrapper around retriever.semantic_search that adds rerank fields."""
    results = semantic_search(query, top_k=top_k)
    for r in results:
        r["semantic_score"] = r["score"]
        r["lexical_score"] = 0.0
        r["rerank_score"] = r["score"]
        r["exact_section_match"] = False
    return results


def legal_retrieve(query: str, top_k: int = 10) -> list:
    """Exact CFR citation results first, semantic results filling the rest."""
    exact_results = exact_section_retrieval(query)
    if not exact_results:
        return semantic_retrieval(query, top_k=top_k)

    semantic_results = semantic_retrieval(query, top_k=30)
    combined, seen = [], set()
    for item in exact_results + semantic_results:
        if item["chunk_id"] not in seen:
            combined.append(item)
            seen.add(item["chunk_id"])

    combined = combined[:top_k]
    for i, item in enumerate(combined, start=1):
        item["rank"] = i
    return combined


def build_context(results: list, max_chars: int = 18000) -> str:
    blocks, total_chars = [], 0
    for i, result in enumerate(results, 1):
        text = result.get("text", "").strip()
        if not text:
            continue
        block = (
            f"[PASSAGE {i}]\nDocument: {result.get('document', 'Unknown')}\n"
            f"Page: {result.get('page', 'Unknown')}\n"
            f"Exact citation match: {result.get('exact_section_match', False)}\nText:\n{text}\n"
        )
        if total_chars + len(block) > max_chars:
            break
        blocks.append(block)
        total_chars += len(block)
    return "\n" + "\n".join(blocks)


def generate_legal_answer(question: str, results: list) -> str:
    groq_client = get_groq_client()
    context = build_context(results)

    prompt = f"""
You are a legal-document question answering assistant.

Answer the user's question using ONLY the supplied passages.

IMPORTANT RULES:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the question contains an exact CFR section such as §1308.45, use the passage containing that section as the primary evidence.
4. If the passages contain the answer, answer directly.
5. Preserve important legal conditions, exceptions, deadlines, procedures, and requirements.
6. Do not say information is missing when the supplied passages actually contain the requested section.
7. Cite the document and page containing the answer.
8. If multiple passages belong to the same section, combine them.
9. Keep the answer concise but complete.

USER QUESTION:
{question}

SUPPLIED PASSAGES:
{context}

Return exactly:

Answer:
<grounded answer>

Source:
<document name>, p. <page>
"""

    response = groq_client.chat.completions.create(
        model=config.TEXT_MODEL,
        messages=[
            {"role": "system", "content": "You answer legal-document questions strictly from supplied evidence."},
            {"role": "user", "content": prompt},
        ],
        reasoning_effort="low",
        include_reasoning=False,
        temperature=0,
        max_completion_tokens=700,
        stream=False,
    )
    return response.choices[0].message.content or ""


def self_rag_answer(question: str) -> str:
    """Full pipeline: exact/semantic retrieval -> judge -> (retrieve again) -> answer."""
    start_time = time.time()

    exact_results = exact_section_retrieval(question)
    results_1 = exact_results[:config.TOP_K_INITIAL] if exact_results else \
        legal_retrieve(question, top_k=config.TOP_K_INITIAL)

    judge_1 = self_rag_judge(question, results_1)

    if judge_1["decision"] == "ANSWER" and judge_1["sufficient"]:
        final_results = results_1
    else:
        if exact_results:
            expanded_semantic = semantic_retrieval(question, top_k=30)
            combined, seen = [], set()
            for item in exact_results + expanded_semantic:
                if item["chunk_id"] not in seen:
                    combined.append(item)
                    seen.add(item["chunk_id"])
            results_2 = combined[:config.TOP_K_EXPANDED]
        else:
            results_2 = legal_retrieve(question, top_k=config.TOP_K_EXPANDED)

        for i, item in enumerate(results_2, start=1):
            item["rank"] = i

        final_results = results_2  # used regardless of round-2 judge outcome, mirrors notebook behavior

    answer = generate_legal_answer(question, final_results)
    print(f"⏱️ Self-RAG answered in {time.time() - start_time:.2f}s")
    return answer


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Question: ").strip()
    print(self_rag_answer(q))
