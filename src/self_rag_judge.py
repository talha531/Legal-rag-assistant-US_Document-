"""
src/self_rag_judge.py
======================
SINGLE ACTION: ask an LLM to judge whether a set of retrieved passages is
relevant and sufficient to answer a question ("Self-RAG" grading step).

Notebook origin: Cell 9 ("SELF-RAG JUDGE — GPT-OSS + JSON MODE").

Run directly (uses semantic_search to fetch passages first):
    python -m src.self_rag_judge "What authority does the Secretary have?"
"""

import json
import re
import sys

import config
from src.groq_client import get_groq_client


def extract_json_object(text: str):
    """Pull the first well-formed {...} JSON object out of an LLM response."""
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def build_judge_context(results: list) -> str:
    blocks = []
    for rank, item in enumerate(results[:10], start=1):
        text = " ".join(item.get("text", "")[:800].split())
        blocks.append(
            f"PASSAGE {rank}\n"
            f"Document: {item.get('document', 'Unknown')}\n"
            f"Page: {item.get('page', '?')}\n"
            f"Score: {item.get('score', 0):.4f}\n"
            f"Text: {text}"
        )
    return "\n\n".join(blocks)


def self_rag_judge(question: str, results: list) -> dict:
    """
    Return {"relevant": bool, "sufficient": bool, "decision": str, "reason": str}
    describing whether `results` can answer `question` without inventing facts.
    """
    groq_client = get_groq_client()
    context = build_judge_context(results)

    prompt = f"""
Evaluate these retrieved legal passages.

QUESTION:
{question}

PASSAGES:
{context}

Determine:

relevant: true if at least one passage helps answer the question.
sufficient: true only if the passages contain enough evidence to answer without inventing information.
decision: ANSWER | RETRIEVE | NOT_RELEVANT | NOT_SUFFICIENT

Use:
ANSWER = relevant and sufficient
RETRIEVE = relevant but more evidence is needed
NOT_RELEVANT = no passage is relevant
NOT_SUFFICIENT = relevant evidence exists but is insufficient

Return ONLY JSON:
{{"relevant": true, "sufficient": false, "decision": "RETRIEVE", "reason": "short explanation"}}
"""

    try:
        response = groq_client.chat.completions.create(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Classify the retrieved passages. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="low",
            include_reasoning=False,
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=300,
            stream=False,
        )
        raw = response.choices[0].message.content or ""

        if not raw.strip():
            return {"relevant": False, "sufficient": False, "decision": "RETRIEVE",
                     "reason": "Groq returned empty content."}

        data = extract_json_object(raw)
        if data is None:
            return {"relevant": False, "sufficient": False, "decision": "RETRIEVE",
                     "reason": "Groq returned invalid JSON."}

        relevant = bool(data.get("relevant", False))
        sufficient = bool(data.get("sufficient", False))
        decision = str(data.get("decision", "RETRIEVE")).upper().strip()
        reason = str(data.get("reason", "")).strip()

        allowed = {"ANSWER", "RETRIEVE", "NOT_RELEVANT", "NOT_SUFFICIENT"}
        if decision not in allowed:
            decision = "ANSWER" if (relevant and sufficient) else ("NOT_SUFFICIENT" if relevant else "NOT_RELEVANT")

        return {"relevant": relevant, "sufficient": sufficient, "decision": decision, "reason": reason}

    except Exception as e:
        return {"relevant": False, "sufficient": False, "decision": "RETRIEVE",
                 "reason": f"Judge error: {e}"}


if __name__ == "__main__":
    from src.retriever import semantic_search

    question = " ".join(sys.argv[1:]) or input("Question: ").strip()
    passages = semantic_search(question, top_k=10)
    verdict = self_rag_judge(question, passages)
    print(json.dumps(verdict, indent=2))
