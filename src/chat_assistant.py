"""
src/chat_assistant.py
======================
SINGLE ACTION: run an interactive terminal chat. The user enters a CFR
section number, the assistant loads exactly that section's text, and
answers follow-up questions using ONLY that loaded material.

Notebook origin: the "ATTRACTIVE ITERATIVE CFR SECTION CHAT" cell.

Run directly:
    python -m src.chat_assistant
"""

import config
from src.groq_client import get_groq_client
from src.section_retrieval import detect_cfr_section, section_exact_answer_context

LEGAL_CHAT_SYSTEM = """
You are a CFR legal-document research assistant.

You must answer ONLY from the supplied CFR source material.

STRICT RULES:
1. Use ONLY the supplied source material.
2. Do NOT use outside knowledge.
3. Do NOT invent facts, requirements, exceptions, procedures, dates, or section numbers.
4. Do NOT change the meaning of the regulation.
5. If the source does not answer the question, clearly say so.
6. Distinguish source-supported facts from plain-English explanation.
7. Preserve subsection references such as (a), (b), (c), etc.
8. Give document and page when available.
9. If asked for a summary, summarize ONLY the supplied material.
10. If asked for "everything", cover all relevant material in the supplied section.
"""


class ChatSession:
    """Holds the currently loaded CFR section + conversation history."""

    def __init__(self):
        self.section = None
        self.results = []
        self.context = ""
        self.history = []
        self.groq_client = get_groq_client()

    def load_section(self, section: str) -> bool:
        results = section_exact_answer_context(section)
        if not results:
            print(f"❌ No exact indexed section found for {section}")
            return False

        self.section = section
        self.results = results
        self.context = "\n".join(
            f"\n================ SOURCE {i} ================\n"
            f"Document:\n{r.get('document', 'Unknown')}\n"
            f"Page:\n{r.get('page', 'Unknown')}\n"
            f"Exact Section:\n{r.get('exact_section', False)}\n"
            f"Text:\n{r.get('text', '')}\n"
            for i, r in enumerate(results, 1)
        )
        print(f"✅ SECTION LOADED: {self.section} ({len(self.results)} chunks)")
        return True

    def show_sources(self):
        if not self.results:
            print("❌ No sources loaded.")
            return
        print(f"📚 SOURCES FOR {self.section}")
        for i, r in enumerate(self.results, 1):
            print(f"[{i}] {r.get('document', 'Unknown')} | page {r.get('page', 'Unknown')}")

    def answer(self, question: str) -> str:
        if not self.context:
            return "No CFR section is currently loaded. Please enter a section such as §231.4."

        prompt = f"""
CURRENT CFR SECTION:
{self.section}

SOURCE MATERIAL:
{self.context}

CONVERSATION HISTORY:
{self.history}

USER QUESTION:
{question}

Answer the user's question using ONLY the supplied CFR material.

Use this structure when appropriate:
### Answer
### Explanation
### Relevant Provision
### Source

If the supplied material does not support an answer, say so instead of guessing.
"""
        try:
            response = self.groq_client.chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": LEGAL_CHAT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                reasoning_effort="low",
                include_reasoning=False,
                temperature=0,
                max_completion_tokens=1200,
                stream=False,
            )
            return response.choices[0].message.content or "⚠️ Empty response from Groq."
        except Exception as e:
            return f"❌ Groq error:\n\n{e}"


def start_cfr_chat():
    session = ChatSession()

    print("=" * 70)
    print("📚 CFR LEGAL RESEARCH ASSISTANT")
    print("=" * 70)
    print("Commands: /sources  /section  /clear  /exit")
    print("Start by entering a CFR citation, e.g. §236.775, §231.4, section 1308.45\n")

    while True:
        try:
            user_input = input("You → ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Chat ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "exit", "quit"):
            print("👋 CFR chat ended.")
            break
        if user_input.lower() == "/clear":
            session.history = []
            print("🧹 Conversation history cleared.")
            continue
        if user_input.lower() == "/sources":
            session.show_sources()
            continue
        if user_input.lower() == "/section":
            new_section_raw = input("Enter CFR section: ").strip()
            detected = detect_cfr_section(new_section_raw)
            if not detected:
                print("❌ Could not detect a CFR section.")
                continue
            if session.load_section(detected):
                session.history = []
            continue

        detected_section = detect_cfr_section(user_input)
        if detected_section and detected_section != session.section:
            if not session.load_section(detected_section):
                print(f"❌ Could not load {detected_section}")
                continue
            session.history = []
            print("You can now ask follow-up questions.\n")
            continue

        if not session.section:
            print("⚠️ Please enter a CFR section first, e.g. §236.775")
            continue

        session.history.append({"role": "user", "content": user_input})
        answer = session.answer(user_input)
        session.history.append({"role": "assistant", "content": answer})

        print(f"\n🤖 {answer}\n")


if __name__ == "__main__":
    start_cfr_chat()
