"""
src/groq_client.py
===================
SINGLE ACTION: create a Groq API client and confirm the key/model works.

Notebook origin: Cell 1 ("Works with a Kaggle Secret / env var / manual prompt").
In VS Code we only use an environment variable (.env file), no notebook widgets.

Run directly:
    python -m src.groq_client
"""

from getpass import getpass

from groq import Groq

import config


def get_groq_client() -> Groq:
    """Return a ready-to-use Groq client, prompting for a key if needed."""
    api_key = config.GROQ_API_KEY

    if not api_key:
        api_key = getpass("Enter your Groq API key: ").strip()

    if not api_key:
        raise ValueError("A Groq API key is required. Set GROQ_API_KEY in your .env file.")

    return Groq(api_key=api_key)


def test_connection(client: Groq) -> str:
    """Send a one-token request to confirm the API key + model work."""
    response = client.chat.completions.create(
        model=config.TEXT_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_completion_tokens=10,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    groq_client = get_groq_client()
    result = test_connection(groq_client)
    print("Groq connection test:", result)
