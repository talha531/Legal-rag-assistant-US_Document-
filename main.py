"""
main.py
=======
ROOT of the project tree. Every branch below is one file that performs
exactly one action; this file only wires them together as CLI subcommands
so you can run the whole pipeline, or any single step, from VS Code's
terminal.

    python main.py setup            # step 1: test the Groq API key
    python main.py extract          # step 3: PDFs -> text on disk
    python main.py chunk            # step 4: text -> chunks.jsonl
    python main.py embed            # step 5: chunks -> embeddings.npy
    python main.py index            # step 6: embeddings -> Qdrant DB
    python main.py pipeline         # steps 3-6 in order (build everything)
    python main.py ask "question"   # step 10: one grounded Self-RAG answer
    python main.py chat             # step 11: interactive CFR chat
    python main.py script "topic"   # step 12: YouTube script generator
"""

import argparse
import sys

import config


def cmd_setup(_args):
    from src.groq_client import get_groq_client, test_connection
    client = get_groq_client()
    print("Groq connection test:", test_connection(client))


def cmd_extract(_args):
    from src.pdf_loader import collect_pdfs, print_summary
    from src.text_extractor import extract_pdfs_to_disk
    pdfs = collect_pdfs()
    print_summary(pdfs)
    extract_pdfs_to_disk(pdfs)


def cmd_chunk(_args):
    from src.chunker import chunk_pages_to_disk
    chunk_pages_to_disk()


def cmd_embed(_args):
    from src.embedder import embed_chunks_to_disk
    embed_chunks_to_disk()


def cmd_index(_args):
    from src.vector_store import build_qdrant_collection
    build_qdrant_collection()


def cmd_pipeline(args):
    cmd_extract(args)
    cmd_chunk(args)
    cmd_embed(args)
    cmd_index(args)
    print("\n✅ Full pipeline complete. Try: python main.py ask \"your question\"")


def cmd_ask(args):
    from src.self_rag_answer import self_rag_answer
    question = " ".join(args.question) or input("Question: ").strip()
    print(self_rag_answer(question))


def cmd_chat(_args):
    from src.chat_assistant import start_cfr_chat
    start_cfr_chat()


def cmd_script(args):
    from src.youtube_script import generate_youtube_script
    topic = " ".join(args.topic) or input("Video topic: ").strip()
    result = generate_youtube_script(topic)
    if result:
        print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legal RAG assistant — pipeline root.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="Test the Groq API connection.").set_defaults(func=cmd_setup)
    sub.add_parser("extract", help="Extract PDF text to disk.").set_defaults(func=cmd_extract)
    sub.add_parser("chunk", help="Chunk extracted text.").set_defaults(func=cmd_chunk)
    sub.add_parser("embed", help="Create embeddings for chunks.").set_defaults(func=cmd_embed)
    sub.add_parser("index", help="Load embeddings into Qdrant.").set_defaults(func=cmd_index)
    sub.add_parser("pipeline", help="Run extract -> chunk -> embed -> index.").set_defaults(func=cmd_pipeline)

    ask_parser = sub.add_parser("ask", help="Ask one grounded question (Self-RAG).")
    ask_parser.add_argument("question", nargs="*")
    ask_parser.set_defaults(func=cmd_ask)

    sub.add_parser("chat", help="Interactive CFR section chat.").set_defaults(func=cmd_chat)

    script_parser = sub.add_parser("script", help="Generate a grounded YouTube script.")
    script_parser.add_argument("topic", nargs="*")
    script_parser.set_defaults(func=cmd_script)

    return parser


if __name__ == "__main__":
    config.ensure_directories()
    arg_parser = build_parser()
    parsed_args = arg_parser.parse_args()
    parsed_args.func(parsed_args)
