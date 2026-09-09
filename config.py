"""
config.py
=========
SINGLE ACTION: hold every path, model name, and tunable constant used by
the pipeline in one place, and load environment variables (.env).

Nothing else in this project should hard-code a path or a model name —
every other module imports what it needs from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a local .env file (GROQ_API_KEY=...) if present.
load_dotenv()

# ----------------------------------------------------------------------
# 1. GROQ / LLM SETTINGS
# ----------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

TEXT_MODEL = os.environ.get("TEXT_MODEL", "openai/gpt-oss-120b")     # main answer model
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-20b")    # self-RAG judge / chat
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3.6-27b")    # reserved, not used yet

# ----------------------------------------------------------------------
# 2. PROJECT / DATA PATHS
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Where source PDFs live before processing (drop your PDFs here)
INPUT_DIR = BASE_DIR / "data" / "input_pdfs"

# Everything the pipeline generates lives under WORK_DIR
WORK_DIR = BASE_DIR / "data" / "book_assistant_data"

TEXT_DIR = WORK_DIR / "texts"
META_DIR = WORK_DIR / "metadata"
CHUNK_DIR = WORK_DIR / "chunks"
EMBED_DIR = WORK_DIR / "embeddings"
QDRANT_DIR = WORK_DIR / "qdrant_db"

CHUNK_FILE = CHUNK_DIR / "chunks.jsonl"
CHUNK_META_FILE = CHUNK_DIR / "chunks_metadata.json"

EMBEDDING_FILE = EMBED_DIR / "embeddings.npy"
EMBED_METADATA_JSON = EMBED_DIR / "embedding_metadata.json"
EMBED_METADATA_JSONL = EMBED_DIR / "embedding_metadata.jsonl"

QDRANT_CONFIG_FILE = WORK_DIR / "qdrant_config.json"

COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "legal_documents")

# ----------------------------------------------------------------------
# 3. CHUNKING SETTINGS
# ----------------------------------------------------------------------
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))      # words
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))  # words

# ----------------------------------------------------------------------
# 4. EMBEDDING SETTINGS
# ----------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")

# ----------------------------------------------------------------------
# 5. RETRIEVAL SETTINGS
# ----------------------------------------------------------------------
TOP_K_DEFAULT = int(os.environ.get("TOP_K_DEFAULT", 5))
TOP_K_INITIAL = int(os.environ.get("TOP_K_INITIAL", 5))
TOP_K_EXPANDED = int(os.environ.get("TOP_K_EXPANDED", 10))
EXACT_MATCH_LIMIT = int(os.environ.get("EXACT_MATCH_LIMIT", 20))


def ensure_directories() -> None:
    """Create every directory this project writes to, if missing."""
    for d in (INPUT_DIR, WORK_DIR, TEXT_DIR, META_DIR, CHUNK_DIR, EMBED_DIR, QDRANT_DIR):
        d.mkdir(parents=True, exist_ok=True)
