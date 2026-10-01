"""Central configuration. Every value comes from environment variables / the .env file.

Secrets (GROQ_API_KEY, QDRANT_API_KEY) are read only from the environment - never hard-coded.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)


class ConfigError(RuntimeError):
    """Raised when required configuration (e.g. the Groq API key) is missing."""


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    p = Path(_str(name) or default)
    return p if p.is_absolute() else ROOT_DIR / p


@dataclass(frozen=True)
class Settings:
    # Groq
    groq_api_key: str
    groq_model: str
    groq_temperature: float
    groq_max_tokens: int
    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_path: Path
    collection: str
    # Embeddings
    embedding_model: str
    # Retrieval / agent
    top_k: int
    min_score: float
    max_retries: int
    max_context_chars: int
    validation_snippet_chars: int
    # Chunking
    chunk_max_chars: int
    chunk_min_chars: int
    chunk_overlap: int
    # OCR
    ocr_enabled: bool
    ocr_lang: str
    min_image_px: int
    # Misc
    default_revision_date: str
    sqlite_path: Path
    images_dir: Path
    pdf_dir: Path

    @property
    def has_groq_key(self) -> bool:
        key = self.groq_api_key
        return bool(key) and not key.lower().startswith("your_")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            groq_api_key=_str("GROQ_API_KEY"),
            groq_model=_str("GROQ_MODEL", "llama-3.1-8b-instant"),
            groq_temperature=_float("GROQ_TEMPERATURE", 0.0),
            groq_max_tokens=_int("GROQ_MAX_TOKENS", 1200),
            qdrant_url=_str("QDRANT_URL"),
            qdrant_api_key=_str("QDRANT_API_KEY"),
            qdrant_path=_path("QDRANT_PATH", "storage/qdrant"),
            collection=_str("QDRANT_COLLECTION", "legal_cfr_chunks"),
            embedding_model=_str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            top_k=_int("TOP_K", 5),
            min_score=_float("MIN_SCORE", 0.35),
            max_retries=_int("MAX_RETRIES", 2),
            max_context_chars=_int("MAX_CONTEXT_CHARS", 7000),
            validation_snippet_chars=_int("VALIDATION_SNIPPET_CHARS", 450),
            chunk_max_chars=_int("CHUNK_MAX_CHARS", 1800),
            chunk_min_chars=_int("CHUNK_MIN_CHARS", 350),
            chunk_overlap=_int("CHUNK_OVERLAP", 200),
            ocr_enabled=_bool("OCR_ENABLED", False),
            ocr_lang=_str("OCR_LANG", "eng"),
            min_image_px=_int("MIN_IMAGE_PX", 120),
            default_revision_date=_str("DEFAULT_REVISION_DATE"),
            sqlite_path=_path("SQLITE_PATH", "storage/app.db"),
            images_dir=ROOT_DIR / "storage" / "images",
            pdf_dir=ROOT_DIR / "data" / "pdfs",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def setup_logging() -> None:
    level = _str("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
