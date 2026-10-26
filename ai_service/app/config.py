"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PROJECT_ENV_FILE, override=True)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # LLM
    groq_api_key: str
    groq_model: str
    llm_mode: str  # "groq" | "mock" | "off"
    llm_concurrency: int
    # Security between Django and this service
    internal_token: str
    # GitHub OAuth App credentials, used only to revoke one-run grants
    github_oauth_client_id: str
    github_oauth_client_secret: str
    # RAG
    embedding_backend: str  # "onnx" | "hash"
    chroma_dir: str
    # Limits
    max_source_files: int
    max_file_bytes: int
    max_zip_mb: int
    chunk_size: int
    chunk_overlap: int
    max_chunks_per_reviewer: int
    job_timeout_s: int

    @property
    def llm_enabled(self) -> bool:
        if self.llm_mode == "mock":
            return True
        return self.llm_mode == "groq" and bool(self.groq_api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
            llm_mode=os.getenv("LLM_MODE", "groq").strip().lower(),
            llm_concurrency=_int("LLM_CONCURRENCY", 1),
            internal_token=os.getenv("AI_SERVICE_TOKEN", "change-me"),
            github_oauth_client_id=os.getenv("GITHUB_OAUTH_CLIENT_ID", "").strip(),
            github_oauth_client_secret=os.getenv("GITHUB_OAUTH_CLIENT_SECRET", "").strip(),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "onnx").lower(),
            chroma_dir=os.getenv("CHROMA_DIR", "./chroma_data"),
            max_source_files=_int("MAX_SOURCE_FILES", 10),
            max_file_bytes=_int("MAX_FILE_BYTES", 200_000),
            max_zip_mb=_int("MAX_ZIP_MB", 60),
            chunk_size=_int("CHUNK_SIZE", 3500),
            chunk_overlap=_int("CHUNK_OVERLAP", 200),
            max_chunks_per_reviewer=_int("MAX_CHUNKS_PER_REVIEWER", 1),
            job_timeout_s=_int("JOB_TIMEOUT_S", 900),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
