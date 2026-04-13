from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Auto-generate a DB encryption key if not provided.
# Persists for the lifetime of the process.
_DEFAULT_DB_KEY = secrets.token_hex(16)


@dataclass(frozen=True)
class Config:
    # LLM
    openai_api_key: str
    langchain_api_key: str
    langchain_project: str

    # Database
    db_encryption_key: str
    db_path: Path

    # Rate Limiting
    max_cost_per_call_usd: float
    max_retries_per_node: int
    llm_timeout_seconds: int

    # LLM Provider
    llm_provider: str  # "openai", "gemini", "groq"

    # Audio
    whisper_model_size: str
    confidence_threshold: float
    low_confidence_halt_ratio: float


def load_config() -> Config:
    def _require(key: str) -> str:
        value = os.environ.get(key)
        if not value:
            raise OSError(f"Required environment variable {key} is not set")
        return value

    llm_provider = os.environ.get("LLM_PROVIDER", "openai")

    # Only require OPENAI_API_KEY if using OpenAI provider
    if llm_provider == "openai":
        openai_key = _require("OPENAI_API_KEY")
    else:
        openai_key = os.environ.get("OPENAI_API_KEY", "")

    return Config(
        openai_api_key=openai_key,
        langchain_api_key=os.environ.get("LANGCHAIN_API_KEY", ""),
        langchain_project=os.environ.get("LANGCHAIN_PROJECT", "call-center-intelligence"),
        db_encryption_key=os.environ.get("DB_ENCRYPTION_KEY", _DEFAULT_DB_KEY),
        db_path=Path(os.environ.get("DB_PATH", "data/calls.db")),
        llm_provider=llm_provider,
        max_cost_per_call_usd=float(os.environ.get("MAX_COST_PER_CALL_USD", "2.00")),
        max_retries_per_node=int(os.environ.get("MAX_RETRIES_PER_NODE", "3")),
        llm_timeout_seconds=int(os.environ.get("LLM_TIMEOUT_SECONDS", "120")),
        whisper_model_size=os.environ.get("WHISPER_MODEL_SIZE", "tiny"),
        confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3")),
        low_confidence_halt_ratio=float(os.environ.get("LOW_CONFIDENCE_HALT_RATIO", "0.8")),
    )
