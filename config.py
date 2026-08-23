import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def _clean_api_key(key: str) -> str:
    """Cleans API key by stripping quotes, whitespace, and potential 'Bearer ' prefix."""
    if not key:
        return ""
    cleaned = str(key).strip().strip("\"'").strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip().strip("\"'").strip()
    return cleaned


def _get_val(env_key: str, default: str = "") -> str:
    """Helper to get a configuration value from os.environ or streamlit secrets."""
    # 1. Check OS environment variable
    val = os.getenv(env_key, "").strip()
    if val:
        return val.strip("\"'")

    # 2. Check Streamlit secrets (for Streamlit Community Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and env_key in st.secrets:
            s_val = str(st.secrets[env_key]).strip().strip("\"'")
            if s_val:
                return s_val
    except Exception:
        pass

    return default


class RAGConfig:
    DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openai/gpt-4o-mini"
    DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
    DEFAULT_FAISS_INDEX_DIR = "./faiss_index"
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 150
    DEFAULT_TOP_K = 4

    @staticmethod
    def get_api_key(override_key: Optional[str] = None) -> str:
        """Returns the OpenRouter API key from explicit argument, environment, or Streamlit secrets."""
        if override_key and override_key.strip():
            return _clean_api_key(override_key)
        return _clean_api_key(_get_val("OPENROUTER_API_KEY", ""))

    @staticmethod
    def get_base_url() -> str:
        """Returns the OpenRouter base URL."""
        return _get_val("OPENROUTER_BASE_URL", RAGConfig.DEFAULT_OPENROUTER_BASE_URL)

    @staticmethod
    def get_model(override_model: Optional[str] = None) -> str:
        """Returns the LLM model name."""
        if override_model and override_model.strip():
            return override_model.strip().strip("\"'")
        return _get_val("OPENROUTER_MODEL", RAGConfig.DEFAULT_MODEL)

    @staticmethod
    def get_embedding_model(override_model: Optional[str] = None) -> str:
        """Returns the embedding model name."""
        if override_model and override_model.strip():
            return override_model.strip().strip("\"'")
        return _get_val("OPENROUTER_EMBEDDING_MODEL", RAGConfig.DEFAULT_EMBEDDING_MODEL)

    @staticmethod
    def get_index_dir() -> str:
        """Returns the path to the FAISS index directory."""
        return _get_val("FAISS_INDEX_DIR", RAGConfig.DEFAULT_FAISS_INDEX_DIR)

    @staticmethod
    def get_chunk_size() -> int:
        """Returns text chunk size."""
        raw = _get_val("CHUNK_SIZE", str(RAGConfig.DEFAULT_CHUNK_SIZE))
        try:
            return int(raw)
        except ValueError:
            return RAGConfig.DEFAULT_CHUNK_SIZE

    @staticmethod
    def get_chunk_overlap() -> int:
        """Returns text chunk overlap."""
        raw = _get_val("CHUNK_OVERLAP", str(RAGConfig.DEFAULT_CHUNK_OVERLAP))
        try:
            return int(raw)
        except ValueError:
            return RAGConfig.DEFAULT_CHUNK_OVERLAP

    @staticmethod
    def get_top_k() -> int:
        """Returns number of documents to retrieve."""
        raw = _get_val("TOP_K_RETRIEVAL", str(RAGConfig.DEFAULT_TOP_K))
        try:
            return int(raw)
        except ValueError:
            return RAGConfig.DEFAULT_TOP_K
