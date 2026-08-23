import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

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
        """Returns the OpenRouter API key from explicit argument or environment."""
        key = (override_key or "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
        return key

    @staticmethod
    def get_base_url() -> str:
        """Returns the OpenRouter base URL."""
        return os.getenv("OPENROUTER_BASE_URL", RAGConfig.DEFAULT_OPENROUTER_BASE_URL).strip()

    @staticmethod
    def get_model(override_model: Optional[str] = None) -> str:
        """Returns the LLM model name."""
        return (override_model or "").strip() or os.getenv("OPENROUTER_MODEL", RAGConfig.DEFAULT_MODEL).strip()

    @staticmethod
    def get_embedding_model(override_model: Optional[str] = None) -> str:
        """Returns the embedding model name."""
        return (override_model or "").strip() or os.getenv("OPENROUTER_EMBEDDING_MODEL", RAGConfig.DEFAULT_EMBEDDING_MODEL).strip()

    @staticmethod
    def get_index_dir() -> str:
        """Returns the path to the FAISS index directory."""
        return os.getenv("FAISS_INDEX_DIR", RAGConfig.DEFAULT_FAISS_INDEX_DIR).strip()

    @staticmethod
    def get_chunk_size() -> int:
        """Returns text chunk size."""
        try:
            return int(os.getenv("CHUNK_SIZE", str(RAGConfig.DEFAULT_CHUNK_SIZE)))
        except ValueError:
            return RAGConfig.DEFAULT_CHUNK_SIZE

    @staticmethod
    def get_chunk_overlap() -> int:
        """Returns text chunk overlap."""
        try:
            return int(os.getenv("CHUNK_OVERLAP", str(RAGConfig.DEFAULT_CHUNK_OVERLAP)))
        except ValueError:
            return RAGConfig.DEFAULT_CHUNK_OVERLAP

    @staticmethod
    def get_top_k() -> int:
        """Returns number of documents to retrieve."""
        try:
            return int(os.getenv("TOP_K_RETRIEVAL", str(RAGConfig.DEFAULT_TOP_K)))
        except ValueError:
            return RAGConfig.DEFAULT_TOP_K
