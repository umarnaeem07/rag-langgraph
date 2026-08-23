import os
import shutil
import tempfile
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader
)

from config import RAGConfig


def get_embeddings(
    api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    base_url: Optional[str] = None
) -> OpenAIEmbeddings:
    """Instantiates OpenAIEmbeddings configured for OpenRouter / OpenAI API."""
    resolved_key = RAGConfig.get_api_key(api_key)
    if not resolved_key:
        raise ValueError("OpenRouter API Key is required. Please set OPENROUTER_API_KEY in .env or provide it.")
    
    resolved_model = RAGConfig.get_embedding_model(embedding_model)
    resolved_base_url = base_url or RAGConfig.get_base_url()

    return OpenAIEmbeddings(
        api_key=resolved_key,
        base_url=resolved_base_url,
        model=resolved_model,
        check_embedding_ctx_length=False,
        default_headers={
            "HTTP-Referer": "https://localhost:8501",
            "X-Title": "LangGraph RAG Assistant"
        }
    )


class VectorStoreManager:
    """Manages document loading, chunking, and FAISS vector index storage."""

    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = index_dir or RAGConfig.get_index_dir()

    def load_document_from_path(self, file_path: str) -> List[Document]:
        """Loads documents from a file path based on its extension."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        elif ext == ".csv":
            loader = CSVLoader(file_path, encoding="utf-8")
            docs = loader.load()
        elif ext in [".txt", ".md", ".markdown", ".json", ".log", ".py", ".html"]:
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            except UnicodeDecodeError:
                loader = TextLoader(file_path, autodetect_encoding=True)
                docs = loader.load()
        else:
            # Fallback text loader
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

        # Enhance metadata
        for doc in docs:
            doc.metadata["filename"] = path.name
            doc.metadata["source"] = str(path)

        return docs

    def load_document_from_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Saves uploaded file bytes to a temp file and parses it into Documents."""
        ext = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            docs = self.load_document_from_path(tmp_path)
            # Fix metadata so it shows the user's original uploaded filename
            for doc in docs:
                doc.metadata["filename"] = filename
                doc.metadata["source"] = filename
            return docs
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def split_documents(
        self,
        docs: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """Splits documents into smaller semantic chunks."""
        c_size = chunk_size or RAGConfig.get_chunk_size()
        c_overlap = chunk_overlap or RAGConfig.get_chunk_overlap()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        return chunks

    def index_exists(self) -> bool:
        """Checks if a FAISS index directory exists on disk with index files."""
        index_path = Path(self.index_dir)
        return (index_path / "index.faiss").exists() and (index_path / "index.pkl").exists()

    def load_index(
        self,
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> Optional[FAISS]:
        """Loads an existing FAISS index from disk."""
        if not self.index_exists():
            return None
        
        embeddings = get_embeddings(api_key=api_key, embedding_model=embedding_model)
        vectorstore = FAISS.load_local(
            self.index_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore

    def add_documents_to_index(
        self,
        docs: List[Document],
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Chunks documents, embeds them, and adds them to the FAISS index.
        Returns: (number_of_raw_docs, number_of_chunks_indexed)
        """
        if not docs:
            return 0, 0

        chunks = self.split_documents(docs, chunk_size, chunk_overlap)
        if not chunks:
            return len(docs), 0

        embeddings = get_embeddings(api_key=api_key, embedding_model=embedding_model)

        if self.index_exists():
            vectorstore = FAISS.load_local(
                self.index_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            vectorstore.add_documents(chunks)
        else:
            os.makedirs(self.index_dir, exist_ok=True)
            vectorstore = FAISS.from_documents(chunks, embeddings)

        vectorstore.save_local(self.index_dir)
        return len(docs), len(chunks)

    def clear_index(self) -> bool:
        """Deletes the local FAISS index folder."""
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir)
            return True
        return False

    def get_stats(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Returns statistics about the currently saved index."""
        if not self.index_exists():
            return {
                "exists": False,
                "total_vectors": 0,
                "sources": []
            }
        
        try:
            vs = self.load_index(api_key=api_key)
            if vs and vs.docstore:
                doc_dict = vs.docstore._dict
                sources = set()
                for doc in doc_dict.values():
                    if hasattr(doc, "metadata") and "filename" in doc.metadata:
                        sources.add(doc.metadata["filename"])
                    elif hasattr(doc, "metadata") and "source" in doc.metadata:
                        sources.add(Path(doc.metadata["source"]).name)
                
                return {
                    "exists": True,
                    "total_vectors": len(doc_dict),
                    "sources": sorted(list(sources))
                }
        except Exception as e:
            return {
                "exists": True,
                "total_vectors": "Available (requires valid key to inspect)",
                "sources": [],
                "error": str(e)
            }

        return {"exists": False, "total_vectors": 0, "sources": []}
