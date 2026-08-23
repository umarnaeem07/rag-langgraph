import os
import sys
import unittest
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import RAGConfig
from vectorstore import VectorStoreManager
from rag_graph import RAGPipeline


class MockEmbeddings(Embeddings):
    """Deterministic local mock embeddings for testing vector operations without API keys."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10) for _ in range(16)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 10) for _ in range(16)]


class TestLangGraphRAG(unittest.TestCase):

    def setUp(self):
        self.test_dir = "./test_faiss_index"
        self.manager = VectorStoreManager(index_dir=self.test_dir)

    def tearDown(self):
        self.manager.clear_index()

    def test_document_splitting_and_indexing(self):
        """Test document parsing, splitting, and FAISS indexing."""
        sample_doc = Document(
            page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. " * 20,
            metadata={"filename": "test_guide.txt", "source": "test_guide.txt"}
        )
        chunks = self.manager.split_documents([sample_doc], chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)

        # Test FAISS from documents with MockEmbeddings
        from langchain_community.vectorstores import FAISS
        mock_emb = MockEmbeddings()
        vs = FAISS.from_documents(chunks, mock_emb)
        vs.save_local(self.test_dir)

        self.assertTrue(self.manager.index_exists())

        # Test retrieval
        loaded_vs = FAISS.load_local(self.test_dir, mock_emb, allow_dangerous_deserialization=True)
        retriever = loaded_vs.as_retriever(search_kwargs={"k": 2})
        retrieved_docs = retriever.invoke("LangGraph stateful")
        self.assertGreater(len(retrieved_docs), 0)

    def test_rag_graph_compilation(self):
        """Test that the LangGraph StateGraph builds and compiles properly."""
        # Create pipeline with a placeholder test key
        pipeline = RAGPipeline(api_key="sk-or-test-dummy-key")
        self.assertIsNotNone(pipeline.app)
        
        # Verify node names in the compiled graph
        node_keys = list(pipeline.workflow.nodes.keys())
        self.assertIn("retrieve", node_keys)
        self.assertIn("grade_documents", node_keys)
        self.assertIn("transform_query", node_keys)
        self.assertIn("generate", node_keys)


if __name__ == "__main__":
    unittest.main()
