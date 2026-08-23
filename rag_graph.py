import os
from typing import List, Dict, Any, Optional, TypedDict
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from config import RAGConfig
from vectorstore import VectorStoreManager, get_embeddings


class GraphState(TypedDict):
    """Represents the state of our RAG execution graph."""
    question: str
    documents: List[Document]
    generation: str
    sources: List[Dict[str, Any]]
    steps_log: List[Dict[str, str]]
    retry_count: int
    is_relevant: bool


def get_chat_model(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    streaming: bool = False
) -> ChatOpenAI:
    """Creates a ChatOpenAI instance pointed at OpenRouter API."""
    resolved_key = RAGConfig.get_api_key(api_key)
    if not resolved_key:
        raise ValueError("OpenRouter API Key is missing. Please set OPENROUTER_API_KEY or provide it in the UI.")
    
    resolved_model = RAGConfig.get_model(model_name)
    resolved_base_url = RAGConfig.get_base_url()

    return ChatOpenAI(
        api_key=resolved_key,
        base_url=resolved_base_url,
        model=resolved_model,
        temperature=temperature,
        streaming=streaming,
        default_headers={
            "HTTP-Referer": "https://openrouter.ai",
            "X-Title": "LangGraph RAG Assistant"
        }
    )


class RAGPipeline:
    """Compiles and executes the LangGraph RAG workflow."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        top_k: Optional[int] = None
    ):
        self.api_key = RAGConfig.get_api_key(api_key)
        self.model_name = RAGConfig.get_model(model_name)
        self.embedding_model = RAGConfig.get_embedding_model(embedding_model)
        self.top_k = top_k or RAGConfig.get_top_k()
        self.vector_manager = VectorStoreManager()
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    def _build_graph(self) -> StateGraph:
        """Constructs the StateGraph with nodes and conditional routing."""
        workflow = StateGraph(GraphState)

        # Define Graph Nodes
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("grade_documents", self._node_grade_documents)
        workflow.add_node("transform_query", self._node_transform_query)
        workflow.add_node("generate", self._node_generate)

        # Define Edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge after grading
        workflow.add_conditional_edges(
            "grade_documents",
            self._route_after_grading,
            {
                "generate": "generate",
                "transform_query": "transform_query"
            }
        )
        
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", END)

        return workflow

    def _node_retrieve(self, state: GraphState) -> Dict[str, Any]:
        """Retrieves relevant document chunks from FAISS vector store."""
        question = state["question"]
        steps = list(state.get("steps_log", []))

        if not self.vector_manager.index_exists():
            steps.append({
                "node": "Retrieve",
                "status": "Warning",
                "message": "No documents uploaded or FAISS index is empty."
            })
            return {"documents": [], "steps_log": steps}

        vectorstore = self.vector_manager.load_index(
            api_key=self.api_key,
            embedding_model=self.embedding_model
        )

        if not vectorstore:
            steps.append({
                "node": "Retrieve",
                "status": "Warning",
                "message": "Failed to load FAISS index."
            })
            return {"documents": [], "steps_log": steps}

        retriever = vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        docs = retriever.invoke(question)

        steps.append({
            "node": "Retrieve",
            "status": "Success",
            "message": f"Retrieved {len(docs)} chunk(s) from FAISS index for query: '{question}'"
        })

        return {"documents": docs, "steps_log": steps}

    def _node_grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """Evaluates whether retrieved documents are relevant to the question."""
        question = state["question"]
        documents = state.get("documents", [])
        steps = list(state.get("steps_log", []))

        if not documents:
            steps.append({
                "node": "Grade Documents",
                "status": "Info",
                "message": "No documents to evaluate."
            })
            return {"documents": [], "is_relevant": False, "steps_log": steps}

        # Fast prompt to grade document relevance
        llm = get_chat_model(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.0
        )

        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a document relevance grader. Evaluate if the retrieved document contains info relevant to the user question. Answer ONLY with 'yes' or 'no'."),
            ("human", "User Question: {question}\n\nDocument snippet:\n{document}\n\nIs this document relevant to answering the question?")
        ])

        grader_chain = grade_prompt | llm
        filtered_docs = []

        for doc in documents:
            try:
                response = grader_chain.invoke({
                    "question": question,
                    "document": doc.page_content[:600]
                })
                decision = response.content.strip().lower()
                if "yes" in decision:
                    filtered_docs.append(doc)
            except Exception:
                # If grading call fails (e.g. rate limit), keep the document safely
                filtered_docs.append(doc)

        is_relevant = len(filtered_docs) > 0
        steps.append({
            "node": "Grade Documents",
            "status": "Success" if is_relevant else "Notice",
            "message": f"Filtered to {len(filtered_docs)} relevant chunk(s) out of {len(documents)} retrieved."
        })

        # If none passed strict grading, retain original docs as fallback rather than dropping all
        final_docs = filtered_docs if filtered_docs else documents

        return {
            "documents": final_docs,
            "is_relevant": is_relevant,
            "steps_log": steps
        }

    def _route_after_grading(self, state: GraphState) -> str:
        """Determines whether to proceed to generation or rewrite query."""
        is_relevant = state.get("is_relevant", False)
        retry_count = state.get("retry_count", 0)

        if is_relevant or retry_count >= 1:
            return "generate"
        return "transform_query"

    def _node_transform_query(self, state: GraphState) -> Dict[str, Any]:
        """Rewrites the query to improve semantic retrieval quality."""
        question = state["question"]
        steps = list(state.get("steps_log", []))
        retry_count = state.get("retry_count", 0)

        llm = get_chat_model(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.3
        )

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that optimizes search queries. Rephrase the input question to be a clear, keyword-rich semantic search query. Respond with ONLY the reformulated query, without any preamble or quotes."),
            ("human", "Original Question: {question}")
        ])

        try:
            rewrite_chain = rewrite_prompt | llm
            rewritten_response = rewrite_chain.invoke({"question": question})
            new_question = rewritten_response.content.strip()
        except Exception:
            new_question = question

        steps.append({
            "node": "Transform Query",
            "status": "Info",
            "message": f"Reformulated query for second retrieval pass: '{new_question}'"
        })

        return {
            "question": new_question,
            "retry_count": retry_count + 1,
            "steps_log": steps
        }

    def _node_generate(self, state: GraphState) -> Dict[str, Any]:
        """Generates the final grounded answer with citations."""
        question = state["question"]
        documents = state.get("documents", [])
        steps = list(state.get("steps_log", []))

        if not documents:
            generation = (
                "I could not find any relevant information in the uploaded documents to answer your question. "
                "Please make sure you have uploaded the appropriate documents and that your question relates to their content."
            )
            steps.append({
                "node": "Generate",
                "status": "Notice",
                "message": "Generated fallback response (no context available)."
            })
            return {
                "generation": generation,
                "sources": [],
                "steps_log": steps
            }

        # Build context string with clear source annotations
        context_parts = []
        sources = []
        seen_snippets = set()

        for idx, doc in enumerate(documents, start=1):
            source_file = doc.metadata.get("filename", doc.metadata.get("source", f"Doc {idx}"))
            page_info = f", Page {doc.metadata.get('page') + 1}" if "page" in doc.metadata and doc.metadata.get('page') is not None else ""
            source_label = f"{Path(source_file).name}{page_info}"
            
            snippet = doc.page_content.strip()
            context_parts.append(f"--- Document [{idx}] ({source_label}) ---\n{snippet}")

            # Collect source metadata
            snippet_hash = hash(snippet[:100])
            if snippet_hash not in seen_snippets:
                seen_snippets.add(snippet_hash)
                sources.append({
                    "id": idx,
                    "source": source_label,
                    "filename": Path(source_file).name,
                    "page": doc.metadata.get("page"),
                    "content_preview": snippet[:300] + ("..." if len(snippet) > 300 else "")
                })

        context_str = "\n\n".join(context_parts)

        llm = get_chat_model(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.2
        )

        system_instruction = (
            "You are an expert AI knowledge assistant.\n"
            "Answer the user's question accurately and thoroughly based ONLY on the provided context documents.\n"
            "Rules:\n"
            "1. Ground your answer strictly in the provided context.\n"
            "2. Whenever mentioning key facts or findings, cite the source using brackets like `[Source: filename, Page X]` or `[Document X]`.\n"
            "3. If the context does not provide sufficient details to fully answer, state what is known and clarify what is missing.\n"
            "4. Structure your response clearly with markdown formatting, headings, and bullet points where helpful."
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "Context Documents:\n{context}\n\nUser Question:\n{question}\n\nAnswer:")
        ])

        generate_chain = qa_prompt | llm

        try:
            response = generate_chain.invoke({
                "context": context_str,
                "question": question
            })
            generation = response.content
        except Exception as e:
            generation = f"An error occurred while generating the answer: {str(e)}"

        steps.append({
            "node": "Generate",
            "status": "Success",
            "message": f"Successfully generated answer citing {len(sources)} source chunk(s)."
        })

        return {
            "generation": generation,
            "sources": sources,
            "steps_log": steps
        }

    def run(self, question: str) -> Dict[str, Any]:
        """Executes the full LangGraph RAG pipeline synchronously."""
        initial_state: GraphState = {
            "question": question,
            "documents": [],
            "generation": "",
            "sources": [],
            "steps_log": [],
            "retry_count": 0,
            "is_relevant": False
        }

        final_state = self.app.invoke(initial_state)
        return {
            "question": question,
            "answer": final_state.get("generation", ""),
            "sources": final_state.get("sources", []),
            "steps": final_state.get("steps_log", []),
            "documents": final_state.get("documents", [])
        }
