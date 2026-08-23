import os
import streamlit as st
from pathlib import Path
from typing import List

from config import RAGConfig
from vectorstore import VectorStoreManager
from rag_graph import RAGPipeline

# Page setup
st.set_page_config(
    page_title="LangGraph RAG Assistant (OpenRouter & FAISS)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS styling
st.markdown("""
<style>
    /* Gradient headers and card styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .stat-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .step-pill {
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        background: #1e293b;
        border-left: 4px solid #6366f1;
        color: #e2e8f0;
    }
    .source-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
    }
    .source-title {
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    .source-snippet {
        color: #cbd5e1;
        font-size: 0.82rem;
        font-family: monospace;
        background: rgba(0,0,0,0.2);
        padding: 0.4rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = VectorStoreManager()

# --- SIDEBAR: Configuration & Knowledge Base Stats ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    saved_key = RAGConfig.get_api_key()
    api_key_input = st.text_input(
        "OpenRouter API Key",
        value=saved_key,
        type="password",
        help="Enter your OpenRouter API Key (starts with sk-or-...). Get one at https://openrouter.ai/keys",
        placeholder="sk-or-v1-..."
    )
    
    if not api_key_input:
        st.warning("⚠️ OpenRouter API Key is required to run embeddings and queries.")
    else:
        st.success("✅ API Key configured")

    st.markdown("---")
    st.subheader("🤖 Model Selection")
    
    # Model Options
    popular_models = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "deepseek/deepseek-chat",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
        "Custom..."
    ]
    
    selected_model_option = st.selectbox(
        "LLM Model",
        options=popular_models,
        index=0,
        help="Select any model hosted on OpenRouter"
    )
    
    if selected_model_option == "Custom...":
        selected_model = st.text_input("Enter Model Name", value="openai/gpt-4o-mini")
    else:
        selected_model = selected_model_option

    # Embedding model selection
    embedding_models = [
        "openai/text-embedding-3-small",
        "text-embedding-3-small",
        "openai/text-embedding-3-large",
        "text-embedding-ada-002",
        "Custom..."
    ]
    
    selected_emb_option = st.selectbox(
        "Embedding Model",
        options=embedding_models,
        index=0,
        help="Model used to generate vector embeddings"
    )
    
    if selected_emb_option == "Custom...":
        selected_embedding_model = st.text_input("Enter Embedding Model Name", value="openai/text-embedding-3-small")
    else:
        selected_embedding_model = selected_emb_option

    top_k = st.slider("Retrieval Top-K Chunks", min_value=1, max_value=10, value=4)

    st.markdown("---")
    st.subheader("📚 FAISS Knowledge Base")
    
    stats = st.session_state.vector_manager.get_stats(api_key=api_key_input)
    if stats["exists"]:
        st.markdown(f"<span class='stat-badge'>📊 Indexed Chunks: {stats['total_vectors']}</span>", unsafe_allow_html=True)
        if stats["sources"]:
            st.write("**Indexed Documents:**")
            for src in stats["sources"]:
                st.markdown(f"- 📄 `{src}`")
    else:
        st.info("No documents currently indexed in FAISS.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Stats", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Index", type="secondary", use_container_width=True):
            st.session_state.vector_manager.clear_index()
            st.session_state.messages = []
            st.success("FAISS index cleared.")
            st.rerun()


# --- MAIN AREA ---
st.markdown("<div class='main-title'>🧠 LangGraph RAG Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Upload documents (PDF, DOCX, TXT, MD, CSV), embed with OpenRouter + FAISS, and query using a LangGraph workflow.</div>",
    unsafe_allow_html=True
)

# --- SECTION 1: Document Upload & Ingestion ---
with st.expander("📤 **Upload & Index Documents**", expanded=not st.session_state.vector_manager.index_exists()):
    uploaded_files = st.file_uploader(
        "Choose documents to add to FAISS vector database",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Upload one or multiple documents to index"
    )

    if uploaded_files:
        st.write(f"Selected **{len(uploaded_files)}** file(s):")
        for f in uploaded_files:
            st.markdown(f"- 📎 `{f.name}` ({round(f.size / 1024, 1)} KB)")

        if st.button("🚀 Process & Index to FAISS", type="primary", use_container_width=True):
            if not api_key_input:
                st.error("Please enter your OpenRouter API Key in the sidebar first!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_docs = []
                total_files = len(uploaded_files)

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Parsing [{i+1}/{total_files}]: {uploaded_file.name}...")
                    file_bytes = uploaded_file.read()
                    docs = st.session_state.vector_manager.load_document_from_bytes(
                        file_bytes, uploaded_file.name
                    )
                    all_docs.extend(docs)
                    progress_bar.progress(int((i + 1) / total_files * 50))

                status_text.text(f"Splitting into chunks and embedding via {selected_embedding_model}...")
                try:
                    num_docs, num_chunks = st.session_state.vector_manager.add_documents_to_index(
                        docs=all_docs,
                        api_key=api_key_input,
                        embedding_model=selected_embedding_model
                    )
                    progress_bar.progress(100)
                    status_text.empty()
                    st.success(f"✅ Successfully indexed **{num_docs}** file(s) into **{num_chunks}** FAISS vector chunks!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")


# --- SECTION 2: Chat Interface ---
st.markdown("### 💬 Ask Questions About Your Documents")

# Render previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display execution trace if present
        if "steps" in msg and msg["steps"]:
            with st.expander("⚡ **LangGraph Execution Trace**", expanded=False):
                for step in msg["steps"]:
                    st.markdown(f"<div class='step-pill'><b>[{step.get('node', 'Node')}]</b> ({step.get('status', '')}): {step.get('message', '')}</div>", unsafe_allow_html=True)

        # Display source citations if present
        if "sources" in msg and msg["sources"]:
            with st.expander("📑 **Source References & Citations**", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class='source-card'>
                        <div class='source-title'>📄 {src['source']}</div>
                        <div class='source-snippet'>{src['content_preview']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input
prompt = st.chat_input("Ask a question about your uploaded documents...")

if prompt:
    if not api_key_input:
        st.error("Please enter your OpenRouter API Key in the sidebar before querying!")
    elif not st.session_state.vector_manager.index_exists():
        st.warning("Please upload and index at least one document first!")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Execute LangGraph RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("🤖 Executing LangGraph workflow (Retrieve -> Grade -> Generate)..."):
                pipeline = RAGPipeline(
                    api_key=api_key_input,
                    model_name=selected_model,
                    embedding_model=selected_embedding_model,
                    top_k=top_k
                )

                try:
                    result = pipeline.run(prompt)
                    answer = result.get("answer", "No answer generated.")
                    sources = result.get("sources", [])
                    steps = result.get("steps", [])

                    st.markdown(answer)

                    # Show Graph Steps
                    if steps:
                        with st.expander("⚡ **LangGraph Execution Trace**", expanded=True):
                            for step in steps:
                                st.markdown(f"<div class='step-pill'><b>[{step.get('node', 'Node')}]</b> ({step.get('status', '')}): {step.get('message', '')}</div>", unsafe_allow_html=True)

                    # Show Sources
                    if sources:
                        with st.expander("📑 **Source References & Citations**", expanded=True):
                            for src in sources:
                                st.markdown(f"""
                                <div class='source-card'>
                                    <div class='source-title'>📄 {src['source']}</div>
                                    <div class='source-snippet'>{src['content_preview']}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "steps": steps
                    })

                except Exception as e:
                    error_msg = f"❌ An error occurred during graph execution: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
