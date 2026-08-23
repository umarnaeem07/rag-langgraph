import os
import sys
import argparse
from typing import List, Optional
from pathlib import Path

from config import RAGConfig
from vectorstore import VectorStoreManager
from rag_graph import RAGPipeline


def index_files(file_paths: List[str], api_key: Optional[str] = None):
    """Indexes specified file paths into FAISS."""
    manager = VectorStoreManager()
    all_docs = []

    print(f"[*] Processing {len(file_paths)} file(s)...")
    for path_str in file_paths:
        p = Path(path_str)
        if not p.exists():
            print(f"[!] File not found: {path_str}")
            continue
        print(f"  -> Loading {p.name}...")
        docs = manager.load_document_from_path(str(p))
        all_docs.extend(docs)

    if not all_docs:
        print("[!] No documents were loaded.")
        return

    print(f"[*] Splitting and embedding {len(all_docs)} loaded pages/sections...")
    num_docs, num_chunks = manager.add_documents_to_index(all_docs, api_key=api_key)
    print(f"[✓] Successfully indexed {num_docs} document(s) into {num_chunks} vector chunks in FAISS!\n")


def query_rag(question: str, api_key: Optional[str] = None, model: Optional[str] = None):
    """Queries the LangGraph RAG pipeline."""
    print(f"\n=========================================")
    print(f"❓ Question: {question}")
    print(f"=========================================\n")

    pipeline = RAGPipeline(api_key=api_key, model_name=model)
    print("[*] Running LangGraph RAG pipeline...")
    result = pipeline.run(question)

    print("\n--- ⚡ LangGraph Execution Steps ---")
    for step in result.get("steps", []):
        print(f"[{step.get('node')}] ({step.get('status')}): {step.get('message')}")

    print("\n--- 🤖 Answer ---")
    print(result.get("answer"))

    sources = result.get("sources", [])
    if sources:
        print("\n--- 📑 Sources Cited ---")
        for src in sources:
            print(f"- {src['source']}")
            print(f"  Preview: {src['content_preview'][:150]}...")
    print("\n=========================================\n")


def interactive_cli():
    """Interactive loop in terminal."""
    api_key = RAGConfig.get_api_key()
    if not api_key:
        api_key = input("Enter your OpenRouter API Key: ").strip()

    manager = VectorStoreManager()
    stats = manager.get_stats(api_key=api_key)
    print("==================================================")
    print("🧠 LangGraph RAG CLI Assistant")
    print(f"FAISS Index Status: {'Ready (' + str(stats['total_vectors']) + ' chunks)' if stats['exists'] else 'Empty'}")
    print("==================================================")
    print("Commands:")
    print("  /index <file_or_dir_path>   -> Index a document or folder")
    print("  /stats                      -> View indexed vector stats")
    print("  /clear                      -> Reset FAISS index")
    print("  /exit                       -> Quit")
    print("  Or type any question to query the indexed documents.\n")

    while True:
        try:
            user_input = input("RAG > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["/exit", "exit", "quit"]:
                print("Goodbye!")
                break
            elif user_input.startswith("/index "):
                target = user_input[7:].strip()
                path = Path(target)
                if path.is_dir():
                    files = [str(f) for f in path.glob("*") if f.is_file() and f.suffix.lower() in [".pdf", ".docx", ".txt", ".md", ".csv"]]
                    index_files(files, api_key=api_key)
                elif path.is_file():
                    index_files([str(path)], api_key=api_key)
                else:
                    print(f"[!] Path does not exist: {target}")
            elif user_input == "/stats":
                s = manager.get_stats(api_key=api_key)
                print(f"Total Chunks: {s.get('total_vectors')}, Sources: {s.get('sources')}")
            elif user_input == "/clear":
                manager.clear_index()
                print("[✓] FAISS index cleared.")
            else:
                query_rag(user_input, api_key=api_key)
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"[!] Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="LangGraph RAG CLI with FAISS & OpenRouter")
    parser.add_argument("--index", nargs="+", help="Path to file(s) to index into FAISS")
    parser.add_argument("--query", "-q", type=str, help="Query the RAG pipeline with a question")
    parser.add_argument("--model", "-m", type=str, default=None, help="OpenRouter model name")
    parser.add_argument("--api-key", "-k", type=str, default=None, help="OpenRouter API Key")

    args = parser.parse_args()

    if args.index:
        index_files(args.index, api_key=args.api_key)
    elif args.query:
        query_rag(args.query, api_key=args.api_key, model=args.model)
    else:
        interactive_cli()


if __name__ == "__main__":
    main()
