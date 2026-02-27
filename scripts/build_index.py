#!/usr/bin/env python3
"""
Build Retrieval Index
Build FAISS index from Civil Code clauses
"""

import json
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import FAISSRetriever
from sentence_transformers import SentenceTransformer
import numpy as np


def build_index(documents_file: str, output_dir: str, embedding_model_path: str):
    """Build FAISS index from legal documents"""
    
    with open(documents_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    documents = [item["content"] for item in data]
    metadata = [item for item in data]
    
    print(f"Building index for {len(documents)} documents...")
    print(f"Using embedding model: {embedding_model_path}")
    
    # Load local embedding model
    embedding_model = SentenceTransformer(embedding_model_path)
    
    # Create FAISS retriever
    retriever = FAISSRetriever(embedding_model)
    retriever.build_index(documents, metadata)
    
    # Save index
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    retriever.save_index(
        str(output_path / "legal_faiss.index"),
        str(output_path / "documents.json")
    )
    
    print(f"Index saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS Index")
    parser.add_argument("--documents", default="./data/civil_code_clauses.json", help="Documents JSON file")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--embedding-model", default="/home/y/project/hgmodels/bge-small-zh-v1.5", 
                        help="Embedding model path")
    
    args = parser.parse_args()
    build_index(args.documents, args.output, args.embedding_model)


if __name__ == "__main__":
    main()
