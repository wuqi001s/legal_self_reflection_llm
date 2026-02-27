#!/usr/bin/env python3
"""
Inference Script
Run Agentic RAG with legal LLM
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import TwoStageRetriever, FAISSRetriever, HyDEGenerator, BGEReranker
from src.inference.agent import LegalAgent


def load_models(model_path: str, embedding_model_path: str):
    """Load LLM and embedding models"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch
    
    print(f"Loading tokenizer and model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    class SimpleLLM:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            
        def generate(self, prompt, max_tokens=512):
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("assistant")[-1].strip()
    
    print(f"Loading embedding model from: {embedding_model_path}")
    embedding_model = SentenceTransformer(embedding_model_path)
    
    class EmbedWrapper:
        def __init__(self, model):
            self.model = model
            
        def encode(self, texts):
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
    
    return SimpleLLM(model, tokenizer), EmbedWrapper(embedding_model)


def build_retriever(embedding_model):
    """Build two-stage retriever"""
    faiss_retriever = FAISSRetriever(embedding_model)
    reranker = BGEReranker()
    
    return TwoStageRetriever(embedding_model, reranker)


def main():
    parser = argparse.ArgumentParser(description="Legal LLM Inference")
    parser.add_argument("--model-path", default="/home/y/project/hgmodels/Qwen3-0.6B", help="Model path")
    parser.add_argument("--embedding-path", default="/home/y/project/hgmodels/bge-small-zh-v1.5", 
                        help="Embedding model path")
    parser.add_argument("--index-path", default="./data/legal_faiss.index", help="FAISS index")
    parser.add_argument("--query", type=str, help="Query to answer")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    print("Loading models...")
    llm, embedding_model = load_models(args.model_path, args.embedding_path)
    
    print("Building retriever...")
    retriever = build_retriever(embedding_model)
    
    try:
        retriever.coarse_retriever.load_index(args.index_path)
        print(f"Loaded FAISS index from {args.index_path}")
    except Exception as e:
        print(f"Warning: Could not load index: {e}")
        print("Please build index first with: python scripts/build_index.py")
    
    agent = LegalAgent(retriever=retriever, llm=llm, max_iterations=3)
    
    if args.interactive:
        print("\n=== Legal LLM Interactive Mode ===")
        print("Type 'exit' to quit\n")
        
        while True:
            query = input("Query: ").strip()
            if query.lower() == "exit":
                break
                
            state = agent.run(query)
            print(f"\nAnswer: {state.final_answer}\n")
            print(f"Retrieved docs: {len(state.retrieved_docs)}")
            print(f"Iterations: {state.iteration}")
            print("-" * 50)
            
    elif args.query:
        state = agent.run(args.query)
        print(f"Answer: {state.final_answer}")
        print(f"Retrieved: {len(state.retrieved_docs)} docs")


if __name__ == "__main__":
    main()
