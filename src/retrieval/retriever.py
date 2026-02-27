import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RetrievedDocument:
    """Retrieved document with score"""
    content: str
    score: float
    metadata: Dict[str, Any]


class HyDEGenerator:
    """
    HyDE (Hypothetical Document Embeddings) 
    Bridges colloquial queries and legal language
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        
    def generate_hypothetical_doc(self, query: str) -> str:
        """Generate hypothetical legal document for query"""
        if not self.llm:
            return self._template_hypothetical(query)
            
        prompt = f"""你是一个法律专家。请根据以下用户问题，写出一个可能包含答案的法律文档片段。
要求：使用正式的法言法语，包含法律条文引用。

用户问题: {query}

法律文档片段:"""
        
        return self.llm.generate(prompt)
    
    def _template_hypothetical(self, query: str) -> str:
        """Template-based hypothetical when LLM not available"""
        return f"根据相关法律规定，关于{query}，应当按照以下条款处理："


class FAISSRetriever:
    """FAISS vector store for coarse retrieval"""
    
    def __init__(self, embedding_model=None, index_path: Optional[str] = None):
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []
        self.metadata = []
        
        if index_path:
            self.load_index(index_path)
            
    def build_index(self, documents: List[str], metadata: List[Dict]):
        """Build FAISS index from documents"""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed")
            
        self.documents = documents
        self.metadata = metadata
        
        embeddings = self._encode_documents(documents)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
    def _encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode documents to embeddings"""
        if self.embedding_model:
            return self.embedding_model.encode(documents)
        return np.random.randn(len(documents), 768).astype('float32')
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding"""
        if self.embedding_model:
            return self.embedding_model.encode([query])
        return np.random.randn(1, 768).astype('float32')
    
    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievedDocument]:
        """Retrieve top-k documents using FAISS"""
        if not self.index:
            return []
            
        query_embedding = self._encode_query(query)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append(RetrievedDocument(
                    content=self.documents[idx],
                    score=float(1.0 / (1.0 + dist)),
                    metadata=self.metadata[idx]
                ))
                
        return results
    
    def save_index(self, index_path: str, docs_path: str):
        """Save FAISS index and documents"""
        if self.index:
            import faiss
            faiss.write_index(self.index, index_path)
            
        import json
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f, ensure_ascii=False)
    
    def load_index(self, index_path: str):
        """Load FAISS index"""
        try:
            import faiss
            self.index = faiss.read_index(index_path)
        except:
            pass


class BGEReranker:
    """
    BGE-Reranker for cross-encoder fine ranking
    Eliminates context blindness
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = None
        self.model_name = model_name
        self._load_model()
        
    def _load_model(self):
        """Load reranker model"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except ImportError:
            print("sentence-transformers not installed, using mock reranker")
    
    def rerank(self, query: str, documents: List[RetrievedDocument], top_k: int = 5) -> List[RetrievedDocument]:
        """Rerank documents using cross-encoder"""
        if not self.model:
            return documents[:top_k]
            
        doc_contents = [doc.content for doc in documents]
        pairs = [(query, doc) for doc in doc_contents]
        
        scores = self.model.predict(pairs)
        
        for doc, score in zip(documents, scores):
            doc.score = float(score)
            
        sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
        return sorted_docs[:top_k]


class TwoStageRetriever:
    """Two-stage retrieval: FAISS coarse + BGE rerank"""
    
    def __init__(self, embedding_model=None, reranker=None):
        self.coarse_retriever = FAISSRetriever(embedding_model)
        self.reranker = reranker or BGEReranker()
        
    def retrieve(self, query: str,hyde_doc: Optional[str] = None, top_k: int = 5) -> List[RetrievedDocument]:
        """
        Two-stage retrieval:
        1. Use HyDE to generate hypothetical doc
        2. FAISS coarse search
        3. BGE rerank
        """
        search_query = query
        if hyde_doc:
            search_query = hyde_doc
            
        coarse_results = self.coarse_retriever.retrieve(search_query, top_k=20)
        
        if not coarse_results:
            return []
            
        final_results = self.reranker.rerank(query, coarse_results, top_k=top_k)
        
        return final_results
    
    def build_index(self, documents: List[str], metadata: List[Dict]):
        """Build the two-stage index"""
        self.coarse_retriever.build_index(documents, metadata)
