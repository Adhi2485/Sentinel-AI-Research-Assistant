import faiss
import numpy as np
from rank_bm25 import BM25Okapi

class VectorStore:
    def reset(self):
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        self.tokenized_corpus = []
        self.bm25 = None

    def __init__(self, embedding_dim: int = 384):
        """
        Initializes FAISS index and BM25 index.
        """
        self.embedding_dim = embedding_dim
        self.reset()

        self.tokenized_corpus = []
        self.bm25 = None

    def add_vectors(self, vectors: list[list[float]], metadata_list: list[dict]):
        """Adds vectors and text to FAISS and BM25 indexes."""
        if not vectors:
            return
            
        vectors_np = np.array(vectors, dtype=np.float32)
        self.index.add(vectors_np)
        
        for m in metadata_list:
            self.metadata.append(m)
            # Tokenize for BM25
            self.tokenized_corpus.append(m["text"].lower().split())
            
        # Rebuild BM25 index
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def search(self, query_vector: list[float], query_text: str, top_k: int = 5) -> list[dict]:
        """Hybrid search combining FAISS and BM25."""
        if self.index.ntotal == 0:
            return []
            
        # 1. FAISS Search
        query_np = np.array([query_vector], dtype=np.float32)
        faiss_distances, faiss_indices = self.index.search(query_np, top_k * 2) # Fetch more for reranking
        
        faiss_results = set()
        for i, idx in enumerate(faiss_indices[0]):
            if idx != -1 and idx < len(self.metadata):
                faiss_results.add(idx)
                
        # 2. BM25 Search
        bm25_results = set()
        if self.bm25:
            tokenized_query = query_text.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            # Get top_k * 2 indices
            top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k * 2]
            for idx in top_bm25_idx:
                if bm25_scores[idx] > 0:
                    bm25_results.add(idx)
                    
        # 3. Combine unique indices
        combined_indices = list(faiss_results.union(bm25_results))
        
        results = []
        for idx in combined_indices:
            results.append({
                "chunk": self.metadata[idx]
            })
            
        return results
