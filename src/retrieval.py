import os
from transformers import pipeline
from sentence_transformers import CrossEncoder

from .ingestion import extract_text_from_pdf
from .chunking import chunk_text
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .qa import QAModel
from .summarizer import Summarizer
from .contradiction import ContradictionDetector

class RAGPipeline:
    def __init__(self):
        print("Loading Embedding Model...")
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        
        print("Loading Cross-Encoder Reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        print("Loading QA Model...")
        self.qa_model = QAModel()
        
        print("Loading Summarizer Model...")
        self.summarizer = Summarizer()
        
        print("Loading Contradiction Detector...")
        self.contradiction_detector = ContradictionDetector()
        
        self.documents_processed = set()
        
    def clear_memory(self):
        """Wipes the entire vector index and processed documents list."""
        self.vector_store.reset()
        self.documents_processed = set()
        
    def process_document(self, pdf_path: str, filename: str) -> dict:
        """
        End-to-end processing of a single PDF document.
        """
        if filename in self.documents_processed:
            return {"status": "already processed"}
            
        # 1. Extraction
        pages_data = extract_text_from_pdf(pdf_path, filename)
        if not pages_data:
            return {"status": "error", "message": "No text extracted."}
            
        # 2. Chunking
        chunks = chunk_text(pages_data)
        
        # 3. Embeddings
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(chunk_texts)
        
        # 4. Store
        self.vector_store.add_vectors(embeddings, chunks)
        
        self.documents_processed.add(filename)
        
        # Optional: Generate a document-level summary
        full_text = " ".join([p["text"] for p in pages_data])
        summary = self.summarizer.summarize(full_text)
        
        return {
            "status": "success", 
            "chunks_processed": len(chunks),
            "summary": summary
        }
        
    def query(self, user_query: str, top_k: int = 5) -> dict:
        """
        Answers a user query using Hybrid Search + Cross-Encoder Reranking + RAG.
        """
        # 1. Embed query
        query_vector = self.embedding_model.encode([user_query])[0]
        
        # 2. Retrieve initial chunks using Hybrid Search (FAISS + BM25)
        # We retrieve more chunks initially (e.g. top_k * 2) so we can rerank them
        hybrid_results = self.vector_store.search(query_vector, query_text=user_query, top_k=top_k * 2)
        
        if not hybrid_results:
            return {
                "answer": "No relevant documents found. Please upload documents first.",
                "citations": [],
                "contradictions": []
            }
            
        # 3. Rerank with Cross-Encoder
        # Pair each chunk with the query
        pairs = [[user_query, res["chunk"]["text"]] for res in hybrid_results]
        scores = self.reranker.predict(pairs)
        
        # Sort by cross-encoder score
        for res, score in zip(hybrid_results, scores):
            res["rerank_score"] = float(score)
            
        hybrid_results = sorted(hybrid_results, key=lambda x: x["rerank_score"], reverse=True)
        
        # Take the top K after reranking
        final_results = hybrid_results[:top_k]
            
        # 4. Generate answer
        answer = self.qa_model.generate_answer(user_query, final_results)
        
        # 5. Format citations
        citations = []
        for res in final_results:
            chunk = res["chunk"]
            citations.append(f"[{chunk['document_name']}, Page {chunk['page_number']}]")
            
        # 6. Detect contradictions in retrieved chunks
        contradictions = self.contradiction_detector.analyze_chunks(final_results)
        
        # Prepare retrieved chunks with their scores for transparency
        transparent_chunks = []
        for res in final_results:
            transparent_chunks.append({
                "text": res["chunk"]["text"],
                "score": res["rerank_score"],
                "source": f"[{res['chunk']['document_name']}, Page {res['chunk']['page_number']}]"
            })
        
        return {
            "answer": answer,
            "citations": list(set(citations)), # Unique citations
            "retrieved_chunks": transparent_chunks,
            "contradictions": contradictions
        }
