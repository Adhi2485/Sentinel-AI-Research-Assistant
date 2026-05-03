import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import RAGPipeline

from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Global pipeline instance
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RAGPipeline()
    os.makedirs("data/raw_pdfs", exist_ok=True)
    os.makedirs("data/processed_text", exist_ok=True)
    yield

app = FastAPI(title="Multi-Document AI Research Assistant", lifespan=lifespan)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    retrieved_chunks: List[dict]
    contradictions: List[dict]

@app.get("/system_telemetry")
async def system_telemetry():
    """Returns the internal state of the RAG system to prove how data is stored."""
    if not pipeline:
        return {"status": "offline"}
        
    total_chunks = len(pipeline.vector_store.metadata)
    doc_names = list(pipeline.documents_processed)
    vector_dim = pipeline.vector_store.embedding_dim
    
    # Grab a sample chunk to show what's in memory
    sample_chunk = pipeline.vector_store.metadata[0] if total_chunks > 0 else None
    
    return {
        "status": "online",
        "documents_indexed": len(doc_names),
        "document_names": doc_names,
        "total_chunks_in_memory": total_chunks,
        "vector_dimension": vector_dim,
        "faiss_index_size": pipeline.vector_store.index.ntotal,
        "memory_sample": sample_chunk
    }

@app.delete("/clear_memory")
async def clear_memory():
    """Wipes the database and memory."""
    if pipeline:
        pipeline.clear_memory()
    return {"status": "success", "message": "Memory and databases wiped clean."}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
    file_path = os.path.join("data/raw_pdfs", file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Process the document
    result = pipeline.process_document(file_path, file.filename)
    
    return {"filename": file.filename, "result": result}

@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    result = pipeline.query(request.query)
    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        retrieved_chunks=result["retrieved_chunks"],
        contradictions=result["contradictions"]
    )

@app.get("/summary")
async def get_summary(filename: str):
    """
    Endpoint to retrieve the summary for a previously processed document.
    Since we don't store summaries persistently in this minimal example,
    we'll just indicate if it was processed.
    In a full implementation, summaries would be saved to a database.
    """
    if filename in pipeline.documents_processed:
        return {"status": "success", "message": f"{filename} is processed. Summary available during upload phase."}
    else:
        return {"status": "not found", "message": f"{filename} not processed yet."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
