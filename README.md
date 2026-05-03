# Multi-Document AI Research Assistant

This is a complete, production-ready RAG system designed to process multiple research papers (PDFs), extract and index their contents semantically, and provide source-backed answers to user queries while detecting potential contradictions across different documents.

## Architecture

1. **Document Ingestion**: Extracts text using `PyMuPDF`.
2. **Chunking**: Chunks text with a sliding window approach for overlap.
3. **Embeddings**: Uses Sentence-BERT (`all-MiniLM-L6-v2`) via `sentence-transformers`.
4. **Vector Store**: Uses `FAISS` for fast similarity search.
5. **RAG Model**: Uses `google/flan-t5-base` for accurate answer generation based on retrieved context.
6. **Summarization**: Uses `facebook/bart-large-cnn` for document-level summarization.
7. **Contradiction Detection**: Uses `cross-encoder/nli-distilroberta-base` to analyze retrieved chunks for conflicts.
8. **API**: Backend built with `FastAPI`.
9. **UI**: Frontend built with `Streamlit`.

## Setup Instructions

### 1. Create a Virtual Environment (Optional but recommended)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Backend API
Start the FastAPI server:
```powershell
python api/main.py
```
Wait for the initial model loading to complete (it will download models on first run).
The API will be available at `http://localhost:8000`.

### 4. Run the Streamlit UI
In a new terminal:
```powershell
streamlit run ui/app.py
```

## Features
- Upload Multiple PDFs
- Document Summarization
- Semantic Search Querying
- Source Citations in Answers
- Automatic Contradiction Detection among retrieved facts
