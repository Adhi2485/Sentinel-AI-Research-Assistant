# 🔬 Sentinel: Multi-Document AI Research Assistant
## Comprehensive Project Summary & Architecture Report

This document outlines the architecture, data pipeline, and technologies used to build a state-of-the-art Retrieval-Augmented Generation (RAG) research assistant. 

---

### 1. System Architecture Overview
The project is built on a decoupled, modern architecture:
*   **Backend:** A high-performance asynchronous REST API built with **FastAPI** (Python). It handles heavy machine learning inference, semantic indexing, and memory management.
*   **Frontend:** A standalone, responsive Single Page Application (SPA) built using **React JS** and **Vite**, featuring modern styling, glassmorphism UI elements, and real-time state management.

---

### 2. Core Processing Pipeline (How It Works)

The backend follows an advanced NLP pipeline designed to ingest complex academic PDFs and answer natural language questions accurately without hallucinations.

#### A. Data Ingestion & Chunking
*   **Extraction:** Uses `PyMuPDF` to accurately extract raw text and layout metadata (page numbers, document names) from uploaded PDFs.
*   **Smart Chunking:** Text is split into overlapping chunks. To prevent Transformer token-limit crashes (`1091 > 512` error), the chunk size is strictly optimized to **250 words** with a **40-word overlap**, maintaining contextual continuity across page breaks.

#### B. Indexing & Storage
*   **Dense Embeddings:** Chunks are passed through a Sentence-Transformer model (`all-MiniLM-L6-v2`) to convert textual meaning into 384-dimensional mathematical vectors.
*   **Vector Storage (FAISS):** These vectors are indexed into a **FAISS** (Facebook AI Similarity Search) L2-distance database for blazing-fast semantic lookups in memory.
*   **Sparse Lexical Indexing (BM25):** Alongside the dense vectors, the exact words are tokenized and indexed using the **Okapi BM25** algorithm.

#### C. Query & Retrieval (Hybrid Search)
When a user asks a question, the system does not just do a simple search. It uses **Hybrid Retrieval**:
1.  It queries both the FAISS (meaning-based) and BM25 (keyword-based) indexes to retrieve an initial broad pool of documents (Top K * 2).
2.  **Cross-Encoder Re-Ranking:** It passes these candidate chunks into a specialized Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`). This model performs deep pairwise analysis between the user's question and each chunk, assigning a highly accurate mathematical "Rank Score" and discarding irrelevant data.

#### D. Synthesis & Contradiction Detection
*   **Generative QA:** The top re-ranked chunks are passed as context to an advanced LLM (**`google/flan-t5-large`**). The prompt explicitly bounds the AI to use *only* the provided context. The generation utilizes **Beam Search**, early stopping, and n-gram repetition penalties for highly professional prose.
*   **Contradiction Analysis:** The retrieved chunks are evaluated by a Natural Language Inference (NLI) model (`nli-distilroberta-base`) to detect if different academic papers disagree with each other (e.g., Paper A says X, Paper B says Y).

---

### 3. Key Advanced Features (Final Touches)

1.  **System Telemetry Dashboard:** An "Under the Hood" UI feature that proves the system's architecture. It queries a `/system_telemetry` endpoint to display live RAM usage, FAISS index sizes, vector dimensions, and a live JSON snapshot of exactly how a chunk is stored in memory.
2.  **Transparency & Rerank Scores:** Underneath every answer, users can expand a "Transparency" drawer to see the exact Cross-Encoder scores and source text that influenced the AI's answer.
3.  **Conversational Memory (Chat History):** The React frontend maintains a persistent, scrollable history of all user interactions, citations, and answers during a session.
4.  **Database "Nuke" Control:** A dedicated endpoint and UI button that safely flushes the FAISS index, RAM, and document logs, preventing cross-contamination between different research sessions.
5.  **Automated RAG Evaluation Script:** A standalone script (`evaluate_rag.py`) utilizing **ROUGE-L** (lexical overlap) and **Cosine Similarity** to mathematically prove the accuracy of the system against ground-truth benchmarks.

---

### 4. Technology Stack & Libraries

#### AI / NLP Libraries (Python)
*   **`transformers`**: HuggingFace library for downloading and running the FLAN-T5 (QA) and BART (Summarization) LLMs.
*   **`sentence-transformers`**: Handles the creation of dense embeddings and the Cross-Encoder re-ranking.
*   **`faiss-cpu`**: Facebook's highly optimized C++ library for vector similarity search.
*   **`rank_bm25`**: Implementation of the Okapi BM25 algorithm for keyword-based sparse retrieval.
*   **`torch` (PyTorch)**: The underlying deep learning framework powering the transformer models.

#### Backend Framework & Utilities
*   **`fastapi` & `uvicorn`**: High-performance asynchronous web framework and ASGI server.
*   **`pydantic`**: Data validation and strict typing for API payloads.
*   **`python-multipart`**: Enables the backend to securely receive PDF file uploads.
*   **`PyMuPDF` (fitz)**: The fastest and most accurate PDF parsing library available for Python.

#### Evaluation Metrics
*   **`scikit-learn`**: Used for mathematical operations like calculating Cosine Similarities in the evaluation script.
*   **`rouge-score`**: Calculates the structural overlap between generated text and ground truth.

#### Frontend Web Stack
*   **`react`**: Component-based UI library.
*   **`vite`**: Next-generation lightning-fast frontend build tool.
*   **Vanilla CSS**: Custom-written CSS variables, animations, and Flexbox grids for a sleek, glassmorphic design without heavy CSS frameworks.

---

### 5. How to Run the Project

This project requires both the FastAPI backend and the React frontend to be running simultaneously.

#### Step 1: Start the Backend (AI Engine)
1. Open a terminal and navigate to the root directory.
2. Install the required Python dependencies (if you haven't already):
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   python api/main.py
   ```
   *(Wait for the AI models to load into memory. It is ready when you see `Uvicorn running on http://0.0.0.0:8000`)*

#### Step 2: Start the Frontend (User Interface)
1. Open a **second** terminal window.
2. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
3. Install the Node modules (first time only):
   ```bash
   npm install
   ```
4. Start the Vite development server:
   ```bash
   npm run dev
   ```

#### Step 3: Access the Application
Open your web browser and navigate to: **[http://localhost:5173](http://localhost:5173)**

#### Evaluation
To run the automated metrics script:
```bash
python evaluate_rag.py
```
