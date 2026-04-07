# Rag-ai-agent 🤖📄

A production-ready Retrieval-Augmented Generation (RAG) AI agent designed to let you easily upload PDFs and ask questions about their content.

## 🌟 Project Overview

This project implements a complete **RAG Architecture** (Retrieval-Augmented Generation) using state-of-the-art tools. It acts as an intelligent assistant that securely processes your own PDF files, breaks them down into searchable knowledge, and uses advanced Large Language Models (LLMs) to accurately answer questions based entirely on your documents. 

### 🔑 Key Features & Technologies
* **FastAPI:** Provides a robust, lightning-fast backend API to manage incoming requests.
* **Streamlit:** Powers the beautiful and interactive frontend User Interface where you can upload documents and chat.
* **Inngest:** Handles resilient background job orchestration, managing complex workflows step-by-step (e.g., parsing, embedding, querying) without task failure or timeouts. 
* **Qdrant (Vector Database):** Efficiently stores and retrieves vector embeddings (semantic representations of your text) for blazing-fast similarity search.
* **LlamaIndex:** Capably reads PDF content and intelligently splits it into smaller, manageable "chunks" of text that preserve context.
* **OpenAI (Embeddings & LLM):** Uses `text-embedding-3-large` to convert text into mathematical vectors, and `gpt-4o-mini` to formulate accurate, human-like responses based on the retrieved document context.

---

## 🛠️ How It Works (Step-by-Step)

### 1. The Ingestion Phase (Uploading a PDF)
1. **Upload:** You upload a PDF via the Streamlit frontend. It saves the file into the `/uploads` directory.
2. **Trigger:** Streamlit orchestrates an Event (`rag/ingest_pdf`) to the Inngest background engine.
3. **Parse & Chunk:** Inngest reads the PDF using LlamaIndex and splits the dense document into smaller, digestible text passages (chunks).
4. **Embed & Store:** The chunks are sent to OpenAI to generate high-dimensional vectors, which are then saved persistently inside the **Qdrant Vector Database** along with the original text.

### 2. The Query Phase (Asking a Question)
1. **Question:** You ask a question via the Streamlit chat interface.
2. **Trigger:** An Event (`rag/query_pdf_ai`) is fired off to Inngest.
3. **Semantic Search:** Inngest transforms your question into a vector and queries Qdrant for the Top `K` most relevant chunks from your PDF.
4. **AI Generation:** Inngest feeds those top chunks (the "context") and your original question into the OpenAI LLM. The AI agent generates a direct answer strictly based on the text found in the PDF.
5. **Display:** Streamlit polls the background job, fetches the final answer, and delightfully presents both the answer and the sources to you!

---

## 🚀 Setup & Execution Instructions

Follow these exact steps to run the complete environment:

### 1. Environment variables
Create a `.env` file in the root directory and add your OpenAI API Key:
```env
OPENAI_API_KEY="sk-your-openai-api-key"
```
*(You can obtain this from [platform.openai.com/API-KEYS](https://platform.openai.com/API-KEYS))*

##### as openai is not free so i used groq for the llm and sentence transformer for the embeddings

### 2. Install Dependencies
```bash
uv pip install -r requirements.txt
```

### 3. Run Qdrant Vector DB (Docker)
Start a local Qdrant server instance to store the AI memory:
```bash
docker run -d --name qdrantRagDb -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant  
```

### 4. Run the Backend API (FastAPI)
```bash
uv run uvicorn main:app
```

### 5. Run the Inngest Dev Server
Start the Inngest engine to power background task orchestration:
```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

### 6. Run the Frontend UI (Streamlit)
Launch the interactive web interface:
```bash
uv run streamlit run streamlit_app.py
```

---

## 🏗️ Project Architecture & Code Structure

To understand how everything fits together under the hood, here is a detailed breakdown of the codebase and its components:

### 1. The Core Application (`main.py`)
This is the heart of the backend logic. It contains the **FastAPI application** and defines the **Inngest Workflow Functions**:
* **`rag_ingest_pdf`:** The background function that handles PDF ingestion. It breaks the job down into two reliable steps: `_load` (to chunk the text) and `_upsert` (to turn text into vectors and save to database).
* **`rag_query_pdf_ai`:** The background function that handles user questions. It consists of `_search` (finding relevant context in Qdrant) and `inngest.experimental.ai.infer` (using OpenAI to generate the final answer).

### 2. The Document Processor (`Data_loader.py`)
This file handles extracting and transforming raw data into AI-readable formats.
* **`PDFReader`:** Leverages LlamaIndex to extract readable text strictly from PDF files.
* **`SentenceSplitter`:** Slices massive blocks of text into overlapping `1000`-character chunks. Overlapping prevents sentences from being cut off mid-thought.
* **`embed_texts`:** Uses the OpenAI API to convert physical text chunks into an array of floating-point numbers (Vectors). 

### 3. The Vector Database Client (`vector_db.py`)
This is the bridge to our Qdrant docker container.
* **`QdrantStorage`:** A custom python class that initializes a connection to Qdrant on port `6333`.
* **`upsert`:** Takes our vector embeddings and pushes them securely into the `docs` collection in the local database.
* **`search`:** Takes a user's question (converted to a vector) and uses *Cosine Similarity* to fetch the top `K` chunks of text that mathematically match the meaning of the question the closest.

### 4. Custom Data Modeling (`custom_type.py`)
This file leverages Pydantic to ensure strict structure for our data passing through the pipeline. It explicitly defines structures like `RAGChunkAndSrc` or `RAGSearchResult` to prevent variable mix-ups.

### 5. The User Interface (`streamlit_app.py`)
The frontend application facing the user.
* Evaluates file uploads and saves them into a temporary `/uploads` directory.
* Uses polling to continuously check the Inngest Dev Server API waiting for the AI task to finish generating its response.
* Renders the final textual answer alongside the relevant context chunks.



