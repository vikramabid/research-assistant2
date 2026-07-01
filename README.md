# AI Research Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) application for answering questions over uploaded PDFs.

The app uses:
- PDF ingestion via `PyMuPDF`
- Document chunking with `langchain_text_splitters`
- Embeddings from HuggingFace via `langchain_huggingface`
- Vector similarity search with `FAISS`
- Hybrid retrieval using BM25 and FAISS
- Query rewriting and reranking
- Gemini 2.5 Flash via `langchain_google_genai`

---

## 1. Overview

This project is a research assistant that lets users upload PDFs, build a searchable vector index, and ask questions about the uploaded documents. The app retrieves relevant document chunks, creates a RAG prompt, and returns answers with source citations.

The main entrypoint is `app.py`, which runs a Streamlit interface and orchestrates all steps.

---

## 2. How it works

### 2.1 Upload and ingestion

1. A user uploads one or more PDF files through the Streamlit UI.
2. `src.utils.hashing.calculate_pdf_hash()` computes an MD5 hash of the uploaded files.
3. Each PDF is loaded page-by-page using `src.ingestion.pdf_loader.load_pdf()`.
4. Each page becomes a `langchain_core.Document` with metadata:
   - `source`: original PDF filename
   - `page`: page number

### 2.2 Chunking

- `src.ingestion.chunker.chunk_documents()` splits long page texts into smaller chunks using `RecursiveCharacterTextSplitter`.
- Default chunk parameters:
  - `chunk_size`: 1000 characters
  - `chunk_overlap`: 200 characters
- The splitter uses separators like double newlines, single newlines, sentence boundaries, and spaces to preserve document structure.

### 2.3 Embeddings and FAISS vector store

- `src.embeddings.hf_embeddings.get_embeddings()` creates a HuggingFace embeddings model using `Config.EMBEDDING_MODEL`.
- `src.vectorstore.faiss_store.create_vectorstore()` converts document chunks into embeddings and stores them in a local FAISS index directory at `data/faiss_index`.
- `src.vectorstore.faiss_store.save_hash()` writes the hash into `data/faiss_index/hash.txt`.
- A helper `src.vectorstore.faiss_store.load_vectorstore()` exists to load a saved FAISS index, though the main app currently rebuilds on upload.

### 2.4 Retrieval

- `src.retriever.retriever.get_retriever()` wraps the FAISS vectorstore as a similarity retriever.
- `src.retriever.hybrid_retriever.HybridRetriever` builds a hybrid retrieval layer:
  - BM25 ranking using `rank_bm25.BM25Okapi`
  - FAISS retrieval from the vector store
  - Combines both rankings while removing duplicates

### 2.5 Query rewriting and reranking

- `src.query_rewriter.rewriter.QueryRewriter` rewrites the incoming question into a standalone query using the chat history.
- `src.reranker.reranker.Reranker` reranks retrieved documents with `sentence_transformers.CrossEncoder`.

### 2.6 RAG chain and LLM response

- `src.prompts.rag_prompt.RAG_PROMPT` defines the prompt template used to instruct the model.
- `src.llm.gemini.get_llm()` initializes Gemini via `ChatGoogleGenerativeAI` with the API key from the environment.
- `src.chains.rag_chain.RAGChain` is designed to:
  1. Rewrite the question
  2. Retrieve candidate documents
  3. Rerank the candidates
  4. Format the best documents into context
  5. Build a final prompt with history, context, and question
  6. Invoke the Gemini model
  7. Append source citations to the answer

> Note: `src.chains.rag_chain.RAGChain.invoke()` currently has an issue where `rewritten_question` is referenced before it is defined. The intended flow is to rewrite the query before document retrieval.

### 2.7 Chat session state

- `src.chat.memory` manages Streamlit session state:
  - `messages` list stores user and assistant chat entries
  - `add_user_message()` and `add_ai_message()` append messages
  - `display_chat()` renders the chat history in the Streamlit UI
  - `get_chat_history()` returns a concatenated history string for query rewriting

---

## 3. Code structure

### Root files
- `app.py`: Streamlit application entrypoint and orchestration.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependency list.
- `data/`: Storage for generated index and PDF data artifacts.
- `logs/`: Logging output.

### Primary source packages
- `src/config.py`: Project configuration and required environment validation.
- `src/ingestion/`: PDF loading and chunking.
- `src/embeddings/`: Embedding generation.
- `src/vectorstore/`: FAISS index creation and loading.
- `src/retriever/`: Retrieval abstractions and hybrid search.
- `src/chains/`: RAG orchestration.
- `src/llm/`: Language model interface.
- `src/prompts/`: Prompt templates.
- `src/query_rewriter/`: Context-aware rewriting.
- `src/reranker/`: Document reranking.
- `src/chat/`: UI chat session state.
- `src/utils/`: Utility helpers.
- `src/logger/`: Logging configuration.

---

## 4. Requirements and environment

### Required environment variables
- `GOOGLE_API_KEY`: Gemini API key used by `src.llm.gemini.get_llm()`.

### Install dependencies

```bash
pip install -r requirements.txt
```

> The repository is designed for Python and Streamlit. Some dependencies include `streamlit`, `langchain`, `langchain-google-genai`, `langchain-huggingface`, `faiss-cpu`, `sentence-transformers`, `rank-bm25`, `PyMuPDF`, and `google-auth`.

### Run the app

```bash
streamlit run app.py
```

---

## 5. Usage flow

1. Open the Streamlit app in your browser.
2. Upload one or more PDF files.
3. The app loads PDF pages and chunks them.
4. It computes embeddings and builds a FAISS index.
5. Ask questions in the chat input.
6. The app retrieves relevant chunks, calls the Gemini LLM, and returns an answer.
7. Use sidebar buttons to clear the FAISS index or chat history.

---

## 6. Notes and improvements

- `data/faiss_index/hash.txt` stores the hash of uploaded files, but the current upload flow does not compare it to skip index rebuilding.
- `src/chains/rag_chain.py` contains a bug in the query rewrite / retrieval order.
- `src/vectorstore/faiss_store.load_vectorstore()` is available but not used in `app.py`; it can be used to load an existing index without rebuilding.
- The prompt currently instructs the LLM to use only provided context, which helps reduce hallucinations.

---

## 7. Summary

This project implements a PDF-based research assistant using RAG architecture. It combines document ingestion, chunking, embeddings, vector search, query rewriting, reranking, and Gemini model inference inside a Streamlit user interface.

If you want, I can also add a diagram or improve the app to reuse the saved FAISS index and fix the RAG chain logic.