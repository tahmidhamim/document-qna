# 📄 Document Q&A System

Ask questions about any PDF — answers are grounded exclusively in the document.

---

## Architecture

```
PDF upload
   │
   ▼
pypdf          — extracts text page-by-page
   │
   ▼
RecursiveCharacterTextSplitter  — 800-char chunks, 100-char overlap
   │
   ▼
HuggingFaceEmbeddings           — all-MiniLM-L6-v2, runs locally (FREE)
   │
   ▼
FAISS vector store              — in-memory similarity index
   │
   ▼
ConversationalRetrievalChain
   ├── Condense follow-ups → standalone question   (gpt-4o-mini)
   ├── Retrieve top-4 chunks from FAISS
   └── Generate answer grounded in chunks          (gpt-4o-mini)
```

### Model choices

| Component  | Model | Cost |
|-----------|-------|------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | **Free** (local) |
| LLM | `openai/gpt-4o-mini` via OpenRouter | ~$0.15/1M input tokens |

A typical Q&A session (20 questions on a 50-page PDF) costs **≈ $0.002**.

---

## Quick Start

### 1 — Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Run the app

```bash
streamlit run app.py
```

### 3 — Use the UI

1. Open **http://localhost:8501** in your browser.
2. Paste your **OpenRouter API key** in the sidebar.
3. Upload a **PDF**.
4. Ask questions in the chat box.

Source chunks used to generate each answer are shown in a collapsible expander.

---

## Project structure

```
doc_qa/
├── app.py            # Streamlit UI
├── qa_engine.py      # Ingestion, embedding, retrieval, generation
├── requirements.txt  # Python dependencies
├── .env.example      # Optional env-var template
└── README.md
```

---

## Configuration (qa_engine.py constants)

| Constant | Default | Purpose |
|----------|---------|---------|
| `LLM_MODEL` | `openai/gpt-4o-mini` | Chat model via OpenRouter |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embeddings |
| `CHUNK_SIZE` | `800` | Characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `TOP_K_DOCS` | `4` | Chunks retrieved per query |
| `MEMORY_WINDOW` | `6` | Conversation turns kept in memory |

---

## How it works

1. **Ingestion** — The PDF is loaded with `pypdf`, split into overlapping chunks, embedded with a local sentence-transformer model, and stored in a FAISS index held in memory.

2. **Retrieval** — When a question arrives, a condensed standalone version is embedded and used to find the top-4 most similar chunks from FAISS.

3. **Generation** — A strict system prompt instructs the LLM to answer using *only* the retrieved chunks, preventing hallucinations. The chain maintains a sliding 6-message conversation window so follow-up questions work naturally.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Live App

The app is live in this [link](https://document-qna-xzuzyglhecw2tsdz8lommf.streamlit.app/).