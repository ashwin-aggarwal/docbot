# DocBot — Local RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Ollama. Upload your PDF, TXT, or Markdown files and ask questions — answers come **only** from your documents, with source citations. No external API calls.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)             │
│  ┌──────────────┐        ┌────────────────────────┐  │
│  │   Sidebar    │        │      Chat Interface    │  │
│  │ File Upload  │        │  st.chat_message()     │  │
│  │ (PDF/TXT/MD) │        │  📄 Sources expander   │  │
│  └──────┬───────┘        └────────────┬───────────┘  │
└─────────┼──────────────────────────── ┼─────────────┘
          │                             │
          ▼                             ▼
┌─────────────────────────────────────────────────────┐
│                  rag_engine.py                       │
│                                                      │
│  1. Load  →  PyPDFLoader / TextLoader                │
│  2. Chunk →  RecursiveCharacterTextSplitter          │
│             (chunk_size=500, overlap=100)            │
│  3. Index →  ┌──────────────────────────────────┐   │
│              │       EnsembleRetriever           │   │
│              │  BM25 (40%)   +   FAISS (60%)    │   │
│              │  keyword            semantic      │   │
│              │  rank-bm25     OllamaEmbeddings   │   │
│              │               nomic-embed-text    │   │
│              └──────────────────────────────────┘   │
│  4. Generate → ChatOllama (llama3, temp=0.2)         │
│               + conversation history (last 5 turns)  │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│   Ollama (local)    │
│  localhost:11434    │
│  • llama3           │
│  • nomic-embed-text │
└─────────────────────┘
```

---

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com)** installed and running
- The following Ollama models pulled:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

---

## Installation

```bash
# 1. Enter the project directory
cd DocBot

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running

```bash
# Make sure Ollama is running in a separate terminal
ollama serve

# Then start the app
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## Usage

1. **Upload files** — Use the sidebar to upload up to 20 PDF, TXT, or MD files.
2. **Process** — Click **Process Documents**. Embeddings are generated locally via `nomic-embed-text`.
3. **Chat** — Type questions in the chat input. Answers are grounded in your documents only.
4. **View sources** — Each response includes a **📄 Sources** expander showing the retrieved chunks and their file names.

---

## Key Design Decisions

| Choice | Reason |
|---|---|
| Hybrid BM25 + FAISS | BM25 handles exact keyword matches; FAISS handles semantic similarity |
| Reciprocal Rank Fusion | EnsembleRetriever fuses rankings without needing score normalization |
| Ollama embeddings | Fully local, no API key required |
| Manual message list | Avoids deprecated `ConversationBufferMemory`; simple and explicit |
| 5-turn history window | Balances context quality vs. prompt length |

---

## File Structure

```
DocBot/
├── app.py            # Streamlit UI
├── rag_engine.py     # Ingestion, retrieval, generation logic
├── requirements.txt  # Python dependencies
└── README.md
```
