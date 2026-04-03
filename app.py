import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

import rag_engine

MAX_FILES = 20
HISTORY_WINDOW = 5  # last N exchanges passed to the LLM


def init_session_state():
    defaults = {
        "messages": [],        # [{"role": "user"/"assistant", "content": "..."}]
        "history": [],         # [HumanMessage / AIMessage] for LLM context
        "retriever": None,
        "chain": None,
        "processed_files": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def process_uploads(uploaded_files):
    """Save uploads to a temp dir, run ingestion, store results in session state."""
    tmp_paths = []
    tmp_dir = tempfile.mkdtemp()

    for uf in uploaded_files:
        dest = os.path.join(tmp_dir, uf.name)
        with open(dest, "wb") as f:
            f.write(uf.getbuffer())
        tmp_paths.append(dest)

    with st.spinner("Processing documents... this may take a minute."):
        try:
            retriever, chain, chunk_count, failed = rag_engine.ingest_files(tmp_paths)
        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                st.error("Cannot connect to Ollama. Please start it first: `ollama serve`")
            else:
                st.error(f"Ingestion failed: {e}")
            return
        finally:
            # Clean up temp files
            for p in tmp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    st.session_state.retriever = retriever
    st.session_state.chain = chain
    st.session_state.processed_files = [uf.name for uf in uploaded_files]
    # Reset chat when new docs are loaded
    st.session_state.messages = []
    st.session_state.history = []

    success_msg = (
        f"Indexed **{len(uploaded_files) - len(failed)}** file(s) → "
        f"**{chunk_count}** chunks."
    )
    if failed:
        st.warning(success_msg + f"\nFailed to load: {', '.join(failed)}")
    else:
        st.success(success_msg)


def trim_history(history, window=HISTORY_WINDOW):
    """Keep only the last `window` human/ai pairs."""
    # Each pair = 2 messages
    return history[-(window * 2):]


# ─── Page Setup ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="DocBot — RAG Chatbot", layout="wide")
init_session_state()

st.title("DocBot")
st.caption("Ask questions about your uploaded documents. All processing runs locally via Ollama.")

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload Documents")
    st.caption(f"Supports PDF, TXT, MD — up to {MAX_FILES} files.")

    uploaded = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and len(uploaded) > MAX_FILES:
        st.warning(f"Maximum {MAX_FILES} files allowed. Only the first {MAX_FILES} will be used.")
        uploaded = uploaded[:MAX_FILES]

    if st.button("Process Documents", disabled=not uploaded, type="primary"):
        process_uploads(uploaded)

    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("**Indexed files**")
        for fname in st.session_state.processed_files:
            st.markdown(f"✅ {fname}")

    st.markdown("---")
    st.markdown(
        "**Requirements**\n"
        "- [Ollama](https://ollama.com) running locally\n"
        "- `ollama pull llama3`\n"
        "- `ollama pull nomic-embed-text`"
    )

# ─── Chat Area ───────────────────────────────────────────────────────────────

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    source_name = src.metadata.get("source", "unknown")
                    snippet = src.page_content[:400].replace("\n", " ")
                    st.markdown(f"**{source_name}**\n\n> {snippet}…")

# Chat input
query = st.chat_input("Ask a question about your documents…")

if query:
    if st.session_state.chain is None:
        st.warning("Please upload and process documents before chatting.")
        st.stop()

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieve sources separately so we can display them
    sources = []
    try:
        sources = rag_engine.get_sources(st.session_state.retriever, query)
    except Exception:
        pass

    # Run chain
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        try:
            trimmed_history = trim_history(st.session_state.history)
            answer = st.session_state.chain.invoke(
                {"question": query, "history": trimmed_history}
            )
        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                answer = "Error: Cannot connect to Ollama. Please run `ollama serve`."
            else:
                answer = f"Error generating response: {e}"

        response_placeholder.markdown(answer)

        if sources:
            with st.expander("📄 Sources"):
                for src in sources:
                    source_name = src.metadata.get("source", "unknown")
                    snippet = src.page_content[:400].replace("\n", " ")
                    st.markdown(f"**{source_name}**\n\n> {snippet}…")

    # Update session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.history.append(HumanMessage(content=query))
    st.session_state.history.append(AIMessage(content=answer))
