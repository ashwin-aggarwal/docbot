import os
import tempfile
from pathlib import Path
from typing import List, Tuple

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage


SYSTEM_PROMPT = """You are a helpful assistant that answers questions ONLY based on the provided context documents.
If the answer is not in the context, say "I don't have enough information in the uploaded documents to answer that."
Always cite which source document(s) your answer comes from.
Do not use any outside knowledge."""


def load_documents(file_paths: List[str]) -> Tuple[List[Document], List[str]]:
    """Load documents from a list of file paths. Returns (docs, failed_files)."""
    docs = []
    failed = []

    for path in file_paths:
        ext = Path(path).suffix.lower()
        filename = Path(path).name
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext in (".txt", ".md"):
                loader = TextLoader(path, encoding="utf-8")
            else:
                failed.append(filename)
                continue

            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = filename
            docs.extend(loaded)
        except Exception as e:
            failed.append(f"{filename} ({e})")

    return docs, failed


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)


def build_retriever(chunks: List[Document]) -> EnsembleRetriever:
    """Build a hybrid BM25 + FAISS ensemble retriever."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    faiss_store = FAISS.from_documents(chunks, embeddings)
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 5})

    bm25_retriever = BM25Retriever.from_documents(chunks, k=5)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6],
    )


def build_chain(retriever: EnsembleRetriever):
    """Build the RAG chain with conversation history support."""
    llm = ChatOllama(model="llama3", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def format_docs(docs: List[Document]) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Source: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def get_context(inputs):
        return format_docs(retriever.invoke(inputs["question"]))

    chain = (
        {
            "context": lambda inputs: get_context(inputs),
            "history": lambda inputs: inputs.get("history", []),
            "question": lambda inputs: inputs["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def get_sources(retriever: EnsembleRetriever, query: str) -> List[Document]:
    """Retrieve source documents for a query."""
    return retriever.invoke(query)


def ingest_files(file_paths: List[str]) -> Tuple[EnsembleRetriever, object, int, List[str]]:
    """
    Full ingestion pipeline.
    Returns (retriever, chain, chunk_count, failed_files).
    """
    docs, failed = load_documents(file_paths)
    if not docs:
        raise ValueError("No documents could be loaded.")

    chunks = chunk_documents(docs)
    retriever = build_retriever(chunks)
    chain = build_chain(retriever)

    return retriever, chain, len(chunks), failed
