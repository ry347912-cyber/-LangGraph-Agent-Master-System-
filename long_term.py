"""
memory/long_term.py — Long-Term Vector Memory
Uses Pinecone (prod) or ChromaDB (local) for persistent memory.
"""
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

_retriever = None


def get_retriever(k: int = 4):
    """Return cached retriever — lazy init."""
    global _retriever
    if _retriever:
        return _retriever

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    use_pinecone = os.getenv("USE_PINECONE", "false").lower() == "true"

    if use_pinecone:
        logger.info("Using Pinecone vector store")
        index_name = os.getenv("PINECONE_INDEX", "agent-memory")
        store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
    else:
        logger.info("Using local ChromaDB")
        store = Chroma(
            collection_name="agent_knowledge",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )

    _retriever = store.as_retriever(
        search_type="mmr",          # Max marginal relevance = diverse results
        search_kwargs={"k": k, "fetch_k": k * 3}
    )
    return _retriever


def add_to_memory(texts: list[str], metadatas: list[dict] = None):
    """Add new documents to the vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma(
        collection_name="agent_knowledge",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    store.add_texts(texts, metadatas=metadatas or [{} for _ in texts])
    logger.info(f"Added {len(texts)} documents to long-term memory")
