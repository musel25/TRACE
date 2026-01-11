"""RAG utilities."""

from tracerag.retrieval import (
    Chunk,
    QdrantRetrievalConfig,
    build_openai_embedding_fn,
    build_qdrant_retriever,
    retrieve_qdrant_chunks,
)
from .naive import (
    RagResponse,
    build_context_text,
    build_openai_chat_fn,
    build_rag_messages,
    naive_rag,
)

__all__ = [
    "Chunk",
    "QdrantRetrievalConfig",
    "RagResponse",
    "build_qdrant_retriever",
    "build_context_text",
    "build_openai_chat_fn",
    "build_openai_embedding_fn",
    "build_rag_messages",
    "naive_rag",
    "retrieve_qdrant_chunks",
]
