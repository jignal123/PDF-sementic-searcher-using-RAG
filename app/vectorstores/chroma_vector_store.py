"""Chroma vector store operations for document chunks."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class ChromaVectorStore:
    """Persist document embeddings and perform similarity searches."""

    def __init__(
        self,
        embedding_model: Embeddings,
        persist_directory: str | Path = "vector_store",
        collection_name: str = "pdf_documents",
    ) -> None:
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=str(persist_directory),
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed and persist document chunks, returning their IDs."""
        if not documents:
            raise ValueError("At least one document is required for indexing.")
        return self._store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        result_count: int = 4,
    ) -> list[Document]:
        """Return the most similar documents for a query."""
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if result_count <= 0:
            raise ValueError("result_count must be greater than zero.")
        return self._store.similarity_search(query, k=result_count)

    def similarity_search_with_scores(
        self,
        query: str,
        result_count: int = 4,
    ) -> list[tuple[Document, float]]:
        """Return similar documents together with Chroma distance scores."""
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if result_count <= 0:
            raise ValueError("result_count must be greater than zero.")
        return self._store.similarity_search_with_score(query, k=result_count)
