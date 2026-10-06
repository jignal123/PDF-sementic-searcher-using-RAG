"""Streamlit user interface for PDF semantic search."""

from hashlib import sha256
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import streamlit as st

# Make the project root importable when this file is launched directly by an IDE.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import Settings
from app.documents.document_chunker import DocumentChunker
from app.documents.pdf_loader import PdfDocumentLoader
from app.models.google_model_provider import GoogleModelProvider
from app.vectorstores.chroma_vector_store import ChromaVectorStore


class StreamlitPdfSearchApp:
    """Coordinate PDF uploads, indexing, and similarity search."""

    def __init__(self) -> None:
        settings = Settings.from_environment()
        model_provider = GoogleModelProvider(settings)
        self._loader = PdfDocumentLoader()
        self._chunker = DocumentChunker()
        self._vector_store = ChromaVectorStore(
            embedding_model=model_provider.create_embedding_model()
        )

    def run(self) -> None:
        """Render the application and handle user actions."""
        st.set_page_config(page_title="PDF Semantic Search", page_icon=":mag:")
        st.title("PDF Semantic Search")
        st.write("Upload PDFs, index their contents, and search across them.")

        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Index uploaded PDFs", type="primary"):
            self._index_uploaded_files(uploaded_files)

        query = st.text_input("Search query")
        result_count = st.slider("Number of results", min_value=1, max_value=10, value=4)
        if st.button("Search") and query.strip():
            self._display_results(query, result_count)

    def _index_uploaded_files(self, uploaded_files: list) -> None:
        indexed_hashes: set[str] = st.session_state.setdefault(
            "indexed_file_hashes", set()
        )
        documents = []
        indexed_names: list[str] = []

        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            file_hash = sha256(file_bytes).hexdigest()
            if file_hash in indexed_hashes:
                continue

            with NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
                temporary_file.write(file_bytes)
                temporary_path = Path(temporary_file.name)

            try:
                loaded_documents = self._loader.load_file(temporary_path)
            finally:
                temporary_path.unlink(missing_ok=True)

            for document in loaded_documents:
                document.metadata["file_hash"] = file_hash
            documents.extend(loaded_documents)
            indexed_hashes.add(file_hash)
            indexed_names.append(uploaded_file.name)

        if not documents:
            st.info("All selected PDFs have already been indexed in this session.")
            return

        chunks = self._chunker.split(documents)
        self._vector_store.add_documents(chunks)
        st.success(
            f"Indexed {len(chunks)} chunks from {len(indexed_names)} PDF file(s)."
        )

    def _display_results(self, query: str, result_count: int) -> None:
        results = self._vector_store.similarity_search_with_scores(
            query, result_count=result_count
        )
        if not results:
            st.info("No matching documents were found.")
            return

        st.subheader("Search results")
        for index, (document, score) in enumerate(results, start=1):
            source = document.metadata.get("file_name", "Unknown file")
            page = document.metadata.get("page", "Unknown page")
            with st.expander(f"{index}. {source} - page {page}"):
                st.caption(f"Chroma distance: {score:.4f}")
                st.write(document.page_content)


def main() -> None:
    """Start the Streamlit application."""
    try:
        application = StreamlitPdfSearchApp()
    except ValueError as error:
        st.error(str(error))
        st.stop()
    application.run()


if __name__ == "__main__":
    main()
