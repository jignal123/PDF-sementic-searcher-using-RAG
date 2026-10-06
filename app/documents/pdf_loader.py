"""PDF loading with source and page metadata."""

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from pypdf import PdfReader


class PdfDocumentLoader:
    """Load PDF pages as LangChain documents."""

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load all pages from one PDF file."""
        path = Path(file_path)
        self._validate_path(path)

        reader = PdfReader(str(path))
        documents: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": str(path),
                        "file_name": path.name,
                        "page": page_number,
                    },
                )
            )
        return documents

    def load_files(self, file_paths: Iterable[str | Path]) -> list[Document]:
        """Load pages from multiple PDF files in the provided order."""
        documents: list[Document] = []
        for file_path in file_paths:
            documents.extend(self.load_file(file_path))
        return documents

    @staticmethod
    def _validate_path(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"PDF path is not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, received: {path}")
