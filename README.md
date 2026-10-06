# PDF Semantic Search

This project will provide semantic search over uploaded PDF documents using
LangChain, Google Gemini models, a vector database, and Streamlit.

## Step 1: Initialize the environment

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`, set `GOOGLE_API_KEY`, and keep
   `GOOGLE_EMBEDDING_MODEL=gemini-embedding-001`.

Google Gemini models are accessed through the API; they are not downloaded
locally. The configured embedding model will be used later by the RAG module.

If you previously indexed documents with `text-embedding-004`, delete the
`vector_store/` directory before indexing again so Chroma does not mix
embeddings from different models.

The PDF loader, semantic chunking, vector database, retrieval, and Streamlit
interface will be added as separately approved steps.

## Step 2: PDF loading and chunking

The document layer loads one or more PDFs page by page and preserves each
source file and page number in LangChain document metadata. The chunker then
splits text at paragraph, line, sentence, and word boundaries while retaining
that metadata for later search-result citations.

## Step 3: Embeddings and vector search

`ChromaVectorStore` uses the configured Google embedding model to persist
chunks locally and supports similarity searches with optional distance scores.
The persistent `vector_store/` directory is excluded from version control.

## Step 4: Streamlit interface

Start the interface with:

```powershell
streamlit run app\streamlit_app.py
```

Upload PDFs, select **Index uploaded PDFs**, and enter a query to view matching
chunks with their source filename, page number, and Chroma distance.
