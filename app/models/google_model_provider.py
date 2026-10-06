"""Object-oriented factory for Google models used by LangChain."""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config.settings import Settings


class GoogleModelProvider:
    """Create configured Google Gemini chat and embedding models."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create_chat_model(self) -> ChatGoogleGenerativeAI:
        """Return the configured Gemini chat model."""
        return ChatGoogleGenerativeAI(
            model=self._settings.generative_ai_model,
            google_api_key=self._settings.google_api_key,
            temperature=0,
        )

    def create_embedding_model(self) -> GoogleGenerativeAIEmbeddings:
        """Return the configured Google embedding model."""
        return GoogleGenerativeAIEmbeddings(
            model=self._settings.embedding_model,
            google_api_key=self._settings.google_api_key,
        )
