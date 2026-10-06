"""Environment-backed application settings."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Configuration required by the Google model provider."""

    google_api_key: str
    generative_ai_model: str
    embedding_model: str

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load and validate settings from the environment."""
        load_dotenv()

        google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not configured. Copy .env.example to .env "
                "and add your Google API key."
            )

        return cls(
            google_api_key=google_api_key,
            generative_ai_model=os.getenv(
                "GOOGLE_GENERATIVE_AI_MODEL", "gemini-2.0-flash"
            ),
            embedding_model=os.getenv(
                "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"
            ),
        )
