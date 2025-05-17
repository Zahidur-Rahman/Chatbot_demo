from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    mistral_api_key: str  # Mistral API key
    mistral_model: str = "mistral-small"  # Default to mistral-small for better performance
    mistral_embedding_model: str = "mistral-embed"  # Mistral's embedding model
    hf_token: Optional[str] = None  # Optional Hugging Face token

    class Config:
        env_file = ".env"

settings = Settings()