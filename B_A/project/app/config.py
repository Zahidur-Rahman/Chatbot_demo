from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    embedding_model: str = "all-MiniLM-L6-v2"    # Lightweight (Consider if you'll still use this or switch)
    mistral_api_key: str  # Mistral API key (Removed default empty string to enforce its presence in .env)
    mistral_model: str = "mistral-tiny"  # Default Mistral model name

    class Config:
        env_file = ".env"

settings = Settings()