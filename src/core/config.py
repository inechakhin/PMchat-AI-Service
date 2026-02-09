import os

from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    
    # Пути и директории
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs") 
    
    # Docker ChromaDB
    """
    CHROMA_DB_PATH: str = os.path.join(BASE_DIR, "chroma")
    
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
    COLLECTION_NAME: str = "pm_documents"
    EMBEDDER_MODEL_NAME: str = "intfloat/multilingual-e5-large-instruct"
    """
    
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "llama3.2:3b"
    
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Config()

