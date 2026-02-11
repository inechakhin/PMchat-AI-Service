import os

from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    
    # Пути и директории
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs") 
    
    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
    
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Config()

