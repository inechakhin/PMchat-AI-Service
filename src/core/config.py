from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    
    # Пути и директории
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    LOGS_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data"
    DATA_MD_DIR: Path = BASE_DIR / "data_md"
    
    # Ollama
    OLLAMA_HOST: str = Field("localhost", env="OLLAMA_HOST")
    OLLAMA_PORT: int = Field(11434, env="OLLAMA_PORT")
    OLLAMA_MODEL_NAME: str = Field("qwen2.5:3b", env = "OLLAMA_MODEL_NAME")
    
    # Qdrant
    QDRANT_HOST: str = Field("localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(6333, env="QDRANT_PORT")
    QDRANT_API_KEY: str = Field(env = "QDRANT_API_KEY")
    
    QDRANT_ON_DISK: bool = True
    QDRANT_HNSW_M: int = 16
    
    # Text splitter
    MAX_CHUNK_SIZE: int = 2048
    CHUNK_OVERLAP: int = 256
    
    # RAG Service
    COLLECTION_NAME: str = Field("project_docs", env = "COLLECTION_NAME")
    EMBEDDER_MODEL_NAME: str = Field("intfloat/multilingual-e5-large-instruct", env = "EMBEDDER_MODEL_NAME")
    
    # CORS
    ALLOWED_ORIGINS: list[str] = Field(["*"], env="ALLOWED_ORIGINS")
    
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Config()

