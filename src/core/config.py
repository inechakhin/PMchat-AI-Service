from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    
    # Пути и директории
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    LOGS_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data"
    DATA_MD_DIR: Path = BASE_DIR / "data_md"
    
    # Providers
    LLM_PROVIDER: str = "yandex"
    EMBEDDER_PROVIDER: str = "ollama"
    
    # Ollama
    OLLAMA_HOST: str = Field("localhost", env="OLLAMA_HOST")
    OLLAMA_PORT: int = Field(11434, env="OLLAMA_PORT")
    OLLAMA_MODEL_NAME: str = Field("qwen2.5:3b", env = "OLLAMA_MODEL_NAME")
    OLLAMA_EMBEDDER_NAME: str = Field("qwen3-embedding:0.6b", env = "OLLAMA_EMBEDDER_NAME")
    
    # Yandex
    YANDEX_FOLDER_ID: str = Field("", env="YANDEX_FOLDER_ID")
    YANDEX_API_KEY: str = Field("", env="YANDEX_API_KEY")
    YANDEX_MODEL: str = Field("qwen3.5-35b-a3b-fp8/latest", env="YANDEX_MODEL")
    YANDEX_EMBEDDER_DOC: str = Field("text-search-doc/latest", env="YANDEX_EMBEDDER_DOC")
    YANDEX_EMBEDDER_QUERY: str = Field("text-search-query-1.0/latest", env="YANDEX_EMBEDDER_QUERY") 
    
    # Qdrant
    QDRANT_HOST: str = Field("localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(6333, env="QDRANT_PORT")
    QDRANT_API_KEY: str = Field("", env = "QDRANT_API_KEY")
    
    QDRANT_ON_DISK: bool = True
    QDRANT_HNSW_M: int = 16
    
    # Text splitter
    MAX_CHUNK_SIZE: int = 2048
    CHUNK_OVERLAP: int = 256
    
    # RAG Service
    COLLECTION_NAME: str = Field("project_docs", env = "COLLECTION_NAME")

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Config()

