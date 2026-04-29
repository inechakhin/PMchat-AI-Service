from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    
    # Пути и директории
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    LOGS_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data"
    MD_DIR: Path = DATA_DIR / "md"
    TEMPLATES_DIR: Path = DATA_DIR / "templates"
    DOCS_DIR: Path = DATA_DIR / "docs"
    
    # MongoDB
    MONGO_HOST: str = Field("localhost", env="MONGO_HOST")
    MONGO_PORT: int = Field(27018, env="MONGO_PORT")
    MONGO_USERNAME: str | None = Field(None, env="MONGO_USERNAME")
    MONGO_PASSWORD: str | None = Field(None, env="MONGO_PASSWORD")
    MONGO_DB: str = Field("pmchat_ai_mongo", env="MONGO_DATABASE")
    
    # S3
    S3_HOST: str = Field("localhost", env="S3_HOST")
    S3_PORT: int = Field("9000", env="S3_PORT")
    S3_ACCESS_KEY: str | None = Field(None, env="S3_ACCESS_KEY")
    S3_SECRET_KEY: str | None = Field(None, env="S3_SECRET_KEY")
    S3_BUCKET_NAME: str = Field("pmchat-documents", env="S3_BUCKET_NAME")
    S3_SECURE: bool = Field(False, env="S3_SECURE")
    
    # Providers
    LLM_PROVIDER: str = "ollama"
    EMBEDDER_PROVIDER: str = "ollama"
    
    # Ollama
    OLLAMA_HOST: str = Field("localhost", env="OLLAMA_HOST")
    OLLAMA_PORT: int = Field(11434, env="OLLAMA_PORT")
    OLLAMA_MODEL_NAME: str = Field("qwen2.5:3b", env = "OLLAMA_MODEL_NAME")
    OLLAMA_EMBEDDER_NAME: str = Field("qwen3-embedding:0.6b", env = "OLLAMA_EMBEDDER_NAME")
    
    # Yandex
    YANDEX_FOLDER_ID: str | None = Field(None, env="YANDEX_FOLDER_ID")
    YANDEX_API_KEY: str | None = Field(None, env="YANDEX_API_KEY")
    YANDEX_MODEL: str = Field("qwen3.5-35b-a3b-fp8/latest", env="YANDEX_MODEL")
    YANDEX_EMBEDDER_DOC: str = Field("text-search-doc/latest", env="YANDEX_EMBEDDER_DOC")
    YANDEX_EMBEDDER_QUERY: str = Field("text-search-query-1.0/latest", env="YANDEX_EMBEDDER_QUERY") 
    
    # Qdrant
    QDRANT_HOST: str = Field("localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(6333, env="QDRANT_PORT")
    QDRANT_API_KEY: str | None = Field(None, env = "QDRANT_API_KEY")
    
    QDRANT_ON_DISK: bool = True
    QDRANT_HNSW_M: int = 16
    
    # Text splitter
    MAX_CHUNK_SIZE: int = 2048
    CHUNK_OVERLAP: int = 256
    
    # RAG Service
    HEADERS_COLLECTION: str = "doc_headers"
    CHUNKS_COLLECTION: str = "doc_chunks"
    
    TYPE_BOOST: float = 0.15
    HEADING_BOOST: float = 0.20
    
    # Template Service
    DISTANCE_THRESHOLD: float = 0.3

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Config()

