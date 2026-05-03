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
    MONGO_HOST: str = Field("localhost", validation_alias="MONGO_HOST")
    MONGO_PORT: int = Field(27018, validation_alias="MONGO_PORT")
    MONGO_USERNAME: str | None = Field(None, validation_alias="MONGO_USERNAME")
    MONGO_PASSWORD: str | None = Field(None, validation_alias="MONGO_PASSWORD")
    MONGO_DB: str = Field("pmchat_ai_mongo", validation_alias="MONGO_DATABASE")
    
    # S3
    S3_HOST: str = Field("localhost", validation_alias="S3_HOST")
    S3_PORT: int = Field("9000", validation_alias="S3_PORT")
    S3_ACCESS_KEY: str | None = Field(None, validation_alias="S3_ACCESS_KEY")
    S3_SECRET_KEY: str | None = Field(None, validation_alias="S3_SECRET_KEY")
    S3_BUCKET_NAME: str = Field("pmchat-documents", validation_alias="S3_BUCKET_NAME")
    S3_SECURE: bool = Field(False, validation_alias="S3_SECURE")
    
    # Providers
    LLM_PROVIDER: str = "ollama"
    EMBEDDER_PROVIDER: str = "ollama"
    
    # Ollama
    OLLAMA_HOST: str = Field("localhost", validation_alias="OLLAMA_HOST")
    OLLAMA_PORT: int = Field(11434, validation_alias="OLLAMA_PORT")
    OLLAMA_MODEL_NAME: str = Field("qwen2.5:3b", validation_alias = "OLLAMA_MODEL_NAME")
    OLLAMA_EMBEDDER_NAME: str = Field("qwen3-embedding:0.6b", validation_alias = "OLLAMA_EMBEDDER_NAME")
    
    # Yandex
    YANDEX_FOLDER_ID: str | None = Field(None, validation_alias="YANDEX_FOLDER_ID")
    YANDEX_API_KEY: str | None = Field(None, validation_alias="YANDEX_API_KEY")
    YANDEX_MODEL: str = Field("qwen3.5-35b-a3b-fp8/latest", validation_alias="YANDEX_MODEL")
    YANDEX_EMBEDDER_DOC: str = Field("text-search-doc/latest", validation_alias="YANDEX_EMBEDDER_DOC")
    YANDEX_EMBEDDER_QUERY: str = Field("text-search-query-1.0/latest", validation_alias="YANDEX_EMBEDDER_QUERY") 
    
    # Qdrant
    QDRANT_HOST: str = Field("localhost", validation_alias="QDRANT_HOST")
    QDRANT_PORT: int = Field(6333, validation_alias="QDRANT_PORT")
    QDRANT_API_KEY: str | None = Field(None, validation_alias = "QDRANT_API_KEY")
    
    QDRANT_ON_DISK: bool = True
    QDRANT_HNSW_M: int = 16
    
    # Text splitter
    MAX_CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 128
    
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

