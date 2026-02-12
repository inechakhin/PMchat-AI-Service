import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    
    # Пути и директории
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs") 
    
    # Ollama
    OLLAMA_HOST: str = Field("localhost", env="OLLAMA_HOST")
    OLLAMA_PORT: int = Field(11434, env="OLLAMA_PORT")
    OLLAMA_MODEL_NAME: str = Field("qwen2.5:3b", env = "OLLAMA_MODEL_NAME")
    
    # CORS
    ALLOWED_ORIGINS: list[str] = Field(["*"], env="ALLOWED_ORIGINS")
    
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Config()

