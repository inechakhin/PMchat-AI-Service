import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.ai_router import ai_router
from utils.logging import logger
from core.config import settings

app = FastAPI(
    title="AI Service API",
    description="Service for AI text generation using Ollama",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # TODO указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router)

if __name__ == "__main__":
    logger.info(f"Starting AI Service on 0.0.0.0:8000")
    logger.info(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
    logger.info(f"Ollama Model: {settings.OLLAMA_MODEL_NAME}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Для продакшена лучше False
        log_level="info"
    )