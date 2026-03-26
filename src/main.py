import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from core.app_state import state
from models.ollama import ChatOllama
from services.ai_service import AiService
from services.rag_service import RagService
from routers.ai_router import ai_router
from utils.logging import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    ollama = ChatOllama()
    await ollama.pull_model()
    state.ollama = ollama
    
    rag_service = RagService(ollama)
    await rag_service.init_vector_store()
    state.rag_service = rag_service
    
    state.ai_service = AiService(ollama, rag_service)

    yield

app = FastAPI(
    title="AI Service API",
    description="Service for AI text generation using Ollama",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ai_router)

if __name__ == "__main__":
    logger.info("Starting AI Service on 0.0.0.0:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Для продакшена лучше False
        log_level="info"
    )