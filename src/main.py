import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.app_state import state
from models.ollama import ChatOllama
from services.ai_service import AiService
from services.rag_service import RagService
from routers.ai_router import ai_router
from utils.logging import logger
from core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = ChatOllama()
    await llm.pull_model()
    
    rag_service = RagService()
    await rag_service.init_vector_store()

    state.llm = llm
    state.rag_service = rag_service
    state.ai_service = AiService(llm)

    yield

app = FastAPI(
    title="AI Service API",
    description="Service for AI text generation using Ollama",
    version="1.0.0",
    lifespan=lifespan,
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
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Для продакшена лучше False
        log_level="info"
    )