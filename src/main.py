import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from core.config import settings
from db.mongo import MongoDB
from db.qdrant import QdrantDBClient
from db.s3 import S3Client
from models.factory import LLM, Embedder
from repositories.template_repository import TemplateRepository
from services.docx_worker import DocxWorker
from services.rag_service import RagService
from services.template_service import TemplateService
from routers.ai_router import ai_router

from utils.logging import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = await LLM.get_llm(settings.LLM_PROVIDER)
    app.state.llm = llm
    embedder = await Embedder.get_embedder(settings.EMBEDDER_PROVIDER)
    app.state.embedder = embedder
    
    mongo = MongoDB()
    await mongo.connect()
    logger.info("MongoDB подключена")
    app.state.mongo = mongo
    
    qdrant = QdrantDBClient()
    logger.info("Qdrant подключен")
    app.state.qdrant = qdrant
    
    s3 = S3Client()
    await s3.start()
    await s3.create_bucket_if_not_exists()
    logger.info("S3 подключен")
    app.state.s3 = s3

    docx_worker = DocxWorker()
    logger.info("DocxWorker подключен")
    
    rag_service = RagService(
        embedder,
        docx_worker,
        qdrant,
    )
    await rag_service.init_vector_store(hard_init=False)
    logger.info("Проиницилизировано векторное хранилище")
    
    template_service = TemplateService(
        TemplateRepository(),
        docx_worker,
        qdrant,
        llm,
    )
    await template_service.init_template_store(hard_init=False)
    logger.info("Проиницилизировано хранилище шаблонов")

    yield
    
    await mongo.disconnect()
    await s3.stop()
    logger.info("Всё было успешно завершено")

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