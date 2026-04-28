from fastapi import Request, Depends

from dependencies.db import get_qdrant, get_s3
from dependencies.models import get_llm, get_embedder
from db.qdrant import QdrantDBClient
from db.s3 import S3Client
from models.base import ChatBase
from repositories.skeleton_repository import SkeletonRepository
from repositories.template_repository import TemplateRepository

from services.document_exporter import DocumentExporter
from services.docling_worker import DoclingWorker
from services.template_service import TemplateService
from services.rag_service import RagService

from services.ai_service import AiService

def get_skeleton_repository() -> SkeletonRepository:
    return SkeletonRepository()

def get_template_repository() -> TemplateRepository:
    return TemplateRepository()

def get_document_exporter() -> DocumentExporter:
    return DocumentExporter()

def get_docling(request: Request) -> DoclingWorker:
    return request.app.state.docling

def get_template_service(
    template_repository: TemplateRepository = Depends(get_template_repository),
    docling: DoclingWorker = Depends(get_docling),
    vector_store: QdrantDBClient = Depends(get_qdrant),
    llm: ChatBase = Depends(get_llm),
) -> TemplateService:
    return TemplateService(
        template_repository,
        docling,
        vector_store,
        llm,
    )
    
def get_rag_service(
    embedder: ChatBase = Depends(get_embedder),
    docling: DoclingWorker = Depends(get_docling),
    vector_store: QdrantDBClient = Depends(get_qdrant),
) -> RagService:
    return RagService(
        embedder,
        docling,
        vector_store,
    )

def get_ai_service(
    skeleton_repository: SkeletonRepository = Depends(get_skeleton_repository),
    template_service: TemplateService = Depends(get_template_service),
    llm: ChatBase = Depends(get_llm),
    rag: RagService = Depends(get_rag_service),
    doc_exporter: DocumentExporter = Depends(get_document_exporter),
    s3_client: S3Client = Depends(get_s3),
) -> AiService:
    return AiService(
        skeleton_repository,
        template_service,
        llm,
        rag,
        doc_exporter,
        s3_client,
    )