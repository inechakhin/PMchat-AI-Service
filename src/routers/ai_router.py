from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from schemas.ai import AiRequest
from services.ai_service import AiService
from dependencies.ai import get_ai_service

ai_router = APIRouter(prefix="/internal/api/ai", tags=["ai"])

@ai_router.post("/generate-stream")
async def generate_stream(
    request: AiRequest,
    ai_service: AiService = Depends(get_ai_service)
):    
    return StreamingResponse(
        ai_service.generate_response(request),
        media_type="text/plain",
    )