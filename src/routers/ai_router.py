import asyncio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from schemas.ai import AiRequest
from services.ai_service import AiService
from dependencies.ai import get_ai_service
from utils.logging import logger
from core.exceptions.ai_generation_error import AiGenerationError

ai_router = APIRouter(prefix="/internal/api/ai", tags=["ai"])

@ai_router.post("/generate-stream")
async def generate_stream(
    request: AiRequest,
    ai_service: AiService = Depends(get_ai_service)
):
    async def stream_generator():
        try:
            async for chunk in ai_service.generate_stream(request):
                yield chunk
        except asyncio.CancelledError:
            yield f"ERROR: Generation cancelled by user"
        except AiGenerationError as e:
            yield f"ERROR: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in generate_stream: {e}")
            yield f"ERROR: Internal server error"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
    )

@ai_router.post("/generate")
async def generate(
    request: AiRequest,
    ai_service: AiService = Depends(get_ai_service)
):
    try:
        full_response = await ai_service.generate_completion(request)
        return {"response": full_response}
    except AiGenerationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in generate: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")