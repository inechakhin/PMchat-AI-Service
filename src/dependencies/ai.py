from core.app_state import state
from services.ai_service import AiService

def get_ai_service() -> AiService:
    if not state.ai_service:
        raise RuntimeError("AI service not initialized")
    return state.ai_service