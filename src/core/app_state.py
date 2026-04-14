from models.base import ChatBase
from services.ai_service import AiService
from services.rag_service import RagService

class AppState:
    llm: ChatBase | None = None
    embedder: ChatBase | None = None
    ai_service: AiService | None = None
    rag_service: RagService | None = None

state = AppState()