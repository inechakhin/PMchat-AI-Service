from models.ollama import ChatOllama
from services.ai_service import AiService
from services.rag_service import RagService

class AppState:
    llm: ChatOllama | None = None
    ai_service: AiService | None = None
    rag_service: RagService | None = None

state = AppState()
