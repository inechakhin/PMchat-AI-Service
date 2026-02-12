from models.ollama import ChatOllama
from services.ai_service import AiService

class AppState:
    llm: ChatOllama | None = None
    ai_service: AiService | None = None

state = AppState()
