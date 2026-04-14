from models.base import ChatBase
from models.type import ProviderChatType
from core.config import settings

class LLM:
    @staticmethod
    async def get_llm(provider_type: str) -> ChatBase:
        match provider_type:
            case ProviderChatType.OLLAMA:
                from models.providers.ollama import ChatOllama
            
                ollama = ChatOllama()
                await ollama.pull_model(settings.OLLAMA_MODEL_NAME)
                return ollama
            case ProviderChatType.YANDEX:
                from models.providers.yandex import ChatYandex
                
                return ChatYandex()

class Embedder:
    @staticmethod
    async def get_embedder(provider_type: str) -> ChatBase:
        match provider_type:
            case ProviderChatType.OLLAMA:
                from models.providers.ollama import ChatOllama
            
                ollama = ChatOllama()
                await ollama.pull_model(settings.OLLAMA_EMBEDDER_NAME)
                return ollama
            case ProviderChatType.YANDEX:
                from models.providers.yandex import ChatYandex
                
                return ChatYandex()