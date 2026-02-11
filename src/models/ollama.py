from typing import AsyncGenerator, Dict, Any, Optional
from ollama import AsyncClient, ResponseError

from core.config import settings
from utils.logging import logger

class ChatOllama:
    
    def __init__(
        self,
        host: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_MODEL_NAME,
        temperature: float = 1.0,
        timeout: Optional[float] = None,
    ):
        self.host = host
        self.model = model
        self.temperature = temperature
        
        self.client = AsyncClient(
            host=self.host,
            timeout=timeout,
        )

    async def generate_response(self, messages: list) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            stream = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"temperature": self.temperature},
            )
            
            async for chunk in stream:
                yield {
                    "message": {
                        "content": chunk.message.content if chunk.message else "",
                    },
                    "done": chunk.done,
                }
                
        except ResponseError as e:
            error_msg = self._handle_ollama_error(e)
            logger.error(f"Ollama API error: {error_msg}")
            raise  
        except Exception as e:
            logger.error(f"Unexpected Ollama error: {str(e)}")
            raise ConnectionError(f"Ollama communication failed: {str(e)}")
    
    def _handle_ollama_error(self, error: ResponseError) -> str:
        error_mapping = {
            404: "Model not found or Ollama not running",
            500: "Ollama internal server error",
            524: "Request timeout",
        }
        
        if error.status_code in error_mapping:
            return error_mapping[error.status_code]
        
        return f"HTTP {error.status_code}: {error.error}"