import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from ollama import AsyncClient, ResponseError

from core.config import settings
from utils.logging import logger

class ChatOllama:
    
    def __init__(
        self,
        host: str = settings.OLLAMA_HOST,
        port: int = settings.OLLAMA_PORT,
        model: str = settings.OLLAMA_MODEL_NAME,
        temperature: float = 1.0,
        timeout: Optional[float] = None,
    ):
        self.ollama_base_url = f"http://{host}:{port}"
        self.model = model
        self.temperature = temperature
        
        self.client = AsyncClient(
            host=self.ollama_base_url,
            timeout=timeout,
        )

    async def pull_model(self, retries: int = 30, delay: float = 2.0) -> None:
        # ensure_ready
        for _ in range(retries):
            try:
                await self.client.list()
                logger.info("Ollama API is available")
                break
            except Exception:
                logger.info("Waiting for Ollama API...")
                await asyncio.sleep(delay)
        else:
            raise RuntimeError("Ollama API not available")

        # pull_model
        models = await self.client.list()
        model_names = {m.model for m in models.models}

        if self.model not in model_names:
            logger.info(f"Pulling Ollama model: {self.model}")
            await self.client.pull(self.model)
            logger.info("Model pulled successfully")
        else:
            logger.info(f"Model {self.model} already exists")

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