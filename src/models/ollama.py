import asyncio
import json
from typing import AsyncGenerator, Dict, Any, List, Optional
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
        self.options = {"temperature": temperature}
        
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

    async def stream(
        self, 
        messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        try:
            stream = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options=self.options,
            )
            
            async for chunk in stream:
                msg = chunk.message
                
                if msg.content and msg.content.strip():
                    yield msg.content
                
                if chunk.done:
                    break
                
        except ResponseError as e:
            error_msg = self._handle_ollama_error(e)
            logger.error(f"Ollama API error: {error_msg}")
            raise  
        except Exception as e:
            logger.error(f"Unexpected Ollama error: {str(e)}")
            raise ConnectionError(f"Ollama communication failed: {str(e)}")

    async def invoke(
        self, 
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=False,
                options=self.options,
            )
            
            message = response.message
            tool_calls = None
            
            if message.tool_calls:
                tool_calls = [
                    {
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": json.loads(call.function.arguments) if isinstance(call.function.arguments, str) else call.function.arguments,
                        }
                    }
                    for call in message.tool_calls
                ]

            return {
                "message": {
                    "content": message.content if message else "",
                    "tool_calls": tool_calls,
                }
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