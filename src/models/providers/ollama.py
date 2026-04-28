import asyncio
import json
from typing import AsyncGenerator, Dict, Any, List, Optional
from ollama import AsyncClient, ResponseError

from models.base import ChatBase
from core.config import settings
from utils.logging import logger

class ChatOllama(ChatBase):
    
    def __init__(
        self,
        host: str = settings.OLLAMA_HOST,
        port: int = settings.OLLAMA_PORT,
        model: str = settings.OLLAMA_MODEL_NAME,
        embedder: str = settings.OLLAMA_EMBEDDER_NAME,
        temperature: float = 1.0,
        timeout: Optional[float] = None,
    ):
        self.ollama_base_url = f"http://{host}:{port}"
        self.model = model
        self.embedder = embedder
        self.options = {"temperature": temperature}
        
        self.client = AsyncClient(
            host=self.ollama_base_url,
            timeout=timeout,
        )

    async def pull_model(
        self,
        model: str,
        retries: int = 30,
        delay: float = 2.0,
    ):
        for _ in range(retries):
            try:
                await self.client.list()
                logger.info("Ollama API доступно")
                break
            except Exception:
                logger.info("Ожидание Ollama API...")
                await asyncio.sleep(delay)
        else:
            raise RuntimeError("Ollama API не доступно")
        
        models = await self.client.list()
        model_names = {m.model for m in models.models}
        if model not in model_names:
            logger.info(f"Загружаем в Ollama модель: {model}")
            await self.client.pull(model)
            logger.info("Модель успешно загружена")
        else:
            logger.info(f"Модель {model} уже существует")
    
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
                
                yield msg.content
                
                if chunk.done:
                    break
                
        except ResponseError as e:
            error_msg = self._handle_ollama_error(e)
            logger.error(f"Ollama API ошибка: {error_msg}")
            raise  
        except Exception as e:
            logger.error(f"Неожиданная Ollama ошибка: {str(e)}")
            raise ConnectionError(f"Связь с Ollama не удалась: {str(e)}")

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
                            "arguments": json.loads(call.function.arguments) 
                                if isinstance(call.function.arguments, str) 
                                else call.function.arguments,
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
            logger.error(f"Ollama API ошибка: {error_msg}")
            raise  
        except Exception as e:
            logger.error(f"Неожиданная Ollama ошибка: {str(e)}")
            raise ConnectionError(f"Связь с Ollama не удалась: {str(e)}")

    async def embed_query(self, text: str) -> List[float]:
        try:
            response = await self.client.embed(
                model=self.embedder,
                input=text
            )
            
            return response.embeddings[0] if response else ()
            
        except ResponseError as e:
            error_msg = self._handle_ollama_error(e)
            logger.error(f"Ollama API ошибка: {error_msg}")
            raise  
        except Exception as e:
            logger.error(f"Неожиданная Ollama ошибка: {str(e)}")
            raise ConnectionError(f"Связь с Ollama не удалась: {str(e)}")
        
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.client.embed(
                model=self.embedder,
                input=texts
            )
            
            return response.embeddings if response else ()
            
        except ResponseError as e:
            error_msg = self._handle_ollama_error(e)
            logger.error(f"Ollama API ошибка: {error_msg}")
            raise  
        except Exception as e:
            logger.error(f"Неожиданная Ollama ошибка: {str(e)}")
            raise ConnectionError(f"Связь с Ollama не удалась: {str(e)}")

    def _handle_ollama_error(self, error: ResponseError) -> str:
        error_mapping = {
            404: "Model not found or Ollama not running",
            500: "Ollama internal server error",
            524: "Request timeout",
        }
        
        if error.status_code in error_mapping:
            return error_mapping[error.status_code]
        
        return f"HTTP {error.status_code}: {error.error}"