import httpx
import json

from core.config import settings
from utils.logging import logger

class ChatOllama:
    
    def __init__(
        self,
        model: str = settings.OLLAMA_MODEL_NAME,
        temperature: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.ollama_base_url = settings.OLLAMA_BASE_URL

    async def generate_response(self, messages: list):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
            },
        }
        
        logger.debug(f"Sending request to Ollama: {payload}")

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.ollama_base_url}/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse Ollama response: {e}")
                            raise ValueError(f"Invalid JSON from Ollama: {line}")
            except httpx.RequestError as e:
                logger.error(f"Ollama connection failed: {e}")
                raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama returned error: {e.response.status_code}")
                raise RuntimeError(f"Ollama error {e.response.status_code}: {str(e)}")