import asyncio
from typing import AsyncGenerator

from models.ollama import ChatOllama
from schemas.ai import AiRequest
from core.prompts import SYSTEM_PROMPT
from utils.logging import logger
from core.exceptions.ai_generation_error import AiGenerationError

class AiService:
    
    def __init__(self):
        self.llm = ChatOllama()
    
    async def generate_stream(self, request: AiRequest) -> AsyncGenerator[str, None]:
        llm_messages = self._build_llm_messages(request)
        
        try:
            async for data_chunk in self.llm.generate_response(llm_messages):
                if "message" in data_chunk and "content" in data_chunk["message"]:
                    content = data_chunk["message"]["content"]
                    if content:
                        yield content
                
                if data_chunk.get("done", False):
                    break
                
        except asyncio.CancelledError:
            logger.info("Generation cancelled by user")
            raise
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            raise AiGenerationError(f"Generation failed: {e}")
    
    async def generate_completion(self, request: AiRequest) -> str:
        full_response = ""
        async for chunk in self.generate_stream(request):
            full_response += chunk
        return full_response
    
    def _build_llm_messages(self, request: AiRequest):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        messages.extend(
            {
                "role": message.sender_type, 
                "content": message.text
            }
            for message in request.messages
        )
        return messages

ai_service = AiService()

def get_ai_service():
    return ai_service