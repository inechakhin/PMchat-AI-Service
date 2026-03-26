import asyncio
import json
from typing import AsyncGenerator, Tuple, List, Dict, Any

from models.ollama import ChatOllama
from schemas.ai import AiRequest, AiMessage, AiAttachment, AiResponse
from core.llm_tools import TOOLS
from core.prompts import SYSTEM_PROMPT, CHAT_TITLE_PROMPT
from services.rag_service import RagService
from utils.logging import logger

class AiService:
    
    def __init__(
        self, 
        llm: ChatOllama,
        rag: RagService,
    ):
        self.llm = llm
        self.rag = rag
        self.tools = TOOLS
    
    async def generate_response(self, request: AiRequest) -> AsyncGenerator[AiResponse, None]:
        try:
            llm_messages = self._build_llm_messages(SYSTEM_PROMPT, request.messages)
            response = await self.llm.invoke(llm_messages, tools=self.tools)
        
            llm_messages, docs = await self._handle_tool_calls(llm_messages, response)
            
            async for token in self.llm.stream(llm_messages):
                if token:
                    yield AiResponse(token=token).model_dump_json() + "\n"
            
            yield AiResponse(attachments=self._build_attachments(docs)).model_dump_json() + "\n"
            
            if len(request.messages) == 1:
                chat_title = await self._generate_chat_title(request)
                yield AiResponse(chat_title=chat_title).model_dump_json() + "\n"
              
            yield AiResponse(is_end=True).model_dump_json() + "\n"
            
        except asyncio.CancelledError:
            logger.info("Generation cancelled by user")
            yield AiResponse(is_end=True).model_dump_json() + "\n"
        except ConnectionError as e:
            logger.error(f"Ollama connection failed: {e}")
            yield AiResponse(
                is_error=True,
                error_message=f"Ollama service unavailable: {str(e)}",
            ).model_dump_json() + "\n"
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            yield AiResponse(
                is_error=True,
                error_message=f"Generation failed: {str(e)}",
            ).model_dump_json() + "\n"
    
    async def _generate_chat_title(self, request: AiRequest) -> str:
        chat_title_messages = self._build_llm_messages(CHAT_TITLE_PROMPT, request.messages[-1:])
        response = await self.llm.invoke(chat_title_messages, tools=None)
        return response["message"]["content"]

    def _build_llm_messages(
        self, 
        prompt: str, 
        ai_messages: List[AiMessage],
    ) -> List[Dict[str, str]]:
        llm_messages = [
            {
                "role": "system",
                "content": prompt,
            }
        ]
        llm_messages.extend(
            {
                "role": message.sender_type, 
                "content": message.text
            }
            for message in ai_messages
        )
        return llm_messages
    
    async def _handle_tool_calls(
        self, 
        llm_messages: List[Dict[str, str]], 
        response: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        docs = []
        
        tool_calls = response["message"].get("tool_calls")
        if not tool_calls:
            return llm_messages, docs
        
        for tool_call in tool_calls:
            llm_messages.append({
                "role": "assistant",
                "content": "",
                "function_call": tool_call["function"],
            })
            
            if tool_call["function"]["name"] == "search_project_docs":
                query = tool_call["function"]["arguments"]["query"]
                top_k = tool_call["function"]["arguments"]["top_k"]
                
                tool_response, docs = await self.rag.get_relevant_docs(query=query, limit=top_k)
                
                llm_messages.append({
                    "role": "tool",
                    "name": "search_project_docs",
                    "content": json.dumps({"retrieved_docs": tool_response}, ensure_ascii=False),
                })
        
        return llm_messages, docs
    
    def _build_attachments(
        self,
        docs: List[Dict[str, Any]],
    ) -> List[AiAttachment]:
        return [
            AiAttachment(
                doc_title=doc["doc_title"]
            )
            for doc in docs
        ]