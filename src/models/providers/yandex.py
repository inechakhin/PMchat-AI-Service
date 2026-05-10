import json
from typing import AsyncGenerator, Dict, Any, List, Optional
from openai import AsyncOpenAI

from models.base import ChatBase
from core.config import settings
from utils.logging import logger

class ChatYandex(ChatBase):
    
    def __init__(
        self,
        folder_id: str = settings.YANDEX_FOLDER_ID,
        api_key: str = settings.YANDEX_API_KEY,
        model: str = settings.YANDEX_MODEL,
        embedder_doc: str = settings.YANDEX_EMBEDDER_DOC,
        embedder_query: str = settings.YANDEX_EMBEDDER_QUERY,
        temperature: float = 1.0,
        timeout: Optional[float] = None,
        max_retries: int = 2,
    ):
        self.folder_id = folder_id
        self.api_key = api_key
        self.model = f"gpt://{folder_id}/{model}"
        self.embedder_doc = f"emb://{folder_id}/{embedder_doc}"
        self.embedder_query = f"emb://{folder_id}/{embedder_query}"
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = "none"

        self.client = AsyncOpenAI(
            base_url="https://ai.api.cloud.yandex.net/v1",
            api_key=api_key,
            project=folder_id,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def stream(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        try:
            input_text = self._format_messages_for_responses(messages)

            stream = await self.client.responses.create(
                model=self.model,
                input=input_text,
                stream=True,
                temperature=self.temperature,
                extra_body={"reasoning_effort": self.reasoning_effort},
            )

            async for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
                elif event.type == "response.completed":
                    break

        except Exception as e:
            error_msg = self._handle_error(e)
            logger.error(f"Yandex API streaming error: {error_msg}")
            raise

    async def invoke(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        try:
            input_text = self._format_messages_for_responses(messages)

            tools_schema = None
            if tools:
                tools_schema = self._convert_tools_to_openai_format(tools)

            response = await self.client.responses.create(
                model=self.model,
                input=input_text,
                tools=tools_schema,
                temperature=self.temperature,
                extra_body={"reasoning_effort": self.reasoning_effort},
            )

            content = response.output_text or ""

            tool_calls = None
            if response.output:
                function_calls = [
                    item for item in response.output if item.type == "function_call"
                ]
                if function_calls:
                    tool_calls = []
                    for call in function_calls:
                        tool_calls.append({
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": (
                                    json.loads(call.arguments)
                                    if isinstance(call.arguments, str)
                                    else call.arguments
                                ),
                            },
                        })

            return {
                "message": {
                    "content": content,
                    "tool_calls": tool_calls,
                }
            }

        except Exception as e:
            error_msg = self._handle_error(e)
            logger.error(f"Yandex API invoke error: {error_msg}")
            raise

    async def embed_query(self, text: str) -> List[float]:
        try:
            response = await self.client.embeddings.create(
                model=self.embedder_query,
                input=text,
            )
            return response.data[0].embedding if response.data else []
        except Exception as e:
            error_msg = self._handle_error(e)
            logger.error(f"Yandex embed query error: {error_msg}")
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.client.embeddings.create(
                model=self.embedder_doc,
                input=texts,
            )
            return [item.embedding for item in response.data] if response.data else []
        except Exception as e:
            error_msg = self._handle_error(e)
            logger.error(f"Yandex embed documents error: {error_msg}")
            raise

    def _format_messages_for_responses(self, messages: List[Dict[str, str]]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(content)
        return "\n".join(formatted)

    def _convert_tools_to_openai_format(self, tools: List[Dict]) -> List[Dict]:
        openai_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                openai_tools.append({
                    "type": "function",
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
        return openai_tools if openai_tools else None

    def _handle_error(self, error: Exception) -> str:
        if hasattr(error, "status_code"):
            status_code = error.status_code
            if status_code == 401:
                return "Authentication error: invalid API key or folder_id"
            elif status_code == 404:
                return "Model not found or endpoint does not exist"
            elif status_code == 429:
                return "Rate limit exceeded"
            elif status_code >= 500:
                return "Yandex Cloud internal server error"
            else:
                return f"HTTP {status_code}: {str(error)}"
        else:
            return str(error)