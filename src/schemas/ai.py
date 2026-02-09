from typing import List, Literal
from pydantic import BaseModel

class AiMessage(BaseModel):
    sender_type: Literal["user", "assistant"]
    text: str

class AiRequest(BaseModel):
    chat_id: str
    messages: List[AiMessage]