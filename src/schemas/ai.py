from typing import List, Literal, Optional
from pydantic import BaseModel

class AiMessage(BaseModel):
    sender_type: Literal["user", "assistant"]
    text: str

class AiRequest(BaseModel):
    chat_id: str
    messages: List[AiMessage]
    
class AiAttachment(BaseModel):
    doc_title: str

class AiResponse(BaseModel):
    token: Optional[str] = None
    chat_title: Optional[str] = None
    attachments: Optional[List[AiAttachment]] = None
    error_message: Optional[str] = None
    is_error: bool = False
    is_end: bool = False