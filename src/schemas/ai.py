from typing import List, Literal, Optional
from pydantic import BaseModel

class AiMessage(BaseModel):
    sender_type: Literal["user", "assistant"]
    text: str

class AiRequest(BaseModel):
    chat_id: str
    chat_type: Literal["communication", "generation"]
    messages: List[AiMessage]
    
class AiSource(BaseModel):
    doc_title: str

class AiAttachment(BaseModel):
    file_name: str
    s3_key: str
    size: int = 0
    content_type: str = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

class AiResponse(BaseModel):
    token: Optional[str] = None
    chat_title: Optional[str] = None
    sources: Optional[List[AiSource]] = None
    attachments: Optional[List[AiAttachment]] = None
    error_message: Optional[str] = None
    is_error: bool = False
    is_end: bool = False