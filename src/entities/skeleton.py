from beanie import Document, Indexed, before_event, Replace, Insert
from pydantic import Field
from datetime import datetime
from typing import List, Optional

from enums.document_type import DocumentType
from section import Section

class Skeleton(Document):

    chat_id: str = Indexed(unique=True)
    type: Optional[DocumentType] = Indexed()
    custom_type_name: Optional[str] = Field(default=None)
    requirements: str = Field(default="")
    document: List[Section] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Settings:
        name = "skeletons"

    @before_event([Replace, Insert])
    def set_timestamps(self):
        now = datetime.now()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now

    def to_dict(self) -> dict:
        data = self.model_dump(exclude={"id", "chat"})
        data["id"] = str(self.id)
        data["chat_id"] = str(self.chat_id)
        return data

    def __repr__(self):
        return f"Skeleton(id={self.id}, chat_id={self.chat_id}, type={self.type.value})"