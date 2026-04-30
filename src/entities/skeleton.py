from beanie import Document, Indexed, before_event, Replace, Insert
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

from entities.enums.document_type import DocumentType
from entities.enums.skeleton_state import SkeletonState
from entities.section import Section

class Skeleton(Document):

    chat_id: str = Indexed(unique=True)
    state: SkeletonState
    type: DocumentType = Field(default=None)
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
        data = self.model_dump()
        data["id"] = str(self.id)
        return data

    def __repr__(self):
        return f"Skeleton(id={self.id}, chat_id={self.chat_id}, state={self.state.value})"

class SkeletonMetadata(BaseModel):
    state: SkeletonState
    type: Optional[DocumentType] = None
    custom_type_name: Optional[str] = None
    requirements: Optional[str] = ""
    
    def get_doc_type(self):
        return self.type.value if self.type != DocumentType.UNKNOWN else self.custom_type_name