from beanie import Document, Indexed, before_event, Replace, Insert
from pydantic import Field
from datetime import datetime
from typing import List

from entities.enums.document_type import DocumentType
from entities.section import Section

class Template(Document):

    type: DocumentType = Indexed(unique=True)
    requirements: str = Field(default="")
    sections: List[Section] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Settings:
        name = "templates"

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
        return f"Template(id={self.id}, type={self.type.value})"