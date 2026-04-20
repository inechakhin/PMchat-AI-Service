from pydantic import BaseModel, Field
from typing import List

class Section(BaseModel):

    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(default="")
    children: List["Section"] = Field(default_factory=list)

    def copy(self) -> "Section":
        return Section(
            title=self.title,
            text=self.text,
            children=[child.copy() for child in self.children]
        )

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)
