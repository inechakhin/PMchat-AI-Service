from pydantic import BaseModel
from typing import List

class HeaderMeta(BaseModel):
    title: str
    section_level: int
    doc_type: str
    parent_titles: List[str]
    source_file: str

class ChunkMeta(BaseModel):
    text: str
    doc_type: str
    heading_titles: List[str]
    page_numbers: List[int]
    source_file: str