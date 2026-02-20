from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class VectorItem(BaseModel):
    id: str
    text: str
    vector: List[float | int]
    metadata: Dict[str, Any]

class GetResult(BaseModel):
    ids: Optional[List[str]]
    documents: Optional[List[str]]
    metadatas: Optional[List[Any]]

class SearchResult(GetResult):
    distances: Optional[List[float | int]]
