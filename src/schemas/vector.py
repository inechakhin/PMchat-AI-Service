from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class VectorItem(BaseModel):
    id: str
    text: str
    vector: List[float | int]
    metadata: Dict[str, Any]

class GetResult(BaseModel):
    ids: Optional[List[str]]
    texts: Optional[List[str]]
    vectors: Optional[List[List[float | int]]]
    metadatas: Optional[List[Any]]

class SearchResult(GetResult):
    distances: Optional[List[float | int]]
