from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Optional

class ChatBase(ABC):
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass
