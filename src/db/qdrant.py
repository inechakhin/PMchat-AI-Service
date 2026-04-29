from typing import Optional, List

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import models

from schemas.vector import VectorItem, GetResult, SearchResult
from core.config import settings
from utils.logging import logger

class QdrantDBClient:
    
    def __init__(
        self,
        host: str = settings.QDRANT_HOST,
        port: int = settings.QDRANT_PORT,
        api_key: str = settings.QDRANT_API_KEY,
        timeout: Optional[float] = None,
    ):
        self.qdrant_base_url = f"http://{host}:{port}"
        
        self.client = AsyncQdrantClient(
            url=self.qdrant_base_url,
            api_key=api_key,
            timeout=timeout,
        )
    
    async def has_collection(self, collection_name: str) -> bool:
        return await self.client.collection_exists(collection_name)
    
    async def delete_collection(self, collection_name: str) -> bool:
        return await self.client.delete_collection(collection_name)
    
    async def search(
        self,
        collection_name: str,
        vector: List[float | int],
        filter: Optional[dict] = None,
        with_vectors: Optional[bool] = False,
        limit: Optional[int] = None,
    ) -> Optional[SearchResult]:
        query_filter = None
        if filter:
            field_conditions = self._create_field_conditions(filter)
            query_filter = models.Filter(should=field_conditions)
        
        query_response = await self.client.query_points(
            collection_name=collection_name,
            query=vector,
            query_filter=query_filter,
            with_vectors=with_vectors,
            limit=limit,
        )
        
        return self._points_to_search_result(query_response.points)
    
    async def query(
        self,
        collection_name: str,
        filter: dict,
        with_vectors: Optional[bool] = False,
        limit: Optional[int] = None,
    ) -> Optional[GetResult]:
        if not await self.has_collection(collection_name):
            return None
        try:
            field_conditions = self._create_field_conditions(filter)
            scroll_filter = models.Filter(should=field_conditions)
            
            points = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                with_vectors=with_vectors,
                limit=limit,
            )
            
            return self._points_to_get_result(points[0])
        except Exception as e:
            logger.exception(f"Ошибочный query для коллекции '{collection_name}': {e}")
            return None
    
    async def insert(self, collection_name: str, items: List[VectorItem]) -> None:
        await self._create_collection_if_not_exists(collection_name, len(items[0].vector))
        points = self._create_points(items)
        await self.client.upload_points(collection_name, points)
    
    async def upsert(self, collection_name: str, items: List[VectorItem]) -> models.UpdateResult:
        await self._create_collection_if_not_exists(collection_name, len(items[0].vector))
        points = self._create_points(items)
        return await self.client.upsert(collection_name, points)
    
    async def delete(
        self,
        collection_name: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ) -> models.UpdateResult:
        if ids:
            return await self.client.delete(
                collection_name=collection_name,
                points_selector=ids,
                # wait=True, if need block
            )
        elif filter:
            field_conditions = self._create_field_conditions(filter)
            delete_filter = models.Filter(must=field_conditions)
            
            return await self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=delete_filter),
            )
    
    async def reset(self) -> None:
        response = await self.client.get_collections()
        collections = response.collections
        for collection in collections:
            await self.client.delete_collection(collection_name=collection.name)
    
    async def _create_collection(self, collection_name: str, dimension: int) -> None:
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
                on_disk=settings.QDRANT_ON_DISK,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=settings.QDRANT_HNSW_M,
            ),
        )
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name='metadata.doc_type',
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=False,
                on_disk=settings.QDRANT_ON_DISK,
            ),
        )
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name='metadata.source_file',
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=False,
                on_disk=settings.QDRANT_ON_DISK,
            ),
        )
    
    async def _create_collection_if_not_exists(self, collection_name: str, dimension: int) -> None:
        if not await self.has_collection(collection_name):
            await self._create_collection(collection_name, dimension)
    
    def _create_points(self, items: List[VectorItem]) -> List[PointStruct]:
        return [
            PointStruct(
                id=item.id,
                vector=item.vector,
                payload={"text": item.text, "metadata": item.metadata},
            )
            for item in items
        ]
    
    def _points_to_get_result(self, points) -> GetResult:
        ids = []
        vectors = []
        documents = []
        metadatas = []

        for point in points:
            payload = point.payload
            ids.append(point.id)
            vectors.append(point.vector if point.vector is not None else [])
            documents.append(payload["text"])
            metadatas.append(payload["metadata"])

        return GetResult(
            ids=ids,
            vectors=vectors,
            texts=documents,
            metadatas=metadatas,
        )
        
    def _points_to_search_result(self, points) -> SearchResult:
        get_result = self._points_to_get_result(points)
        return SearchResult(
            ids=get_result.ids,
            vectors=get_result.vectors,
            texts=get_result.texts,
            metadatas=get_result.metadatas,
            # qdrant distance is [-1, 1], normalize to [0, 1]
            distances=[(point.score + 1.0) / 2.0 for point in points],
        )
    
    def _create_field_conditions(self, filter: dict) -> List[models.FieldCondition]:
        return [
            models.FieldCondition(
                key=f"metadata.{key}", 
                match=models.MatchValue(value=value),
            )
            for key, value in filter.items()
        ]