import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from models.base import ChatBase
from db.qdrant import QdrantDBClient
from schemas.vector import VectorItem
from schemas.docling import HeaderMeta, ChunkMeta
from services.docling_worker import DoclingWorker
from core.config import settings
from utils.data_working import get_files_by_ext
from utils.logging import logger

class RagService:
    
    def __init__(self, embedder: ChatBase):
        self.embedder = embedder
        self.docling_worker = DoclingWorker()
        self.vector_store: Optional[QdrantDBClient] = None

    async def init_vector_store(self, hard_init: bool = False) -> None:
        logger.info(f"Инициализация векторного хранилища (hard_init={hard_init})")
        if self.vector_store:
            logger.info("Векторное хранилище уже инициализировано, пропускаем")
            return
        self.vector_store = QdrantDBClient()
        
        if hard_init:
            logger.info("Запрошен полный сброс – очистка векторного хранилища")
            await self.vector_store.reset()
        
        for collection in [settings.HEADERS_COLLECTION, settings.CHUNKS_COLLECTION]:
            if not await self.vector_store.has_collection(collection):
                logger.info(f"Коллекция '{collection}' не существует, добавляем документы из {settings.DOCS_DIR}")
                await self.add_docs_in_vector_store(settings.DOCS_DIR)
                break
            else:
                logger.info(f"Коллекция '{collection}' уже существует") 

    async def add_docs_in_vector_store(self, folder: Path, extensions: List = [".pdf", ".docx"]) -> None:
        logger.info(f"Начинаем индексацию документов из {folder} с расширениями {extensions}")
        async for file_path in get_files_by_ext(folder, extensions):
            try:
                heading_text_to_id: Dict[str, str] = {}
                header_batch: List[VectorItem] = []
                chunk_batch: List[VectorItem] = []
                BATCH_SIZE = 20
                
                async for meta in self.docling_worker.process_document(file_path):
                    if isinstance(meta, HeaderMeta):
                        parent_ids = [
                            heading_text_to_id.get(pt) 
                            for pt in meta.parent_titles
                            if pt in heading_text_to_id
                        ]
                        header_vector = await self._create_header_vector_item(meta, parent_ids)
                        heading_text_to_id[header_vector.text] = header_vector.id
                        header_batch.append(header_vector)
                        if len(header_batch) >= BATCH_SIZE:
                            await self._flush_batch(settings.HEADERS_COLLECTION, header_batch)
                            header_batch.clear()
                            
                    elif isinstance(meta, ChunkMeta):
                        heading_ids = {
                            str(idx): heading_text_to_id[h_title]
                            for idx, h_title in enumerate(meta.heading_titles, start=1)
                            if h_title in heading_text_to_id
                        }
                        chunk_vector = await self._create_chunk_vector_item(meta, heading_ids)
                        chunk_batch.append(chunk_vector)
                        if len(chunk_batch) >= BATCH_SIZE:
                            await self._flush_batch(settings.CHUNKS_COLLECTION, chunk_batch)
                            chunk_batch.clear()
                        
                if header_batch:
                    await self._flush_batch(settings.HEADERS_COLLECTION, header_batch)
                if chunk_batch:
                    await self._flush_batch(settings.CHUNKS_COLLECTION, chunk_batch)
                
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path}: {e}")
                continue
        logger.info("Индексация документов завершена")

    async def search_with_boosting(
        self,
        query: str,
        doc_type: str,
        header: str,
        limit: int = 5,
    ) -> str:
        logger.info(f"Поиск с бустингом: doc_type='{doc_type}', header='{header}'")
        
        chunk_vector = await self.embedder.embed_query(query)
        chunks_result = await self.vector_store.search(
            collection_name=settings.CHUNKS_COLLECTION,
            vector=chunk_vector,
            limit=limit * 3,
        )
        if not chunks_result or not chunks_result.texts:
            logger.info("Релевантные чанки не найдены")
            return ""
        logger.debug(f"Получено {len(chunks_result.texts)} чанков для бустинга")

        header_vector = await self.embedder.embed_query(header)
        headers_result = await self.vector_store.search(
            collection_name=settings.HEADERS_COLLECTION,
            vector=header_vector,
            limit=limit,
        )
        header_ids = set()
        if headers_result and headers_result.ids:
            header_ids = set(headers_result.ids)
            logger.debug(f"Найдено {len(header_ids)} релевантных заголовков")
        else:
            logger.debug("Релевантные заголовки не найдены")

        boosted_items = []
        for text, metadata, distance in zip(
            chunks_result.texts,
            chunks_result.metadatas,
            chunks_result.distances,
        ):
            boosted_score = distance

            if metadata.get("doc_type") == doc_type:
                boosted_score += settings.TYPE_BOOST

            heading_values = set(metadata.get("headings", {}).values())
            if heading_values.intersection(header_ids):
                boosted_score += settings.HEADING_BOOST

            boosted_score = min(boosted_score, 1.0)

            boosted_items.append({
                "text": text,
                "metadata": metadata,
                "boosted_score": boosted_score,
                "original_distance": distance,
            })

        boosted_items.sort(key=lambda x: x["boosted_score"], reverse=True)
        final_results = [item["text"] for item in boosted_items[:limit]]
        
        logger.info(f"Возвращено {len(final_results)} чанков после бустинга")
        return "\n---\n".join(final_results)
    
    async def get_relevant_docs(self, query: str, limit: int = 3) -> Tuple[str, List[Dict[str, str]]]:
        logger.info(f"Поиск релевантных документов по запросу: '{query}', лимит={limit}")
        vector = await self.embedder.embed_query(query)
        
        search_result = await self.vector_store.search(
            collection_name=settings.CHUNKS_COLLECTION,
            vector=vector,
            limit=limit,
        )
        
        if not search_result:
            logger.info("Релевантные документы не найдены")
            return "", []
        
        logger.info(f"Найдено {len(search_result.texts)} релевантных фрагментов")

        formatted_texts = []
        docs = []
        for doc, meta in zip(search_result.texts, search_result.metadatas):
            metadata_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
            formatted_texts.append(
                f"Текст:\n{doc}\n\nМетаданные:\n{metadata_str}"
            )
            
            title = meta.get("source_file", "Неизвестный документ")
            docs.append({
                "doc_title": title,
            })
        
        logger.info("Фрагменты документов отформатированы для ответа")
        return "\n---\n".join(formatted_texts), docs
    
    async def _create_header_vector_item(self, meta: HeaderMeta, parent_ids: List[str]) -> VectorItem:
        vector = await self.embedder.embed_query(meta.title)

        return VectorItem(
            id=str(uuid.uuid4()),
            text=meta.title,
            vector=vector,
            metadata={
                "section_level": meta.section_level,
                "doc_type": meta.doc_type,
                "parent_ids": parent_ids,
                "source_file": meta.source_file,
            }
        )
    
    async def _create_chunk_vector_item(self, meta: ChunkMeta, heading_ids: Dict[str, str]) -> VectorItem:
        vector = await self.embedder.embed_query(meta.text)

        return VectorItem(
            id=str(uuid.uuid4()),
            text=meta.text,
            vector=vector,
            metadata={
                "doc_type": meta.doc_type,
                "headings": heading_ids,
                "page_numbers": meta.page_numbers,
                "source_file": meta.source_file,
            }
        )
    
    async def _flush_batch(self, collection_name: str, batch: List[VectorItem]) -> None:
        if not batch:
            return
        await self.vector_store.upsert(collection_name, batch)