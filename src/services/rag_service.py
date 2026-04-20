import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from models.base import ChatBase
from db.qdrant import QdrantDBClient
from schemas.vector import VectorItem
from schemas.docling import HeaderMeta, ChunkMeta
from services.docling_worker import DoclingWorker
from core.config import settings
from utils.logging import logger

class RagService:
    
    def __init__(self, embedder: ChatBase):
        self.embedder = embedder
        self.docling_worker = DoclingWorker()
        self.vector_store: Optional[QdrantDBClient] = None

    async def init_vector_store(self, hard_init: bool = False) -> None:
        logger.info(f"Initializing vector store (hard_init={hard_init})")
        if self.vector_store:
            logger.info("Vector store already initialized, skipping")
            return
        self.vector_store = QdrantDBClient()
        
        if hard_init:
            logger.info("Hard reset requested – clearing vector store")
            await self.vector_store.reset()
        
        for collection in [settings.HEADERS_COLLECTION, settings.CHUNKS_COLLECTION]:
            if not await self.vector_store.has_collection(collection):
                logger.info(f"Collection '{collection}' does not exist, adding documents from {settings.DATA_DIR}")
                await self.add_docs_in_vector_store(settings.DATA_DIR)
                break
            else:
                logger.info(f"Collection '{collection}' already exists") 

    async def add_docs_in_vector_store(self, folder: Path, extensions: List = [".pdf", ".docx"]) -> None:
        logger.info(f"Starting document ingestion from {folder} with extensions {extensions}")
        if not folder.exists():
            raise FileNotFoundError(f"Папка не найдена: {folder}")    
        if not folder.is_dir():
            raise NotADirectoryError(f"Указанный путь не является папкой: {folder}")
        
        for ext in extensions:
            clean_ext = ext[1:] if ext.startswith('.') else ext
            pattern = f"*.{clean_ext}"
            
            for file_path in folder.glob(pattern):
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
        logger.info("Document ingestion completed")
       
    async def get_relevant_docs(self, query: str, limit: int = 3) -> Tuple[str, List[Dict[str, str]]]:
        logger.info(f"Retrieving relevant documents for query: '{query}', limit={limit}")
        vector = await self.embedder.embed_query(query)
        
        search_result = await self.vector_store.search(
            collection_name=settings.CHUNKS_COLLECTION,
            vector=vector,
            limit=limit,
        )
        
        if not search_result:
            logger.info("No relevant documents found")
            return "", []
        
        logger.info(f"Found {len(search_result.documents)} relevant documents")

        formatted_texts = []
        docs = []
        for doc, meta in zip(search_result.documents, search_result.metadatas):
            metadata_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
            formatted_texts.append(
                f"Текст:\n{doc}\n\nМетаданные:\n{metadata_str}"
            )
            
            title = meta.get("source_file", "Неизвестный документ")
            docs.append({
                "doc_title": title,
            })
        
        logger.debug("Formatted document chunks for response")
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