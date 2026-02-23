from pathlib import Path
from typing import List
import uuid

from langchain_huggingface import HuggingFaceEmbeddings

from db.qdrant import QdrantDBClient
from schemas.vector import VectorItem
from core.config import settings
from utils.data_working import read_file, pdf_to_markdown, split_text_into_chunks, save_markdown
from utils.logging import logger

class RagService:

    def __init__(self):        
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDER_MODEL_NAME,
            model_kwargs={
                # "device": "cuda" if th.cuda.is_available() else "cpu"
                "device": "cpu",
                "trust_remote_code": True,
            },
            encode_kwargs={"normalize_embeddings": True},
        )
    
    async def init_vector_store(self, hard_init: bool = False):
        self.vector_store = QdrantDBClient()
        
        if hard_init:
            await self.vector_store.reset()
            logger.success("Векторное хранилище полностью очищено")
            
        if not await self.vector_store.has_collection(settings.COLLECTION_NAME):
            await self.add_docs_in_vector_store(settings.DATA_DIR)
            logger.success("В векторное хранилище загружены документы")
    
    async def add_docs_in_vector_store(self, folder: Path, extensions: List = ['.pdf']) -> None:
        if not folder.exists():
            raise FileNotFoundError(f"Папка не найдена: {folder}")    
        if not folder.is_dir():
            raise NotADirectoryError(f"Указанный путь не является папкой: {folder}")
        
        for ext in extensions:
            clean_ext = ext[1:] if ext.startswith('.') else ext
            pattern = f"*.{clean_ext}"
            
            for file_path in folder.glob(pattern):
                try:
                    text = ""
                    if clean_ext == "pdf":
                        text = pdf_to_markdown(file_path)
                        md_file = settings.DATA_MD_DIR / f"{file_path.stem}.md"
                        save_markdown(text, md_file)
                    else:
                        text = read_file(file_path)
                    
                    chunks = split_text_into_chunks(text)
                    
                    vector_items = self._create_vector_items(file_path, chunks)
                    
                    await self.vector_store.upsert(settings.COLLECTION_NAME, vector_items)
                    
                    logger.info(f"Обработан файл {file_path}, создано {len(chunks)} чанков.")
                    
                except Exception as e:
                    print(f"Ошибка при обработке файла {file_path}: {e}")
                    continue
                
    async def get_relevant_docs(self, query: str, limit: int = 2):
        vector = self.embedding_function.embed_query(query)
        
        search_result = await self.vector_store.search(
            collection_name=settings.COLLECTION_NAME,
            vector=vector,
            limit=limit,
        )
        
        if not search_result:
            return [], []
        
        formatted_texts = []
        docs = []
        
        for doc, meta in zip(search_result.documents, search_result.metadatas):
            metadata_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
            formatted_texts.append(
                f"Текст:\n{doc}\n\nМетаданные:\n{metadata_str}"
            )
            
            title = meta.get("doc_title", "Не известный документ")
            docs.append({
                "doc_title": title,
            })
        
        return "\n---\n".join(formatted_texts), docs
                
    def _create_vector_items(self, file_path: Path, chunks: List[str]):
        doc_id = file_path.stem
        doc_title = file_path.name
        total_chunks = len(chunks)
        
        vectors = self.embedding_function.embed_documents(chunks)
        
        return [
            VectorItem(
                id=str(uuid.uuid4()),
                text=chunks[id],
                vector=vectors[id],
                metadata={
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "chunk_index": id,
                    "total_chunks": total_chunks,
                    "file_path": str(file_path)
                }
            )
            for id in range(total_chunks)
        ]
        
        