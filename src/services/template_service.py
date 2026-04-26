import json
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from typing import AsyncGenerator, List, Dict, Any

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from db.qdrant import QdrantDBClient
from entities.enums.document_type import DocumentType
from entities.section import Section
from models.base import ChatBase
from repositories.template_repository import TemplateRepository
from services.docling_worker import DoclingWorker
from core.config import settings
from core.prompts import TEMPLATE_GENERATING_PROMPT, TEMPLATE_BUILDER_PROMPT
from utils.data_working import get_files_by_ext, get_document_type
from utils.logging import logger

LIMIT = 5000

class TemplateService:
    
    def __init__(
        self,
        template_repository: TemplateRepository,
        docling_worker: DoclingWorker,
        vector_store: QdrantDBClient,
        llm: ChatBase,
    ):
        self.template_repository = template_repository
        self.docling_worker = docling_worker
        self.vector_store = vector_store
        self.llm = llm

    async def init_template_store(self, hard_init: bool = False) -> None:
        logger.info(f"Инициализация хранилища для шаблонов (hard_init={hard_init})")
        
        if hard_init:
            for doc_type in DocumentType:
                if await self.template_repository.exist_by_type(doc_type):
                    await self.template_repository.delete_by_type(doc_type)
        
        await self._add_templates_from_files(settings.TEMPLATES_DIR)
        for doc_type in DocumentType:
            if (doc_type == DocumentType.UNKNOWN or
                await self.template_repository.exist_by_type(doc_type)):
                return
            await self._build_template(doc_type)

    async def get_template(self, doc_type: DocumentType) -> AsyncGenerator[Section, None]:
        logger.info(f"Возврщаем шаблон для типа: {doc_type.value}")
       
        total = await self.template_repository.get_total_sections_count(doc_type)
        if total == 0:
            logger.warning(f"Шаблон для типа {doc_type.value} пуст или не найден")
            return
        
        for idx in range(total):
            section = await self.template_repository.get_section_model(doc_type, idx)
            if section:
                yield section

    async def generate_template(self, custom_type_name: str, requirements: str) -> AsyncGenerator[Section, None]:
        logger.info(f"Генерация шаблона для типа: {custom_type_name}")

        request = [
            {
                "role": "system",
                "content": TEMPLATE_GENERATING_PROMPT.format(
                    doc_type=custom_type_name,
                    requirements=requirements,
                ),
            }
        ]

        response = await self.llm.invoke(request)
        response_text = response["message"].get("content")
        sections = self._parse_llm_response(response_text)

        for section in sections:
            yield section

    async def _add_templates_from_files(self, folder: Path, extensions: List = [".docx"]) -> None:
        logger.info("Создание шаблонов на основе существующих файлов")
        
        async for file_path in get_files_by_ext(folder, extensions):
            doc_type = get_document_type(file_path)      
            if await self.template_repository.exist_by_type(doc_type):
                continue
              
            await self.template_repository.create_empty(doc_type)
            logger.debug(f"Шаблон {doc_type} инициализирован в базе")

            async for root_section in self.docling_worker.process_template_document(file_path):
                await self.template_repository.add_section(doc_type, root_section)
                logger.debug(f"Секция '{root_section.title}' добавлена в БД")

            logger.info(f"Шаблон для {doc_type} был загружен в базу")

    async def _build_template(self, doc_type: DocumentType) -> None:
        logger.info(f"Начало сборки шаблона через кластеризацию для: {doc_type.value}")
        
        headers_get_result = await self.vector_store.query(
            collection_name=settings.HEADERS_COLLECTION,
            filter_dict={"doc_type": doc_type.value},
            with_vectors=True,
            limit=LIMIT,
        )
        
        if not headers_get_result:
            logger.warning("Нет данных в БД для кластеризации")
            return None
          
        dist_matrix = cosine_distances(headers_get_result.vectors)
        
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            metric='precomputed', 
            linkage='average',
            distance_threshold=settings.DISTANCE_THRESHOLD
        )
        labels = clustering.fit_predict(dist_matrix)

        clusters: Dict[int, Dict[str, Any]] = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = {"titles": [], "levels": []}
            clusters[label]["titles"].append(headers_get_result.texts[i])
            clusters[label]["levels"].append(headers_get_result.metadatas[i].get("section_level", 1))

        min_cluster_size = max(2, len(headers_get_result) * 0.05)
        valid_clusters = [c for c in clusters.values() if len(c["titles"]) >= min_cluster_size]        
        
        final_sections = await self._process_clusters_with_llm(doc_type.value, valid_clusters)

        await self.template_repository.create_empty(doc_type)
        for section in final_sections:
            await self.template_repository.add_section(doc_type, section)
         
        logger.info(f"Шаблон для {doc_type.value} успешно сгенерирован и сохранен.")

    async def _process_clusters_with_llm(self, doc_type: str, clusters: List[Dict[str, Any]]) -> List[Section]:
        cluster_summaries = ""
        for idx, cluster in enumerate(clusters):
            unique_titles = list(set(cluster["titles"]))
            avg_level = round(sum(cluster["levels"]) / len(cluster["levels"]))
            cluster_summaries += f"""
                Кластер #{idx}:\n
                - Варианты названий: {', '.join(unique_titles)}\n
                - Средний уровень вложенности: {avg_level}\n\n
            """

        request = [
            {
                "role": "system",
                "content": TEMPLATE_BUILDER_PROMPT.format(
                    doc_type=doc_type,
                    cluster_summaries=cluster_summaries
                ),
            }
        ]
        
        response = await self.llm.invoke(request)
        response_text = response["message"].get("content")
        return self._parse_llm_response(response_text)
    
    def _parse_llm_response(self, response_text: str) -> List[Section]:
        try:
            clean_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(clean_text)
            
            sections_adapter = TypeAdapter(List[Section])
            sections = sections_adapter.validate_python(data)
            
            return sections
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка декодирования JSON: {e}\nТекст ответа:\n{response_text}")
            return []
        except ValidationError as e:
            logger.error(f"Ошибка валидации Pydantic: {e}\nДанные:\n{data}")
            return []
