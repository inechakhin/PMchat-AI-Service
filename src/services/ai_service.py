import asyncio
from datetime import datetime
import json
from typing import AsyncGenerator, Tuple, List, Dict, Any, Optional

from entities.enums.document_type import DocumentType
from entities.enums.skeleton_state import SkeletonState
from entities.section import Section
from models.base import ChatBase
from repositories.skeleton_repository import SkeletonRepository
from schemas.ai import (
    AiRequest,
    AiMessage,
    AiSource,
    AiAttachment,
    AiResponse,
)
from core.llm_tools import (
    COMMUNICATION_TOOLS,
    ELICITATION_TOOLS,
    REVISION_TOOLS,
)
from core.prompts import (
    CHAT_TITLE_PROMPT,
    COMMUNICATION_PROMPT,
    ELICITATION_PROMPT,
    REVISION_PROMPT,
    GENERATING_PROMPT,
    REGENERATING_PROMPT,
    DESCRIPTION_PROMPT,
)
from services.document_exporter import DocumentExporter
from services.rag_service import RagService
from db.s3 import S3Client
from services.template_service import TemplateService
from utils.logging import logger

class AiService:
    
    def __init__(
        self, 
        skeleton_repository: SkeletonRepository,
        template_service: TemplateService,
        llm: ChatBase,
        rag: RagService,
        doc_exporter: DocumentExporter,
        s3_client: S3Client,
    ):
        self.skeleton_repository = skeleton_repository
        self.template_service = template_service
        self.llm = llm
        self.rag = rag
        self.doc_exporter = doc_exporter
        self.s3_client = s3_client
    
    async def generate_response(self, request: AiRequest) -> AsyncGenerator[AiResponse, None]:
        logger.info("Начало генерации ответа для сообщений")
        try:
            if not await self.skeleton_repository.exist_by_chat_id(request.chat_id):
                if request.chat_type == "communication":
                    await self.skeleton_repository.create_empty(request.chat_id, SkeletonState.COMMUNICATION)
                elif request.chat_type == "generation":
                    await self.skeleton_repository.create_empty(request.chat_id, SkeletonState.ELICITATION)
                else:
                    logger.error(f"Неожиданный тип чата: {request.chat_type} для чата {request.chat_id}")
                    yield AiResponse(
                        is_error=True,
                        error_message=f"Неожиданный тип чата: {request.chat_type} для чата {request.chat_id}",
                    ).model_dump_json() + "\n"
            
            metadata = await self.skeleton_repository.get_metadata_by_chat_id(request.chat_id)
            if metadata.state == SkeletonState.COMMUNICATION:
                prompt = COMMUNICATION_PROMPT
                tools = COMMUNICATION_TOOLS
            elif metadata.state == SkeletonState.ELICITATION:
                prompt = ELICITATION_PROMPT.format(
                    doc_types_list="\n".join(
                        f"- {t.value}"
                        for t in DocumentType if t != DocumentType.UNKNOWN
                    )
                )
                tools = ELICITATION_TOOLS
            elif metadata.state == SkeletonState.REVISION:
                prompt = REVISION_PROMPT.format(
                    document_type=metadata.get_doc_type(),
                )
                tools = REVISION_TOOLS
            else:
                logger.error(f"Неожиданное состояние: {metadata.state} для чата {request.chat_id}")
                yield AiResponse(
                    is_error=True,
                    error_message=f"Неожиданное состояние: {metadata.state} для чата {request.chat_id}",
                ).model_dump_json() + "\n"
            
            llm_messages = self._build_llm_messages(prompt, request.messages)
            response = await self.llm.invoke(llm_messages, tools=tools)
            
            llm_messages, attachments, sources = await self._handle_tool_calls(request.chat_id, llm_messages, response)
            
            async for token in self.llm.stream(llm_messages):
                if token:
                    yield AiResponse(token=token).model_dump_json() + "\n"

            if attachments:
                yield AiResponse(attachments=attachments).model_dump_json() + "\n"
        
            if sources:
                yield AiResponse(sources=sources).model_dump_json() + "\n"
        
            if len(request.messages) == 1:
                chat_title = await self._generate_chat_title(request)
                yield AiResponse(chat_title=chat_title).model_dump_json() + "\n"
        
            yield AiResponse(is_end=True).model_dump_json() + "\n"
        
        except asyncio.CancelledError:
            logger.info("Генерация отменена пользователем")
            yield AiResponse(is_end=True).model_dump_json() + "\n"
        except ConnectionError as e:
            logger.error(f"Ошибка подключения к Ollama: {e}")
            yield AiResponse(
                is_error=True,
                error_message=f"Ollama недоступна: {str(e)}",
            ).model_dump_json() + "\n"
        except Exception as e:
            logger.error(f"Ошибка генерации AI: {e}")
            yield AiResponse(
                is_error=True,
                error_message=f"Ошибка генерации AI: {str(e)}",
            ).model_dump_json() + "\n"
    
    async def _generate_chat_title(self, request: AiRequest) -> str:
        logger.info("Построение сообщений для генерации названия чата")
        chat_title_messages = self._build_llm_messages(CHAT_TITLE_PROMPT, request.messages[-1:])
        response = await self.llm.invoke(chat_title_messages, tools=None)
        return response["message"]["content"]

    def _build_llm_messages(
        self,
        prompt: str,
        ai_messages: List[AiMessage],
    ) -> List[Dict[str, Any]]:
        llm_messages = [
            {
                "role": "system",
                "content": prompt,
            }
        ]
        llm_messages.extend(
            {
                "role": message.sender_type, 
                "content": message.text
            }
            for message in ai_messages
        )
        return llm_messages
    
    async def _handle_tool_calls(
        self,
        chat_id: str,
        llm_messages: List[Dict[str, Any]], 
        response: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[AiAttachment], List[AiSource]]:
        attachments = []
        sources = []
        
        tool_calls = response["message"].get("tool_calls")
        if not tool_calls:
            logger.info("Вызовы инструментов не обнаружены")
            llm_messages.append({
                "role": "tool",
                "name": "none",
                "content": "Инструменты не вызваны, необходимо ответить пользователю.",
            })
            return llm_messages, attachments, sources
        
        logger.info("Обработка вызова(ов) инструментов")
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            
            if func_name == "finalize_requirements":
                doc_type_str = tool_call["function"]["arguments"].get("doc_type")
                custom_type_name = tool_call["function"]["arguments"].get("custom_type_name")
                requirements = tool_call["function"]["arguments"].get("requirements")
                logger.info(f"ИИ вызвала finalize_requirements с параметрами doc_type={doc_type_str}, custom_type_name={custom_type_name}")

                await self._init_document_skeleton(chat_id, doc_type_str, custom_type_name, requirements)
                metadata = await self.skeleton_repository.get_metadata_by_chat_id(chat_id)
                doc_type = metadata.get_doc_type()
                
                retelling = await self._generate_document(chat_id, doc_type, requirements)

                retelling=requirements

                attachment = await self._export_and_upload_document(chat_id, doc_type)
                if attachment:
                    attachments.append(attachment)
                
                llm_messages[0]["content"] = DESCRIPTION_PROMPT
                llm_messages.append({
                    "role": "tool",
                    "name": func_name,
                    "content": json.dumps({
                        "status": "success", 
                        "message": "Документ сгенерирован.",
                        "retelling": retelling,
                    }, ensure_ascii=False)
                })
                await self.skeleton_repository.update(chat_id, {"state": SkeletonState.REVISION})
            
            elif func_name == "finalize_comments":
                section_title = tool_call["function"]["arguments"]["section_title"]
                comments = tool_call["function"]["arguments"]["comments"]
                logger.info(f"ИИ вызвала finalize_comments с параметром section_title={section_title}")
                
                new_section_text = await self._regenerate_section(chat_id, section_title, comments)
                
                if new_section_text:
                    attachment = await self._export_and_upload_document(chat_id)
                    if attachment:
                        attachments.append(attachment)
                    
                    tool_content = {
                        "status": "success", 
                        "message": f"Раздел '{section_title}' успешно переписан.",
                        "new_content": new_section_text
                    }
                else:
                    tool_content = {
                        "status": "error", 
                        "message": "Раздел не найден."
                    }
                
                llm_messages[0]["content"] = DESCRIPTION_PROMPT
                llm_messages.append({
                    "role": "tool",
                    "name": func_name,
                    "content": json.dumps(tool_content, ensure_ascii=False)
                })
        
            elif func_name == "search_project_docs":
                query = tool_call["function"]["arguments"]["query"]
                top_k = tool_call["function"]["arguments"]["top_k"]
                logger.info(f"ИИ вызвала finalize_comments с параметрами query={query}, top_k={top_k}")
                
                tool_response, docs = await self.rag.get_relevant_docs(query=query, limit=top_k)
                sources = [AiSource.model_validate(doc) for doc in docs]
                
                llm_messages.append({
                    "role": "tool",
                    "name": func_name,
                    "content": json.dumps({
                        "status": "success",
                        "retrieved_docs": tool_response,
                    }, ensure_ascii=False),
                })
        
        return llm_messages, attachments, sources

    async def _init_document_skeleton(
        self,
        chat_id: str,
        doc_type_str: str,
        custom_type_name: str,
        requirements: str,
    ) -> None:
        logger.info("Инициализация скелета для документа")
        try:
            doc_type = DocumentType(doc_type_str)
        except ValueError:
            doc_type = DocumentType.UNKNOWN
            if doc_type_str:
                custom_type_name = doc_type_str
        
        update_data = {
            "type": doc_type,
            "custom_type_name": custom_type_name,
            "requirements": requirements,
        }
        
        await self.skeleton_repository.update(chat_id, update_data)
        
        if doc_type == DocumentType.UNKNOWN:
            async for section in self.template_service.generate_template(custom_type_name, requirements):
                await self.skeleton_repository.add_section(chat_id, section)
        else:
            async for section in self.template_service.get_template(doc_type):
                await self.skeleton_repository.add_section(chat_id, section)

    async def _generate_document(
        self, 
        chat_id: str,
        doc_type: str, 
        requirements: str
    ) -> str:
        total_sections = await self.skeleton_repository.get_total_sections_count(chat_id)
        logger.info(f"Начинаем полную генерацию документа. Разделов: {total_sections}")
        
        previous_context = ""
        combined_top_level_text = []

        async def process_section(sec: Section, prev_ctx: str) -> Tuple[Section, str]:
            logger.info(f"Генерация раздела: {sec.title}")
            
            rag_context = await self.rag.search_with_boosting(sec.text, doc_type, sec.title)
            
            prompt = GENERATING_PROMPT.format(
                document_type=doc_type,
                requirements=requirements,
                section_title=sec.title,
                section_requirements=sec.text,
                previous_context=prev_ctx or "Это первый раздел документа.",
                rag_context=rag_context or "Нет данных в базе."
            )

            response = await self.llm.invoke([{"role": "user", "content": prompt}])
            generated_text = response["message"].get("content")
            
            sec.text = generated_text
            new_prev_ctx = generated_text[-500:] if generated_text else prev_ctx

            if sec.children:
                populated_children = []
                for child in sec.children:
                    populated_child, new_prev_ctx = await process_section(child, new_prev_ctx)
                    populated_children.append(populated_child)
                sec.children = populated_children

            return sec, new_prev_ctx

        for index in range(total_sections):
            section = await self.skeleton_repository.get_section_model(chat_id, index)
            if not section:
                continue
                
            populated_section, previous_context = await process_section(section, previous_context)
            await self.skeleton_repository.update_section(chat_id, index, populated_section)
        
            combined_top_level_text.append(f"**{populated_section.title}**\n{populated_section.text}")

        return "\n\n".join(combined_top_level_text)

    async def _regenerate_section(
        self,
        chat_id: str,
        section_title: str,
        comments: str
    ) -> str:
        total_sections = await self.skeleton_repository.get_total_sections_count(chat_id)
        
        target_root_index = -1
        target_section = None
        root_section = None

        def find_in_tree(node: Section, target: str) -> Optional[Section]:
            if node.title == target:
                return node
            for child in node.children:
                found = find_in_tree(child, target)
                if found:
                    return found
            return None

        for index in range(total_sections):
            root_section = await self.skeleton_repository.get_section_model(chat_id, index)
            if not root_section:
                continue
                
            target_section = find_in_tree(root_section, section_title)
            if target_section:
                target_root_index = index
                break
                
        if target_root_index == -1 or not target_section:
            logger.error(f"Раздел '{section_title}' не найден в скелете.")
            return ""

        logger.info(f"Регенерация раздела: {section_title}")

        rag_context = await self.rag.search_with_boosting(comments, "", section_title)

        prompt = REGENERATING_PROMPT.format(
            section_title=section_title,
            old_text=target_section.text,
            comments=comments,
            rag_context=rag_context or "Новых данных не найдено."
        )

        response = await self.llm.invoke([{"role": "user", "content": prompt}])
        new_text = response.get("message", {}).get("content", "")

        target_section.text = new_text

        await self.skeleton_repository.update_section(chat_id, target_root_index, root_section)
        
        return new_text

    async def _export_and_upload_document(
        self,
        chat_id: str,
        doc_type: str
    ) -> Optional[AiAttachment]:
        logger.info(f"Сборка DOCX для чата {chat_id}, тип {doc_type}")
        
        buffer = await self.doc_exporter.build_docx_iteratively(chat_id, self.skeleton_repository)
        
        buffer.seek(0, 2)
        size = buffer.tell()
        buffer.seek(0)
        
        file_name = f"{doc_type}_{chat_id}_{int(datetime.now().timestamp())}.docx"
        object_key = f"documents/{chat_id}/{file_name}"
        
        uploaded_key = await self.s3_client.upload_fileobj(buffer, object_key)
        if not uploaded_key:
            logger.error("Не удалось загрузить документ в S3")
            return None
        
        return AiAttachment(
            file_name=file_name,
            s3_key=uploaded_key,
            size=size,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )