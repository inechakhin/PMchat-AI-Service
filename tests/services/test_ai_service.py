import json
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock
import asyncio

from services.ai_service import AiService
from entities.enums.document_type import DocumentType
from entities.enums.skeleton_state import SkeletonState
from entities.section import Section
from schemas.ai import AiRequest, AiMessage, AiAttachment

async def async_gen(*items):
    for item in items:
        yield item

async def collect_aiter(aiter):
    items = []
    async for item in aiter:
        items.append(item)
    return items

@pytest.fixture
def mock_skeleton_repo():
    repo = AsyncMock()
    repo.exist_by_chat_id = AsyncMock()
    repo.create_empty = AsyncMock()
    repo.get_metadata_by_chat_id = AsyncMock()
    repo.get_total_sections_count = AsyncMock()
    repo.get_section_model = AsyncMock()
    repo.add_section = AsyncMock()
    repo.update_section = AsyncMock()
    repo.update = AsyncMock()
    return repo

@pytest.fixture
def mock_template_service():
    service = AsyncMock()
    service.get_template = Mock()
    service.generate_template = Mock()
    return service

@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.invoke = AsyncMock()
    llm.stream = Mock()
    return llm

@pytest.fixture
def mock_rag():
    rag = AsyncMock()
    rag.get_relevant_docs = AsyncMock()
    rag.search_with_boosting = AsyncMock()
    return rag

@pytest.fixture
def mock_doc_exporter():
    exporter = AsyncMock()
    exporter.build_docx_iteratively = AsyncMock()
    return exporter

@pytest.fixture
def mock_s3_client():
    s3 = AsyncMock()
    s3.upload_fileobj = AsyncMock()
    return s3

@pytest.fixture
def ai_service(mock_skeleton_repo, mock_template_service, mock_llm, mock_rag,
               mock_doc_exporter, mock_s3_client):
    return AiService(
        skeleton_repository=mock_skeleton_repo,
        template_service=mock_template_service,
        llm=mock_llm,
        rag=mock_rag,
        doc_exporter=mock_doc_exporter,
        s3_client=mock_s3_client,
    )

@pytest.fixture
def base_request():
    return AiRequest(
        chat_id="chat123",
        chat_type="communication",
        messages=[AiMessage(sender_type="user", text="Привет")],
    )

@pytest.mark.asyncio
async def test_generate_response_communication_flow(
    ai_service, mock_skeleton_repo, mock_llm, base_request
):
    mock_skeleton_repo.exist_by_chat_id.return_value = True
    meta_mock = MagicMock()
    meta_mock.state = SkeletonState.COMMUNICATION
    mock_skeleton_repo.get_metadata_by_chat_id.return_value = meta_mock

    mock_llm.invoke.side_effect = [
        {"message": {"content": ""}},
        {"message": {"content": "Заголовок чата"}},
    ]

    mock_llm.stream.side_effect = lambda *args: async_gen("Токен ответа")

    sample_llm_messages = [{"role": "system", "content": "Система"}, {"role": "user", "content": "Привет"}]
    attachment = AiAttachment(file_name="file.docx", s3_key="s3key", size=1234, content_type="docx")
    ai_service._handle_tool_calls = AsyncMock(
        return_value=(sample_llm_messages, [attachment], [])
    )

    responses = await collect_aiter(ai_service.generate_response(base_request))
    parsed = [json.loads(r) for r in responses]

    assert any(p.get("token") == "Токен ответа" for p in parsed), "Нет токена"
    assert any(p.get("attachments") for p in parsed), "Нет вложений"
    assert any(p.get("chat_title") == "Заголовок чата" for p in parsed), "Нет названия чата"
    assert any(p.get("is_end") is True for p in parsed), "Нет is_end"

@pytest.mark.asyncio
async def test_generate_response_elicitation_state(
    ai_service, mock_skeleton_repo, mock_llm, base_request
):
    mock_skeleton_repo.exist_by_chat_id.return_value = True
    meta_mock = MagicMock()
    meta_mock.state = SkeletonState.ELICITATION
    mock_skeleton_repo.get_metadata_by_chat_id.return_value = meta_mock

    mock_llm.invoke.side_effect = [
        {"message": {"content": ""}},
        {"message": {"content": "Уточнение"}},
    ]
    mock_llm.stream.side_effect = lambda *args: async_gen("Elicit token")

    ai_service._handle_tool_calls = AsyncMock(
        return_value=([{"role": "system"}], [], [])
    )

    responses = await collect_aiter(ai_service.generate_response(base_request))
    parsed = [json.loads(r) for r in responses]
    assert any(p.get("token") == "Elicit token" for p in parsed)

    first_invoke = mock_llm.invoke.await_args_list[0]
    tools = first_invoke.kwargs.get("tools")
    assert tools is not None and len(tools) > 0, "Должны быть инструменты"

@pytest.mark.asyncio
async def test_generate_response_revision_state(
    ai_service, mock_skeleton_repo, mock_llm, base_request
):
    mock_skeleton_repo.exist_by_chat_id.return_value = True
    meta_mock = MagicMock()
    meta_mock.state = SkeletonState.REVISION
    meta_mock.get_doc_type.return_value = "Техническое задание на внедрение ИС"
    mock_skeleton_repo.get_metadata_by_chat_id.return_value = meta_mock

    mock_llm.invoke.side_effect = [
        {"message": {"content": ""}},
        {"message": {"content": "Доработка"}},
    ]
    mock_llm.stream.side_effect = lambda *args: async_gen("Revised")

    ai_service._handle_tool_calls = AsyncMock(
        return_value=([{"role": "system"}], [], [])
    )

    responses = await collect_aiter(ai_service.generate_response(base_request))
    parsed = [json.loads(r) for r in responses]
    assert any(p.get("token") == "Revised" for p in parsed), "Нет токена"

    first_invoke = mock_llm.invoke.await_args_list[0]
    system_msg = first_invoke.args[0][0]["content"]
    assert "Техническое задание на внедрение ИС" in system_msg

@pytest.mark.asyncio
async def test_generate_response_skeleton_not_exist_communication(
    ai_service, mock_skeleton_repo, mock_llm, base_request
):
    mock_skeleton_repo.exist_by_chat_id.return_value = False
    meta_mock = MagicMock()
    meta_mock.state = SkeletonState.COMMUNICATION
    mock_skeleton_repo.get_metadata_by_chat_id.return_value = meta_mock
    mock_llm.invoke.side_effect = [{"message": {"content": ""}}]
    mock_llm.stream.side_effect = lambda *args: async_gen("ok")
    ai_service._handle_tool_calls = AsyncMock(return_value=([], [], []))

    await collect_aiter(ai_service.generate_response(base_request))
    mock_skeleton_repo.create_empty.assert_awaited_once_with(
        base_request.chat_id, SkeletonState.COMMUNICATION
    )

@pytest.mark.asyncio
async def test_generate_response_skeleton_not_exist_generation(
    ai_service, mock_skeleton_repo, mock_llm, base_request
):
    base_request.chat_type = "generation"
    mock_skeleton_repo.exist_by_chat_id.return_value = False
    meta_mock = MagicMock()
    meta_mock.state = SkeletonState.ELICITATION
    mock_skeleton_repo.get_metadata_by_chat_id.return_value = meta_mock
    mock_llm.invoke.side_effect = [{"message": {"content": ""}}]
    mock_llm.stream.side_effect = lambda *args: async_gen("ok")
    ai_service._handle_tool_calls = AsyncMock(return_value=([], [], []))

    await collect_aiter(ai_service.generate_response(base_request))
    mock_skeleton_repo.create_empty.assert_awaited_once_with(
        base_request.chat_id, SkeletonState.ELICITATION
    )

@pytest.mark.asyncio
async def test_generate_response_unexpected_chat_type(
    ai_service, mock_skeleton_repo, base_request
):
    base_request.chat_type = "invalid"
    mock_skeleton_repo.exist_by_chat_id.return_value = False
    responses = await collect_aiter(ai_service.generate_response(base_request))
    parsed = [json.loads(r) for r in responses]
    assert any(p.get("is_error") for p in parsed)
    error_msg = next(p["error_message"] for p in parsed if p.get("is_error"))
    assert "Неожиданный тип чата" in error_msg

@pytest.mark.asyncio
async def test_generate_response_cancelled_error(
    ai_service, mock_skeleton_repo, base_request
):
    mock_skeleton_repo.exist_by_chat_id.side_effect = asyncio.CancelledError()
    responses = await collect_aiter(ai_service.generate_response(base_request))
    parsed = [json.loads(r) for r in responses]
    assert len(parsed) == 1
    assert parsed[0]["is_end"] is True

@pytest.mark.asyncio
async def test_generate_response_connection_error(
    ai_service, mock_skeleton_repo, base_request
):
    mock_skeleton_repo.exist_by_chat_id.side_effect = ConnectionError("Ollama down")
    responses = await collect_aiter(ai_service.generate_response(base_request))
    parsed = [json.loads(r) for r in responses]
    assert any(p.get("is_error") for p in parsed)
    error = next(p for p in parsed if p.get("is_error"))
    assert "Ollama недоступна" in error["error_message"]

@pytest.mark.asyncio
async def test_generate_response_general_exception(
    ai_service, mock_skeleton_repo, base_request
):
    mock_skeleton_repo.exist_by_chat_id.side_effect = RuntimeError("Что-то сломалось")
    responses = await collect_aiter(ai_service.generate_response(base_request))
    parsed = [json.loads(r) for r in responses]
    error = next(p for p in parsed if p.get("is_error"))
    assert "Ошибка генерации AI" in error["error_message"]

@pytest.mark.asyncio
async def test_generate_chat_title(ai_service, mock_llm, base_request):
    mock_llm.invoke.return_value = {"message": {"content": "Заголовок чата"}}
    title = await ai_service._generate_chat_title(base_request)
    assert title == "Заголовок чата"

def test_build_llm_messages(ai_service):
    messages = [
        AiMessage(sender_type="user", text="Вопрос"),
        AiMessage(sender_type="assistant", text="Ответ")
    ]
    result = ai_service._build_llm_messages("Системный промпт", messages)
    assert len(result) == 3
    assert result[0] == {"role": "system", "content": "Системный промпт"}
    assert result[1] == {"role": "user", "content": "Вопрос"}
    assert result[2] == {"role": "assistant", "content": "Ответ"}

@pytest.mark.asyncio
async def test_handle_tool_calls_none(ai_service):
    llm_messages = [{"role": "system", "content": "system"}]
    response = {"message": {}}
    new_msgs, atts, srcs = await ai_service._handle_tool_calls("chat", llm_messages, response)
    assert len(new_msgs) == 2
    assert new_msgs[-1]["name"] == "none"
    assert atts == []
    assert srcs == []

@pytest.mark.asyncio
async def test_handle_tool_calls_finalize_requirements(
    ai_service, mock_skeleton_repo, mock_template_service
):
    ai_service._init_document_skeleton = AsyncMock()
    ai_service._generate_document = AsyncMock()
    ai_service._export_and_upload_document = AsyncMock(
        return_value=AiAttachment(file_name="f.docx", s3_key="key", size=100, content_type="docx")
    )

    meta_mock = MagicMock()
    meta_mock.get_doc_type.return_value = DocumentType.TECHNICAL_SPEC
    mock_skeleton_repo.get_metadata_by_chat_id.return_value = meta_mock

    llm_messages = [{"role": "system", "content": "old"}]
    response = {
        "message": {
            "tool_calls": [{
                "function": {
                    "name": "finalize_requirements",
                    "arguments": {"doc_type": "Техническое задание",
                                 "custom_type_name": "", "requirements": "req"}
                }
            }]
        }
    }
    msgs, atts, srcs = await ai_service._handle_tool_calls("chat1", llm_messages, response)
    ai_service._init_document_skeleton.assert_awaited_once()
    ai_service._generate_document.assert_awaited_once()
    ai_service._export_and_upload_document.assert_awaited()
    assert len(atts) == 1
    assert msgs[0]["content"] != "old"
    assert msgs[-1]["name"] == "finalize_requirements"
    mock_skeleton_repo.update.assert_awaited_with("chat1", {"state": SkeletonState.REVISION})

@pytest.mark.asyncio
async def test_handle_tool_calls_finalize_comments(
    ai_service, mock_skeleton_repo
):
    ai_service._regenerate_section = AsyncMock(return_value="новый текст")
    ai_service._export_and_upload_document = AsyncMock(
        return_value=AiAttachment(file_name="f2.docx", s3_key="key2", size=200,
                                 content_type="docx")
    )
    llm_messages = [{"role": "system", "content": "old"}]
    response = {
        "message": {
            "tool_calls": [{
                "function": {
                    "name": "finalize_comments",
                    "arguments": {"section_title": "Введение", "comments": "слишком длинно"}
                }
            }]
        }
    }
    msgs, atts, srcs = await ai_service._handle_tool_calls("chat2", llm_messages, response)
    ai_service._regenerate_section.assert_awaited_once_with("chat2", "Введение", "слишком длинно")
    assert len(atts) == 1
    content = json.loads(msgs[-1]["content"])
    assert content["status"] == "success"

@pytest.mark.asyncio
async def test_handle_tool_calls_search_project_docs(ai_service, mock_rag):
    mock_rag.get_relevant_docs.return_value = ("документы...", [{"title": "doc.pdf"}])
    llm_messages = [{"role": "system", "content": "sys"}]
    response = {
        "message": {
            "tool_calls": [{
                "function": {
                    "name": "search_project_docs",
                    "arguments": {"query": "запрос", "top_k": 3}
                }
            }]
        }
    }
    msgs, atts, srcs = await ai_service._handle_tool_calls("chat3", llm_messages, response)
    mock_rag.get_relevant_docs.assert_awaited_once_with(query="запрос", limit=3)
    assert len(srcs) == 1
    assert srcs[0].title == "doc.pdf"

@pytest.mark.asyncio
async def test_init_document_skeleton_known_type(
    ai_service, mock_skeleton_repo, mock_template_service
):
    mock_template_service.get_template.side_effect = lambda dt: async_gen(Section(title="1"))
    await ai_service._init_document_skeleton("chat", "Техническое задание на разработку ИС", "", "требования")
    mock_skeleton_repo.update.assert_awaited_once()
    update_data = mock_skeleton_repo.update.await_args.args[1]
    assert update_data["type"] == DocumentType.TECHNICAL_SPEC
    assert update_data["requirements"] == "требования"
    mock_template_service.get_template.assert_called_once_with(DocumentType.TECHNICAL_SPEC)
    assert mock_skeleton_repo.add_section.await_count == 1

@pytest.mark.asyncio
async def test_init_document_skeleton_unknown_type(
    ai_service, mock_skeleton_repo, mock_template_service
):
    mock_template_service.generate_template.side_effect = lambda ct, req: async_gen(Section(title="Кастомный раздел"))
    await ai_service._init_document_skeleton("chat", "Свой тип", "МойТип", "требования")
    mock_skeleton_repo.update.assert_awaited_once()
    update_data = mock_skeleton_repo.update.await_args.args[1]
    assert update_data["type"] == DocumentType.UNKNOWN
    assert update_data["custom_type_name"] == "Свой тип"
    mock_template_service.generate_template.assert_called_once_with("Свой тип", "требования")
    assert mock_skeleton_repo.add_section.await_count == 1

@pytest.mark.asyncio
async def test_generate_document(ai_service, mock_skeleton_repo, mock_rag, mock_llm):
    total = 2
    mock_skeleton_repo.get_total_sections_count.return_value = total
    sec1 = Section(title="Раздел 1", text="Требования к разделу 1",
                   children=[Section(title="Подраздел 1.1", text="Требования подраздела")])
    sec2 = Section(title="Раздел 2", text="Требования к разделу 2")
    mock_skeleton_repo.get_section_model.side_effect = [sec1, sec2]
    mock_rag.search_with_boosting.return_value = "релевантный контекст"
    mock_llm.invoke.side_effect = [
        {"message": {"content": "Сгенерированный текст раздела 1"}},
        {"message": {"content": "Сгенерированный текст подраздела 1.1"}},
        {"message": {"content": "Сгенерированный текст раздела 2"}},
    ]
    result = await ai_service._generate_document("chat", "ТЗ", "общие требования")
    assert mock_skeleton_repo.update_section.await_count == 2
    assert "Раздел 1" in result
    assert "Раздел 2" in result

@pytest.mark.asyncio
async def test_regenerate_section_found(
    ai_service, mock_skeleton_repo, mock_rag, mock_llm
):
    root = Section(title="Глава 1", text="Старый текст",
                   children=[Section(title="Целевой раздел", text="Старый текст вложенного")])
    mock_skeleton_repo.get_total_sections_count.return_value = 1
    mock_skeleton_repo.get_section_model.return_value = root
    mock_rag.search_with_boosting.return_value = "контекст"
    mock_llm.invoke.return_value = {"message": {"content": "Новый текст"}}
    new_text = await ai_service._regenerate_section("chat", "Целевой раздел", "комментарии")
    assert new_text == "Новый текст"
    mock_skeleton_repo.update_section.assert_awaited_once_with("chat", 0, root)

@pytest.mark.asyncio
async def test_regenerate_section_not_found(
    ai_service, mock_skeleton_repo
):
    mock_skeleton_repo.get_total_sections_count.return_value = 1
    mock_skeleton_repo.get_section_model.return_value = Section(title="Другое")
    new_text = await ai_service._regenerate_section("chat", "Нет такого", "комментарии")
    assert new_text == ""

@pytest.mark.asyncio
async def test_export_and_upload_document(
    ai_service, mock_doc_exporter, mock_s3_client
):
    buffer_mock = MagicMock()
    buffer_mock.tell.return_value = 500
    mock_doc_exporter.build_docx_iteratively.return_value = buffer_mock
    mock_s3_client.upload_fileobj.return_value = "s3://bucket/key"
    attachment = await ai_service._export_and_upload_document("chat1", "ТЗ")
    assert attachment is not None
    assert attachment.file_name.startswith("ТЗ_chat1_")
    mock_s3_client.upload_fileobj.assert_awaited_once()

@pytest.mark.asyncio
async def test_export_and_upload_document_upload_fails(
    ai_service, mock_doc_exporter, mock_s3_client
):
    buffer_mock = MagicMock()
    mock_doc_exporter.build_docx_iteratively.return_value = buffer_mock
    mock_s3_client.upload_fileobj.return_value = None
    attachment = await ai_service._export_and_upload_document("chat1", "ТЗ")
    assert attachment is None