import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path
import json
import numpy as np

from services.template_service import TemplateService
from entities.enums.document_type import DocumentType
from entities.section import Section
from repositories.template_repository import TemplateRepository
from core.config import settings

@pytest.fixture
def mock_template_repo():
    repo = AsyncMock(spec=TemplateRepository)
    repo.exist_by_type = AsyncMock()
    repo.delete_by_type = AsyncMock()
    repo.create_empty = AsyncMock()
    repo.add_section = AsyncMock()
    repo.get_total_sections_count = AsyncMock()
    repo.get_section_model = AsyncMock()
    return repo

@pytest.fixture
def mock_docx_worker():
    worker = AsyncMock()
    worker.process_template_document = MagicMock()
    return worker

@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.query = AsyncMock()
    return store

@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.invoke = AsyncMock()
    return llm

@pytest.fixture
def template_service(mock_template_repo, mock_docx_worker, mock_vector_store, mock_llm):
    return TemplateService(
        template_repository=mock_template_repo,
        docx_worker=mock_docx_worker,
        vector_store=mock_vector_store,
        llm=mock_llm,
    )

def make_section(title="", text="", children=None):
    return Section(title=title, text=text, children=children or [])

async def async_gen(items):
    for item in items:
        yield item

@pytest.mark.asyncio
async def test_init_template_store_hard_init(template_service, mock_template_repo):
    with patch.object(template_service, "_add_templates_from_files", AsyncMock()) as mock_add, \
         patch.object(template_service, "_build_template", AsyncMock()) as mock_build:
        
        calls = 0
        def exist_side_effect(doc_type):
            nonlocal calls
            calls += 1
            if calls <= len(DocumentType):
                return doc_type == DocumentType.TECHNICAL_SPEC
            return doc_type == DocumentType.IMPLEMENT_TECH_SPEC
        mock_template_repo.exist_by_type.side_effect = exist_side_effect

        await template_service.init_template_store(hard_init=True)

        mock_template_repo.delete_by_type.assert_awaited_once_with(DocumentType.TECHNICAL_SPEC)
        mock_add.assert_awaited_once_with(settings.TEMPLATES_DIR)
        mock_build.assert_awaited_once_with(DocumentType.TECHNICAL_SPEC)

@pytest.mark.asyncio
async def test_init_template_store_no_hard_init(template_service, mock_template_repo):
    with patch.object(template_service, "_add_templates_from_files", AsyncMock()) as mock_add, \
         patch.object(template_service, "_build_template", AsyncMock()) as mock_build:
        
        def exist_side_effect(doc_type):
            if doc_type == DocumentType.UNKNOWN:
                return False
            if doc_type == DocumentType.TECHNICAL_SPEC:
                return False
            return True
        mock_template_repo.exist_by_type.side_effect = exist_side_effect

        await template_service.init_template_store(hard_init=False)

        mock_template_repo.delete_by_type.assert_not_awaited()
        mock_add.assert_awaited_once_with(settings.TEMPLATES_DIR)
        mock_build.assert_awaited_once_with(DocumentType.TECHNICAL_SPEC)

@pytest.mark.asyncio
async def test_get_template_returns_sections(template_service, mock_template_repo):
    doc_type = DocumentType.TECHNICAL_SPEC
    s1 = make_section("Header", "Content")
    s2 = make_section("Footer", "More content")
    mock_template_repo.get_total_sections_count.return_value = 2
    mock_template_repo.get_section_model.side_effect = [s1, s2]

    sections = [section async for section in template_service.get_template(doc_type)]

    assert len(sections) == 2
    assert sections[0] == s1
    assert sections[1] == s2
    mock_template_repo.get_total_sections_count.assert_awaited_once_with(doc_type)
    mock_template_repo.get_section_model.assert_has_awaits([call(doc_type, 0), call(doc_type, 1)])

@pytest.mark.asyncio
async def test_get_template_empty(template_service, mock_template_repo):
    mock_template_repo.get_total_sections_count.return_value = 0
    sections = [s async for s in template_service.get_template(DocumentType.IMPLEMENT_TECH_SPEC)]
    assert len(sections) == 0
    mock_template_repo.get_section_model.assert_not_awaited()

@pytest.mark.asyncio
async def test_generate_template(template_service, mock_llm):
    response_content = json.dumps([
        {"title": "Введение", "text": "Текст1", "children": []},
        {"title": "Основная часть", "text": "Текст2", "children": []}
    ])
    mock_llm.invoke.return_value = {"message": {"content": response_content}}
    
    sections = [s async for s in template_service.generate_template("Договор", "Требования безопасности")]
    
    assert len(sections) == 2
    assert sections[0].title == "Введение"
    assert sections[1].title == "Основная часть"
    call_args = mock_llm.invoke.await_args[0][0]
    assert isinstance(call_args, list)
    assert call_args[0]["role"] == "system"
    assert "Договор" in call_args[0]["content"]
    assert "Требования безопасности" in call_args[0]["content"]

@pytest.mark.asyncio
async def test_add_templates_from_files_new(template_service, mock_template_repo, mock_docx_worker):
    doc_type = DocumentType.TECHNICAL_SPEC
    section_root = make_section("Корень", "", [
        make_section("Дочерняя", "текст")
    ])
    
    with patch("services.template_service.get_files_by_ext") as mock_get_files, \
         patch("services.template_service.get_document_type", return_value=doc_type):

        file_path = Path("/templates/tech_spec.docx")
        mock_get_files.side_effect = lambda *a, **kw: async_gen([file_path])
        mock_template_repo.exist_by_type.return_value = False
        mock_docx_worker.process_template_document.return_value = async_gen([section_root])

        await template_service._add_templates_from_files(Path("/fake"))

        mock_template_repo.exist_by_type.assert_awaited_once_with(doc_type)
        mock_template_repo.create_empty.assert_awaited_once_with(doc_type)
        mock_template_repo.add_section.assert_awaited_once_with(doc_type, section_root)

@pytest.mark.asyncio
async def test_add_templates_from_files_existing_skipped(template_service, mock_template_repo, mock_docx_worker):
    doc_type = DocumentType.IMPLEMENT_TECH_SPEC
    with patch("services.template_service.get_files_by_ext") as mock_get_files, \
         patch("services.template_service.get_document_type", return_value=doc_type):
        file_path = Path("/templates/impl.docx")
        mock_get_files.side_effect = lambda *a, **kw: async_gen([file_path])
        mock_template_repo.exist_by_type.return_value = True

        await template_service._add_templates_from_files(Path("/fake"))

        mock_template_repo.create_empty.assert_not_awaited()
        mock_docx_worker.process_template_document.assert_not_called()

@pytest.mark.asyncio
async def test_build_template(template_service, mock_vector_store, mock_template_repo):
    doc_type = DocumentType.TECHNICAL_SPEC
    query_result = MagicMock()
    query_result.texts = ["Введение", "Технические требования", "Заключение"]
    query_result.vectors = [
        [0.1, 0.2], [0.15, 0.25], [0.9, 0.8]
    ]
    query_result.metadatas = [
        {"section_level": 1}, {"section_level": 2}, {"section_level": 1}
    ]
    mock_vector_store.query.return_value = query_result

    with patch("services.template_service.cosine_distances", return_value=np.array([[0,0.1,0.9],[0.1,0,0.8],[0.9,0.8,0]])), \
         patch("services.template_service.AgglomerativeClustering") as mock_clustering, \
         patch.object(template_service, "_process_clusters_with_llm", AsyncMock()) as mock_process:
        
        cluster_instance = MagicMock()
        cluster_instance.fit_predict.return_value = [0, 0, 1]
        mock_clustering.return_value = cluster_instance

        final_sections = [make_section("Введение")]
        mock_process.return_value = final_sections

        await template_service._build_template(doc_type)

        mock_vector_store.query.assert_awaited_once()
        mock_clustering.assert_called_once_with(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=settings.DISTANCE_THRESHOLD
        )
        cluster_instance.fit_predict.assert_called_once()
        expected_clusters = [{"titles": ["Введение", "Технические требования"], "levels": [1,2]}]
        mock_process.assert_awaited_once_with(doc_type.value, expected_clusters)
        mock_template_repo.create_empty.assert_awaited_once_with(doc_type)
        mock_template_repo.add_section.assert_awaited_once_with(doc_type, final_sections[0])

@pytest.mark.asyncio
async def test_build_template_no_data(template_service, mock_vector_store, mock_template_repo):
    mock_vector_store.query.return_value = None
    await template_service._build_template(DocumentType.TECHNICAL_SPEC)
    mock_template_repo.create_empty.assert_not_awaited()

@pytest.mark.asyncio
async def test_process_clusters_with_llm(template_service, mock_llm):
    clusters = [
        {"titles": ["Введение", "Начало"], "levels": [1, 1]},
        {"titles": ["Требования"], "levels": [2]}
    ]
    response_json = json.dumps([
        {"title": "Введение", "text": "", "children": []},
        {"title": "Требования к системе", "text": "", "children": []}
    ])
    mock_llm.invoke.return_value = {"message": {"content": response_json}}

    sections = await template_service._process_clusters_with_llm("ТЗ", clusters)

    assert len(sections) == 2
    invoke_args = mock_llm.invoke.await_args[0][0]
    prompt_text = invoke_args[0]["content"]
    assert "ТЗ" in prompt_text
    assert "Введение" in prompt_text
    assert "Начало" in prompt_text
    assert "Требования" in prompt_text

class TestParseLLMResponse:
    def test_valid_json(self, template_service):
        json_str = json.dumps([{"title": "Test", "text": "Text", "children": []}])
        sections = template_service._parse_llm_response(json_str)
        assert len(sections) == 1
        assert sections[0].title == "Test"

    def test_json_with_markdown_fence(self, template_service):
        json_str = '```json\n[{"title": "A", "text": "B", "children": []}]\n```'
        sections = template_service._parse_llm_response(json_str)
        assert len(sections) == 1
        assert sections[0].title == "A"

    def test_invalid_json_returns_empty(self, template_service):
        sections = template_service._parse_llm_response("not a json")
        assert sections == []

    def test_valid_json_invalid_pydantic(self, template_service):
        json_str = json.dumps([{"title": 123}])
        sections = template_service._parse_llm_response(json_str)
        assert sections == []

    def test_extra_fields_ignored(self, template_service):
        json_str = json.dumps([{"title": "X", "text": "Y", "children": [], "unknown": 1}])
        sections = template_service._parse_llm_response(json_str)
        assert len(sections) == 1
        assert sections[0].title == "X"