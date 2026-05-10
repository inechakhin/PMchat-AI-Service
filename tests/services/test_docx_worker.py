import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from services.docx_worker import DocxWorker
from schemas.docx import HeaderMeta, ChunkMeta

class MockPara:
    def __init__(self, text: str, style_name: str):
        self.text = text
        self.style = MagicMock()
        self.style.name = style_name

@pytest.mark.parametrize("style_name, expected", [
    ("Heading 1", 1),
    ("Heading 2", 2),
    ("Heading 10", 10),
    ("Normal", None),
    ("List Bullet", None),
    ("", None),
    ("Heading", 1),
])
def test_get_heading_level(style_name, expected):
    worker = DocxWorker()
    para = MockPara("some text", style_name)
    assert worker._get_heading_level(para) == expected

@pytest.mark.asyncio
async def test_process_document():
    worker = DocxWorker()
    worker.chunker = MagicMock()
    worker.chunker.split_text.side_effect = lambda text: [text]

    mock_doc = MagicMock()
    mock_doc.paragraphs = [
        MockPara("Глава 1", "Heading 1"),
        MockPara("Абзац 1.", "Normal"),
        MockPara("Абзац 2.", "Normal"),
        MockPara("Глава 1.1", "Heading 2"),
        MockPara("Абзац 1.1.", "Normal"),
        MockPara("", "Normal"),
        MockPara("   ", "Normal"),
        MockPara("Глава 2", "Heading 1"),
        MockPara("Абзац 2.1.", "Normal"),
    ]

    loop = MagicMock()
    loop.run_in_executor = AsyncMock(side_effect=lambda executor, func, *args: func())

    with patch("services.docx_worker.Document", return_value=mock_doc), \
         patch("services.docx_worker.get_document_type", return_value="docx"), \
         patch("asyncio.get_event_loop", return_value=loop):
        results = [item async for item in worker.process_document(Path("test.docx"))]

    assert len(results) == 6

    h1 = results[0]
    assert isinstance(h1, HeaderMeta)
    assert h1.title == "Глава 1"
    assert h1.section_level == 1
    assert h1.parent_titles == []

    c1 = results[1]
    assert isinstance(c1, ChunkMeta)
    assert c1.text == "Абзац 1.\nАбзац 2."
    assert c1.heading_titles == ["Глава 1"]

    h2 = results[2]
    assert isinstance(h2, HeaderMeta)
    assert h2.title == "Глава 1.1"
    assert h2.section_level == 2
    assert h2.parent_titles == ["Глава 1"]

    c2 = results[3]
    assert isinstance(c2, ChunkMeta)
    assert c2.text == "Абзац 1.1."
    assert c2.heading_titles == ["Глава 1", "Глава 1.1"]

    h3 = results[4]
    assert isinstance(h3, HeaderMeta)
    assert h3.title == "Глава 2"
    assert h3.section_level == 1
    assert h3.parent_titles == []

    c3 = results[5]
    assert isinstance(c3, ChunkMeta)
    assert c3.text == "Абзац 2.1."
    assert c3.heading_titles == ["Глава 2"]

@pytest.mark.asyncio
async def test_process_template_document():
    worker = DocxWorker()
    mock_doc = MagicMock()
    mock_doc.paragraphs = [
        MockPara("Раздел 1", "Heading 1"),
        MockPara("Текст раздела 1.", "Normal"),
        MockPara("Подраздел 1.1", "Heading 2"),
        MockPara("Текст 1.1.", "Normal"),
        MockPara("Раздел 2", "Heading 1"),
        MockPara("Текст раздела 2.", "Normal"),
    ]

    loop = MagicMock()
    loop.run_in_executor = AsyncMock(side_effect=lambda executor, func, *args: func())

    with patch("services.docx_worker.Document", return_value=mock_doc), \
         patch("asyncio.get_event_loop", return_value=loop):
        sections = [sec async for sec in worker.process_template_document(Path("template.docx"))]

    assert len(sections) == 2
    root1 = sections[0]
    assert root1.title == "Раздел 1"
    assert root1.text == "Текст раздела 1."
    assert len(root1.children) == 1
    child = root1.children[0]
    assert child.title == "Подраздел 1.1"
    assert child.text == "Текст 1.1."

    root2 = sections[1]
    assert root2.title == "Раздел 2"
    assert root2.text == "Текст раздела 2."
    assert root2.children == []

@pytest.mark.asyncio
async def test_process_template_no_headings():
    worker = DocxWorker()
    mock_doc = MagicMock()
    mock_doc.paragraphs = [
        MockPara("Просто текст без стилей.", "Normal"),
        MockPara("Ещё текст.", "Normal"),
    ]

    loop = MagicMock()
    loop.run_in_executor = AsyncMock(side_effect=lambda executor, func, *args: func())

    with patch("services.docx_worker.Document", return_value=mock_doc), \
         patch("asyncio.get_event_loop", return_value=loop):
        sections = [sec async for sec in worker.process_template_document(Path("nohead.docx"))]

    assert len(sections) == 1
    assert sections[0].title == ""
    assert sections[0].text == "Просто текст без стилей.\n\nЕщё текст."