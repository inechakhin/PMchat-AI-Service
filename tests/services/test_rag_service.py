import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

from services.rag_service import RagService
from schemas.docx import HeaderMeta, ChunkMeta
from core.config import settings

@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return embedder

@pytest.fixture
def mock_docx_worker():
    return MagicMock()

@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.reset = AsyncMock()
    store.has_collection = AsyncMock()
    store.upsert = AsyncMock()
    store.search = AsyncMock()
    return store

@pytest.fixture
def rag_service(mock_embedder, mock_docx_worker, mock_vector_store):
    return RagService(
        embedder=mock_embedder,
        docx_worker=mock_docx_worker,
        vector_store=mock_vector_store,
    )

def header_meta(title, level=1, parent_titles=None, doc_type="docx", source_file="test.docx"):
    return HeaderMeta(
        title=title,
        section_level=level,
        doc_type=doc_type,
        parent_titles=parent_titles or [],
        source_file=source_file,
    )

def chunk_meta(text, doc_type="docx", heading_titles=None, source_file="test.docx"):
    return ChunkMeta(
        text=text,
        doc_type=doc_type,
        heading_titles=heading_titles or [],
        source_file=source_file,
    )

async def async_gen(items):
    for item in items:
        yield item

async def mock_get_single_file(folder, extensions):
    yield Path("test.docx")

async def mock_get_big_file(folder, extensions):
    yield Path("big.docx")

@pytest.mark.asyncio
async def test_init_vector_store_no_reset(rag_service, mock_vector_store):
    mock_vector_store.has_collection.return_value = True

    await rag_service.init_vector_store(hard_init=False)

    mock_vector_store.reset.assert_not_awaited()
    assert mock_vector_store.has_collection.await_count == 2

@pytest.mark.asyncio
async def test_init_vector_store_hard_reset(rag_service, mock_vector_store):
    mock_vector_store.has_collection.return_value = True

    await rag_service.init_vector_store(hard_init=True)

    mock_vector_store.reset.assert_awaited_once()
    assert mock_vector_store.has_collection.await_count == 2

@pytest.mark.asyncio
async def test_init_vector_store_missing_collection(rag_service, mock_vector_store):
    mock_vector_store.has_collection.side_effect = [False, True]
    
    with patch.object(rag_service, "add_docs_in_vector_store", AsyncMock()) as mock_add_docs:
        await rag_service.init_vector_store(hard_init=False)

    mock_add_docs.assert_awaited_once_with(settings.DOCS_DIR)

@pytest.mark.asyncio
async def test_add_docs_single_file(
    rag_service, mock_docx_worker, mock_vector_store, mock_embedder
):
    h1 = header_meta("H1")
    c1 = chunk_meta("chunk1", heading_titles=["H1"])
    h2 = header_meta("H2", parent_titles=["H1"])
    c2 = chunk_meta("chunk2", heading_titles=["H1", "H2"])

    mock_docx_worker.process_document.side_effect = lambda path: async_gen([h1, c1, h2, c2])

    with patch("services.rag_service.get_files_by_ext", side_effect=mock_get_single_file):
        await rag_service.add_docs_in_vector_store(Path("/fake"))

    assert mock_vector_store.upsert.await_count == 2

    headers_call = mock_vector_store.upsert.await_args_list[0]
    header_batch = headers_call.args[1]
    assert len(header_batch) == 2
    assert header_batch[0].text == "H1"
    assert header_batch[0].metadata["parent_ids"] == []
    assert header_batch[1].text == "H2"
    assert header_batch[1].metadata["parent_ids"] == [header_batch[0].id]

    chunks_call = mock_vector_store.upsert.await_args_list[1]
    chunk_batch = chunks_call.args[1]
    assert len(chunk_batch) == 2
    assert chunk_batch[0].metadata["headings"]["1"] == header_batch[0].id
    assert chunk_batch[1].metadata["headings"]["1"] == header_batch[0].id
    assert chunk_batch[1].metadata["headings"]["2"] == header_batch[1].id

@pytest.mark.asyncio
async def test_add_docs_batch_flush(
    rag_service, mock_docx_worker, mock_vector_store, mock_embedder
):
    total = 25
    headers = [header_meta(f"H{i}") for i in range(total)]
    mock_docx_worker.process_document.side_effect = lambda path: async_gen(headers)

    with patch("services.rag_service.get_files_by_ext", side_effect=mock_get_big_file):
        await rag_service.add_docs_in_vector_store(Path("/fake"))

    calls = mock_vector_store.upsert.await_args_list
    batch_sizes = [len(c.args[1]) for c in calls]
    assert sorted(batch_sizes) == [5, 20]

@pytest.mark.asyncio
async def test_search_with_boosting(rag_service, mock_embedder, mock_vector_store):
    mock_embedder.embed_query.side_effect = AsyncMock(side_effect=[[0.1, 0.2], [0.2, 0.3]])
    chunks_result = MagicMock()
    chunks_result.texts = ["chunk1", "chunk2"]
    chunks_result.metadatas = [
        {"doc_type": "docx", "headings": {"1": "h1_id"}},
        {"doc_type": "pdf", "headings": {}}
    ]
    chunks_result.distances = [0.8, 0.5]
    chunks_result.ids = ["c1", "c2"]
    mock_vector_store.search.side_effect = AsyncMock(side_effect=[chunks_result, AsyncMock(ids=["h1_id"])])

    result = await rag_service.search_with_boosting(
        query="test query", doc_type="docx", header="test header", limit=2
    )

    assert mock_embedder.embed_query.await_count == 2
    mock_vector_store.search.assert_has_awaits([
        call(collection_name=settings.CHUNKS_COLLECTION, vector=[0.1, 0.2], limit=6),
        call(collection_name=settings.HEADERS_COLLECTION, vector=[0.2, 0.3], limit=2),
    ])
    assert "chunk1" in result
    assert "chunk2" in result

@pytest.mark.asyncio
async def test_get_relevant_docs(rag_service, mock_embedder, mock_vector_store):
    mock_embedder.embed_query.return_value = [0.5, 0.5]
    search_result = MagicMock()
    search_result.texts = ["text1", "text2"]
    search_result.metadatas = [
        {"source_file": "doc1.pdf", "author": "A"},
        {"source_file": "doc2.pdf", "author": "B"}
    ]
    search_result.ids = ["id1", "id2"]
    search_result.distances = [0.9, 0.8]
    mock_vector_store.search.return_value = search_result

    formatted, docs = await rag_service.get_relevant_docs("query", limit=2)

    assert "text1" in formatted
    assert "doc1.pdf" in formatted
    assert len(docs) == 2
    assert docs[0]["title"] == "doc1.pdf"
    assert docs[1]["title"] == "doc2.pdf"
    mock_vector_store.search.assert_awaited_once_with(
        collection_name=settings.CHUNKS_COLLECTION,
        vector=[0.5, 0.5],
        limit=2,
    )