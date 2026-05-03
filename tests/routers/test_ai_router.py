import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from unittest.mock import MagicMock

from routers.ai_router import ai_router
from dependencies.ai import get_ai_service
from schemas.ai import AiResponse

@pytest.fixture
def mock_ai_service():
    return MagicMock()

@pytest.fixture
def test_app(mock_ai_service):
    app = FastAPI()
    app.include_router(ai_router)

    async def override_get_ai_service():
        return mock_ai_service

    app.dependency_overrides[get_ai_service] = override_get_ai_service
    return app

@pytest.fixture
async def client(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test"
    ) as ac:
        yield ac

@pytest.mark.asyncio
async def test_generate_stream_success(client, mock_ai_service):
    request_payload = {
        "chat_id": "123",
        "chat_type": "communication",
        "messages": [{"sender_type": "user", "text": "Hello"}]
    }

    async def success_stream(_):
        yield AiResponse(token="Hello").model_dump_json() + "\n"
        yield AiResponse(token=" world").model_dump_json() + "\n"
        yield AiResponse(is_end=True).model_dump_json() + "\n"

    mock_ai_service.generate_response.side_effect = success_stream

    response = await client.post(
        "/internal/api/ai/generate-stream",
        json=request_payload
    )

    assert response.status_code == 200

    body = ""
    async for chunk in response.aiter_text():
        body += chunk

    assert "Hello" in body
    assert '"is_end":true' in body

@pytest.mark.asyncio
async def test_generate_stream_internal_error_safe(client, mock_ai_service):
    request_payload = {
        "chat_id": "123",
        "chat_type": "communication",
        "messages": []
    }

    async def error_stream(_):
        yield AiResponse(token="start").model_dump_json() + "\n"
        raise Exception("Simulated failure")

    mock_ai_service.generate_response.side_effect = error_stream

    response = await client.post(
        "/internal/api/ai/generate-stream",
        json=request_payload
    )

    assert response.status_code == 200

    chunks = []
    async for chunk in response.aiter_text():
        chunks.append(chunk)

    full_body = "".join(chunks)

    assert "start" in full_body

    assert '"is_error":true' in full_body
    assert "Simulated failure" in full_body