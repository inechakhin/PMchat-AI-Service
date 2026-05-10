from fastapi import Request

from models.base import ChatBase

def get_llm(request: Request) -> ChatBase:
    return request.app.state.llm

def get_embedder(request: Request) -> ChatBase:
    return request.app.state.embedder