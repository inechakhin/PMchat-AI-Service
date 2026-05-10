from fastapi import Request

from db.mongo import MongoDB
from db.qdrant import QdrantDBClient
from db.s3 import S3Client

def get_mongo(request: Request) -> MongoDB:
    return request.app.state.mongo

def get_qdrant(request: Request) -> QdrantDBClient:
    return request.app.state.qdrant

def get_s3(request: Request) -> S3Client:
    return request.app.state.s3