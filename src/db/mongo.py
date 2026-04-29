from typing import Optional
from threading import Lock

from pymongo import AsyncMongoClient
from beanie import init_beanie

from core.config import settings
from entities.template import Template
from entities.skeleton import Skeleton
from utils.logging import logger

class MongoDB:

    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MongoDB, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            logger.info("MongoDB уже проинициалирована")
            return
        
        self.client: Optional[AsyncMongoClient] = None
        self.db = None
        self._initialized = True 
    
    async def connect(
        self, 
        host: str = settings.MONGO_HOST,
        port: int = settings.MONGO_PORT,
        username: str = settings.MONGO_USERNAME,
        password: str = settings.MONGO_PASSWORD,
        database: str = settings.MONGO_DB,
    ):
        connection_string = f"mongodb://{username}:{password}@{host}:{port}/{database}?authSource=admin"
        
        try:
            self.client = AsyncMongoClient(connection_string)
            
            self.db = self.client[database]
            
            await init_beanie(
                database=self.db,
                document_models=[Template, Skeleton]
            )
            
            print(f"MongoDB подключена: {host}:{port}/{database}")
        except Exception as e:
            print(f"Ошибка подключения к MongoDB: {e}")
            raise
    
    async def disconnect(self):
        if self.client:
            await self.client.close()
            self.client = None
            self.db = None
            self._initialized = False
            print("MongoDB отключена")
    
    # Доп. метод
    async def get_stats(self) -> dict:
        db_stats = await self.db.command("dbstats")
        chat_stats = await self.db.command("collstats", "templates")
        message_stats = await self.db.command("collstats", "skeletons")
        
        return {
            "database": {
                "name": db_stats.get("db"),
                "size": db_stats.get("dataSize"),
                "collections": db_stats.get("collections"),
                "objects": db_stats.get("objects"),
            },
            "templates": {
                "count": chat_stats.get("count"),
                "size": chat_stats.get("size"),
                "avg_obj_size": chat_stats.get("avgObjSize"),
            },
            "skeletons": {
                "count": message_stats.get("count"),
                "size": message_stats.get("size"),
                "avg_obj_size": message_stats.get("avgObjSize"),
            }
        }

    # Доп. метод (для тестов)
    async def clear_database(self):
        if self.db:
            collections = await self.db.list_collection_names()
            for collection in collections:
                await self.db[collection].delete_many({})
