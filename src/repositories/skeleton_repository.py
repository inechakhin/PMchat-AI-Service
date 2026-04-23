from datetime import datetime
from typing import Optional, Dict, Any

from entities.skeleton import Skeleton, Section, ChatState

class SkeletonRepository:

    async def create_empty(
        self,
        chat_id: str,
        state: ChatState,
    ) -> None:
        skeleton = Skeleton(
            chat_id=chat_id,
            state=state,
        )
        return await skeleton.insert()
    
    async def update(
        self,
        chat_id: str,
        update_data: Dict[str, Any]
    ) -> None:
        if not update_data:
            return
        update_fields = {**update_data, "updated_at": datetime.now()}
        await Skeleton.find_one(
            Skeleton.chat_id == chat_id
        ).update({
            "$set": update_fields
        })
        
    async def update_section(
        self,
        chat_id: str,
        section_index: int,
        section: Section
    ) -> None:
        await Skeleton.find_one(
            Skeleton.chat_id == chat_id
        ).update({
            "$set": {
                f"document.{section_index}": section.model_dump(exclude_none=True),
                "updated_at": datetime.now()
            }
        })
    
    async def add_section(
        self,
        chat_id: str,
        section: Section
    ) -> None:
        await Skeleton.find_one(
            Skeleton.chat_id == chat_id
        ).update({
            "$push": {"document": section.model_dump()},
            "$set": {"updated_at": datetime.now()},
        })

    async def get_metadata_by_chat_id(
        self,
        chat_id: str
    ) -> Optional[Dict[str, Any]]:
        skeleton = await Skeleton.find_one(
            Skeleton.chat_id == chat_id
        ).project({
            "document": 0,
            "_id": 0,
        })
        if skeleton is None:
            return None
        return skeleton.model_dump()
    
    async def get_total_sections_count(
        self,
        chat_id: str
    ) -> int:
        res = await Skeleton.find(
            Skeleton.chat_id == chat_id
        ).aggregate([
            {"$project": {"count": {"$size": "$document"}}}
        ]).to_list()
        return res[0].get("count", 0) if res else 0

    async def get_section_model(
        self,
        chat_id: str,
        section_index: int
    ) -> Optional[Section]:
        skeleton = await Skeleton.find_one(
            Skeleton.chat_id == chat_id
        ).project({
            "document": {"$slice": [section_index, 1]}
        })
        if skeleton and skeleton.document:
            return skeleton.document[0]
        return None
