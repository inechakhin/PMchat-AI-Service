from datetime import datetime
from typing import Optional, Dict, Any

from entities.skeleton import Skeleton, SkeletonMetadata, Section, SkeletonState

class SkeletonRepository:
    
    async def exist_by_chat_id(
        self,
        chat_id: str,
    ) -> bool:
        return await Skeleton.find(
            Skeleton.chat_id == chat_id
        ).exists()

    async def create_empty(
        self,
        chat_id: str,
        state: SkeletonState,
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
    ) -> Optional[SkeletonMetadata]:
        return await Skeleton.find_one(
            Skeleton.chat_id == chat_id
        ).project(SkeletonMetadata)
    
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
        results = await Skeleton.aggregate([
            {"$match": {"chat_id": chat_id}},
            {"$project": {
                "_id": 0, 
                "target_section": {"$arrayElemAt": ["$document", section_index]}
            }},
        ]).to_list()
        if not results or "target_section" not in results[0] or results[0]["target_section"] is None:
            return None
        return Section.model_validate(results[0]["target_section"])
