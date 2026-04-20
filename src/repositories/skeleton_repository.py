from datetime import datetime

from entities.skeleton import Skeleton, DocumentType, Section

class SkeletonRepository:

    async def exist_by_chat_id(
        self,
        chat_id: str
    ) -> bool:
        count = await Skeleton.find(
            Skeleton.chat_id == chat_id
        ).count()
        return count > 0

    async def create_empty(
        self,
        chat_id: str,
        doc_type: DocumentType,
    ) -> None:
        skeleton = Skeleton(chat_id=chat_id, type=doc_type)
        return await skeleton.insert()

    async def update_requirements(
        self,
        chat_id: str,
        requirements: str,
    ) -> None:
        await Skeleton.find_one(
            Skeleton.chat_id == chat_id
        ).update({
            "$set": {
                "requirements": requirements,
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
            "$set": {"updated_at": datetime.now()}
        })
