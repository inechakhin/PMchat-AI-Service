from typing import Optional, List
from datetime import datetime

from entities.template import Template, DocumentType, Section

class TemplateRepository:

    async def exist_by_type(
        self, 
        doc_type: DocumentType,
    ) -> bool:
        count = await Template.find(
            Template.type == doc_type
        ).count()
        return count > 0

    async def create_empty(
        self,
        doc_type: DocumentType,
    ) -> Template:
        template = Template(type=doc_type)
        return await template.insert()

    async def set_requirements(
        self,
        doc_type: DocumentType,
        requirements: str,
    ) -> None:
        await Template.find_one(
            Template.type == doc_type
        ).update({
            "$set": {
                "requirements": requirements,
                "updated_at": datetime.now()
            }
        })

    async def add_section(
        self,
        doc_type: DocumentType,
        section: Section,
    ) -> None:
        await Template.find_one(
            Template.type == doc_type
        ).update({
            "$push": {"sections": section.model_dump()}, 
            "$set": {"updated_at": datetime.now()},
        })

    async def delete_by_type(
        self,
        doc_type: DocumentType
    ) -> None:
        await Template.find(
            Template.type == doc_type
        ).delete()

    async def get_sections_structure(
        self,
        doc_type: DocumentType
    ) -> List[str]:
        template = await Template.find_one(
            Template.type == doc_type
        ).project({"sections.title": 1})
        if not template or not template.sections:
            return []
        return [s.title for s in template.sections]

    async def get_total_sections_count(
        self,
        doc_type: DocumentType
    ) -> int:
        res = await Template.find(
            Template.type == doc_type
        ).aggregate([
            {"$project": {"count": {"$size": "$sections"}}}
        ]).to_list()
        return res[0].get("count", 0) if res else 0

    async def get_template_section(
        self,
        doc_type: DocumentType,
        section_index: int,
    ) -> Optional[dict]:
        template = await Template.find_one(
           Template.type == doc_type
        ).project({
            "sections": {"$slice": [section_index, 1]}
        })
        return template.sections[0] if template and template.sections else None
