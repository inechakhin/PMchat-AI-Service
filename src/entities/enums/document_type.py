from enum import StrEnum

class DocumentType(StrEnum):
    TECHNICAL_SPEC = "Техническое задание на разработку ИС"
    IMPLEMENT_TECH_SPEC = "Техническое задание на внедрение ИС"
    UNKNOWN = "Неизвестный тип"
