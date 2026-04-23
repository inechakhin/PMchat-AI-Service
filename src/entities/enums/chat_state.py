from enum import StrEnum

class ChatState(StrEnum):
    COMMUNICATION = "communication"
    ELICITATION = "elicitation"
    REVISION = "revision"