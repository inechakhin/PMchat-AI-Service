import os
from loguru import logger

from core.config import settings

logger.remove()

logger.add(
    os.path.join(settings.LOGS_DIR, "rag_component.log"),
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}"
)