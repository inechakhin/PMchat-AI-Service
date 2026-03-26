from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import settings
from utils.logging import logger


def read_file(file_path: Path) -> str:
    encodings_to_try = ["utf-8", "cp1251", "utf-8-sig"]
    for encoding in encodings_to_try:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError as e:
            logger.warning(f"Не удалось прочитать кеш {file_path} в кодировке {encoding}: {e}.")
            continue


def save_markdown(md_text: str, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(md_text, encoding="utf-8")


def split_text_into_chunks(text: str) -> List[str]:
    #separators = ["\n\n", "\n", ".", "!", "?", ";", " ", ""]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.MAX_CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        #separators=separators,
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_text(text)
    return chunks