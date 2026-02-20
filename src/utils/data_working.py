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

"""
def text_split_with_overlap(
    text: str,
    max_chunk_size: int,
    overlap_paragraphs: int,
) -> List[str]:
    
    paragraphs = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]
    
    if not paragraphs:
        return []
    
    chunks = []
    current_chunk = ""
    current_chunk_size = 0

    i = 0
    while i < len(paragraphs):
        paragraph = paragraphs[i]
        paragraph_len = len(paragraph)

        test_chunk_size = current_chunk_size + paragraph_len

        if test_chunk_size <= max_chunk_size:
            current_chunk = (current_chunk + " " + paragraph).strip()
            current_chunk_size = test_chunk_size
            i += 1
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Начинаем новый чанк с перекрытием
            overlap_start = max(0, i - overlap_paragraphs)
            current_chunk = " ".join(paragraphs[overlap_start:i])
            current_chunk_size = len(current_chunk)

    if current_chunk:
        chunks.append(current_chunk.strip())
"""
