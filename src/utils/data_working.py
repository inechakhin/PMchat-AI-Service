from pathlib import Path
from typing import List, AsyncGenerator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from entities.enums.document_type import DocumentType
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


async def get_files_by_ext(
    folder: Path, 
    extensions: List = [".pdf", ".docx"],
) -> AsyncGenerator[Path, None]:
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder}")    
    if not folder.is_dir():
        raise NotADirectoryError(f"Указанный путь не является папкой: {folder}")
    
    for ext in extensions:
        clean_ext = ext[1:] if ext.startswith('.') else ext
        pattern = f"*.{clean_ext}"
        for file_path in folder.glob(pattern):
            yield file_path


def get_document_type(file_path: Path) -> str:    
    filename_lower = file_path.name.lower()
    if filename_lower.startswith("техническое задание на разработку"):
        return DocumentType.TECHNICAL_SPEC.value
    elif filename_lower.startswith("техническое задание на внедрение"):
        return DocumentType.IMPLEMENT_TECH_SPEC.value
    return DocumentType.UNKNOWN.value