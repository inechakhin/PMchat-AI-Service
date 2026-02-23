from pathlib import Path
from typing import List

import fitz
from tabulate import tabulate
import re

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
    

def pdf_to_markdown(file_path: Path) -> str:
    doc = fitz.open(file_path)
    md_lines: list[str] = []

    for _, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")

        for block in blocks:
            text = block[4].strip()
            if not text:
                continue

            # 1. Заголовки вида "1. Назначение"
            if re.match(r"^\d+(\.\d+)*\.\s+[А-ЯA-Z]", text):
                level = text.count(".")
                md_lines.append(f"\n{'#' * (level + 1)} {text}\n")
                continue

            # 2. Таблицы (эвристика: много пробелов / столбцов)
            if is_probable_table(text):
                table = parse_table(text)
                if table:
                    md_lines.append("\n" + table + "\n")
                continue

            # 3. Обычный текст
            md_lines.append(text + "\n")

        md_lines.append("\n---\n")  # разделитель страниц

    return "\n".join(md_lines)


def is_probable_table(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 2:
        return False

    # если в строках есть несколько колонок
    col_counts = [len(re.split(r"\s{2,}", line)) for line in lines]
    return max(col_counts) >= 2


def parse_table(text: str) -> str | None:
    rows = []
    for line in text.splitlines():
        cols = [c.strip() for c in re.split(r"\s{2,}", line) if c.strip()]
        if len(cols) >= 2:
            rows.append(cols)

    if len(rows) < 2:
        return None

    return tabulate(rows, headers="firstrow", tablefmt="github")


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