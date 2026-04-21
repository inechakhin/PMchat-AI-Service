import asyncio
from pathlib import Path
from typing import AsyncGenerator, Union, List

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import DocItemLabel

from entities.section import Section
from schemas.docling import HeaderMeta, ChunkMeta
from utils.data_working import get_document_type
from utils.logging import logger

class DoclingWorker:
    
    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
        self.converter = DocumentConverter(format_options=format_options)
        self.chunker = HybridChunker(
            tokenizer="BAAI/bge-small-en-v1.5",
            max_tokens=512,
            merge_peers=True,
        )

    async def process_document(
        self, 
        file_path: Path
    ) -> AsyncGenerator[Union[HeaderMeta, ChunkMeta], None]:
        logger.info(f"DoclingWorker: начало обработки {file_path.name}")
        
        doc_type = get_document_type(file_path)
        source_file = file_path.name

        loop = asyncio.get_event_loop()
        docling_doc = await loop.run_in_executor(
            None, 
            lambda: self.converter.convert(file_path).document
        )

        heading_stack: List[str] = []
        
        for item, level in docling_doc.iterate_items():
            if item.label == DocItemLabel.SECTION_HEADER:
                header_text = item.text
                
                while len(heading_stack) >= level:
                    heading_stack.pop()
                parent_titles = list(heading_stack)
                heading_stack.append(header_text)
                
                yield HeaderMeta(
                    title=header_text,
                    section_level=level,
                    doc_type=doc_type,
                    parent_titles=parent_titles,
                    source_file=source_file,
                )

        chunks = await loop.run_in_executor(
            None,
            lambda: list(self.chunker.chunk(docling_doc))
        )
        
        for chunk in chunks:
            heading_titles = getattr(chunk.meta, 'headings', [])
            page_numbers = getattr(chunk.meta, 'page_numbers', [])
            
            yield ChunkMeta(
                text=chunk.text,
                doc_type=doc_type,
                heading_titles=heading_titles,
                page_numbers=page_numbers,
                source_file=source_file,
            )
        
        logger.info(f"DoclingWorker: завершена обработка {file_path.name}")
        
    async def process_template_document(
        self, 
        file_path: Path
    ) -> AsyncGenerator[Section, None]:
        logger.info(f"DoclingWorker: начало потокового парсинга {file_path.name}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.converter.convert(file_path))
        docling_doc = result.document

        current_root: Section = None
        stack: list[tuple[int, Section]] = []
        
        min_header_level = None

        for item, level in docling_doc.iterate_items():
            if item.label == DocItemLabel.SECTION_HEADER:
                if min_header_level is None:
                    min_header_level = level

                new_section = Section(title=item.text, text="", children=[])

                if level == min_header_level:
                    if current_root:
                        yield current_root
                    current_root = new_section
                    stack = [(level, current_root)]
                else:
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    if stack:
                        stack[-1][1].children.append(new_section)
                    stack.append((level, new_section))

            elif item.label == DocItemLabel.TEXT:
                current_section = stack[-1][1]
                if current_section.text:
                    current_section.text += "\n\n" + item.text
                else:
                    current_section.text = item.text

        if current_root:
            yield current_root

        logger.info("DoclingWorker: потоковый парсинг завершен")