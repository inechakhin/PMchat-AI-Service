import asyncio
from pathlib import Path
from typing import AsyncGenerator, Union, List

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import DocItemLabel

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
