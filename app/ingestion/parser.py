"""Docling PDF parser.

Per-page text extraction with tables rendered as markdown, OCR fallback for
scanned pages handled by Docling natively. Returns a list of `PageText` so
downstream chunking can preserve `page_num` for citations.

Docling lazily downloads layout / OCR models on first run (~hundreds of MB).
The DocumentConverter is constructed once per worker process.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TableItem

from app.core.config import settings


@dataclass
class PageText:
    page_num: int  # 1-indexed
    text: str  # plaintext + inline tables-as-markdown


_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        pdf_opts = PdfPipelineOptions(do_ocr=settings.DOCLING_OCR_ENABLED)
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            }
        )
    return _converter


def parse_pdf(file_path: str | Path) -> list[PageText]:
    """Convert a PDF (or any Docling-supported format) to per-page text."""
    result = _get_converter().convert(str(file_path))
    doc = result.document

    # Bucket every text/table item by its first provenance page.
    pages: dict[int, list[str]] = {}
    for item, _level in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue
        page_no = item.prov[0].page_no
        if isinstance(item, TableItem):
            text = item.export_to_markdown(doc=doc)
        else:
            text = getattr(item, "text", None)
        if text:
            pages.setdefault(page_no, []).append(text)

    return [
        PageText(page_num=p, text="\n\n".join(parts))
        for p, parts in sorted(pages.items())
    ]
