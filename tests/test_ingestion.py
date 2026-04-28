"""Tests for the ingestion layer: parser, chunker, embedder, crud."""

from __future__ import annotations

import pytest

from app.db import crud
from app.db.session import async_session_maker
from app.ingestion.chunker import (
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_SIZE,
    hierarchical_chunk,
)
from app.ingestion.embedder import embed_chunks
from app.ingestion.parser import PageText, parse_pdf
from tests.conftest import FIXTURE_PDF

# ── Pure helpers ──────────────────────────────────────────────────


def test_compute_sha256_is_deterministic_and_correct_length():
    a = crud.compute_sha256(b"hello world")
    b = crud.compute_sha256(b"hello world")
    assert a == b
    assert len(a) == 64
    # different content -> different hash
    assert crud.compute_sha256(b"hello world!") != a


# ── Parser ────────────────────────────────────────────────────────


@pytest.mark.slow
def test_parse_pdf_returns_page_texts_with_content():
    """Docling round-trip on the fixture PDF. Marked slow because
    the layout model loads on first call."""
    pages = parse_pdf(FIXTURE_PDF)
    assert len(pages) >= 1, "expected at least one page"
    assert all(isinstance(p, PageText) for p in pages)
    # The page numbering must start at 1 and be ascending.
    page_nums = [p.page_num for p in pages]
    assert page_nums == sorted(page_nums)
    assert page_nums[0] >= 1
    # The fixture PDF is page 9 of an arxiv paper — should have meaningful text.
    total_chars = sum(len(p.text) for p in pages)
    assert total_chars > 500, f"expected substantial text content, got {total_chars} chars"


# ── Chunker ───────────────────────────────────────────────────────


@pytest.mark.slow
def test_hierarchical_chunker_produces_linked_parents_and_children():
    """Every child must reference a parent that actually exists.
    This is the contract `parent_fetch` and the agent depend on."""
    pages = parse_pdf(FIXTURE_PDF)
    parents, children = hierarchical_chunk(pages)

    assert parents, "expected at least one parent chunk"
    assert children, "expected at least one child chunk"

    # Linkage: every child.parent_id matches some parent.parent_id
    parent_ids = {p.parent_id for p in parents}
    orphans = [c for c in children if c.parent_id not in parent_ids]
    assert not orphans, (
        f"{len(orphans)}/{len(children)} children have no matching parent — "
        "linkage broken; agent.parent_fetch will silently lose context"
    )

    # IDs must be unique within their level.
    assert len({p.parent_id for p in parents}) == len(parents), "parent_id collision"
    assert len({c.chunk_id for c in children}) == len(children), "chunk_id collision"

    # Page propagation: every chunk has a sane page_num
    assert all(p.page_num >= 1 for p in parents)
    assert all(c.page_num >= 1 for c in children)


def test_chunker_constants_match_artifact():
    """The artifact specifies 1024-token parents / 256-token children."""
    assert PARENT_CHUNK_SIZE == 1024
    assert CHILD_CHUNK_SIZE == 256


# ── Embedder ──────────────────────────────────────────────────────


@pytest.mark.integration
async def test_embed_chunks_returns_correct_dims():
    """Single round-trip via OpenRouter for embedding + local FastEmbed BM25.
    Doesn't re-parse the PDF — uses synthetic chunks to keep it cheap."""
    from app.core.config import settings
    from app.ingestion.chunker import ChildNode

    children = [
        ChildNode(
            chunk_id="test-chunk-1",
            parent_id="test-parent-1",
            text="The quick brown fox jumps over the lazy dog.",
            page_num=1,
            chunk_index=0,
        ),
        ChildNode(
            chunk_id="test-chunk-2",
            parent_id="test-parent-1",
            text="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            page_num=1,
            chunk_index=1,
        ),
    ]

    embedded = await embed_chunks(children)

    assert len(embedded) == len(children)
    for e, original in zip(embedded, children, strict=True):
        assert e.child is original, "embed_chunks must preserve order"
        assert len(e.dense) == settings.EMBEDDING_DIMS
        assert all(isinstance(v, float) for v in e.dense[:8])
        assert len(e.sparse_indices) == len(e.sparse_values) > 0
        assert all(isinstance(i, int) for i in e.sparse_indices[:8])


# ── CRUD round-trip ───────────────────────────────────────────────


@pytest.mark.integration
async def test_dedup_check_finds_existing_document(ingested_doc):
    """The fixture document was just inserted — dedup should hit on its hash."""
    async with async_session_maker() as session:
        existing = await crud.dedup_check(session, ingested_doc["content_hash"])
    assert existing is not None
    assert str(existing.id) == ingested_doc["document_id"]
    assert existing.tenant_id == ingested_doc["tenant_id"]
    assert existing.is_active is True


@pytest.mark.integration
async def test_dedup_check_misses_for_unknown_hash():
    async with async_session_maker() as session:
        result = await crud.dedup_check(session, "0" * 64)
    assert result is None


@pytest.mark.integration
async def test_mark_inactive_flips_is_active(scratch_doc_id):
    """mark_inactive should soft-delete both the document and its parent_chunks
    (cascade is handled in crud, not at the SQL level for parent_chunks)."""
    import uuid as _uuid

    async with async_session_maker() as session:
        await crud.mark_inactive(session, _uuid.UUID(scratch_doc_id))
        await session.commit()

    async with async_session_maker() as session:
        from sqlalchemy import select

        from app.db.models import Document

        stmt = select(Document).where(Document.id == _uuid.UUID(scratch_doc_id))
        doc = (await session.execute(stmt)).scalar_one()
        assert doc.is_active is False
