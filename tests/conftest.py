"""Shared pytest fixtures.

These tests run against the *live* dev stack — they assume `docker compose
up -d` has brought up postgres + redis + qdrant, and that the test tenant
has been seeded (`make seed`). They also need OPENROUTER_API_KEY for any
test that exercises embeddings or chat models. Missing prerequisites skip
the whole suite with a clear reason.

Tests target internal Python modules directly (no FastAPI HTTP, no Logto
round-trip, no Celery dispatch) so they stay fast and self-contained.

Lifecycle: a session-scoped `ingested_doc` fixture idempotently inserts
the fixture PDF for the test tenant on first use and leaves it in place
on exit, so re-runs are fast (dedup hit). Tests that mutate independent
data use `scratch_doc_id` which is cleaned up.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.qdrant_client import collection_name_for, get_qdrant_client
from app.db import crud
from app.db.session import async_session_maker
from app.ingestion.chunker import hierarchical_chunk
from app.ingestion.embedder import embed_chunks
from app.ingestion.parser import parse_pdf
from app.ingestion.upserter import build_point, upsert_to_qdrant

FIXTURE_PDF = Path("data/uploads/sample.pdf")
TEST_TENANT_ID = "jzi706s5503k"  # the seeded local-dev tenant


# ── Live-stack reachability gate ──────────────────────────────────


def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(items):
    """If the live stack isn't reachable, skip the whole suite up-front.
    Cheap socket checks beat letting individual tests blow up with cryptic
    connection errors."""
    missing: list[str] = []
    if not _tcp_open("localhost", 5433):
        missing.append("postgres @ localhost:5433")
    if not _tcp_open("localhost", 6379):
        missing.append("redis @ localhost:6379")
    if not _tcp_open("localhost", 6333):
        missing.append("qdrant @ localhost:6333")
    if not settings.OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY (in .env)")
    if not FIXTURE_PDF.exists():
        missing.append(f"fixture PDF at {FIXTURE_PDF}")

    if missing:
        skip = pytest.mark.skip(
            reason=f"live stack / config missing: {', '.join(missing)}"
        )
        for item in items:
            item.add_marker(skip)


# ── Fresh engine per test ─────────────────────────────────────────
# pytest-asyncio creates a new event loop per test (default). The app's
# module-level SQLAlchemy engine pools asyncpg connections, and a
# connection opened on test N's loop becomes unusable on test N+1's loop
# ("Event loop is closed"). Disposing the pool *before* every test forces
# a brand-new connection on the current test's loop. close=False skips
# the orphan-close step that would itself try to await on the dead loop.


@pytest.fixture(autouse=True)
async def _fresh_engine_per_test():
    from app.db.session import engine

    await engine.dispose(close=False)
    yield


# ── Async session (per-test) ──────────────────────────────────────


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    async with async_session_maker() as session:
        yield session
        try:
            await session.rollback()
        except Exception:
            pass


# ── Fixture document — ingested once per session ──────────────────


@pytest.fixture
async def ingested_doc() -> AsyncIterator[dict]:
    """Idempotently ensure data/uploads/sample.pdf is ingested for the test
    tenant.

    Function-scoped (not session): keeps each test on its own event loop so
    the SQLAlchemy async engine, asyncpg, and pytest-asyncio stay aligned.
    The first call in a fresh DB does the full pipeline; every subsequent
    call hits dedup and finishes in milliseconds.

    Yields {document_id, parent_id, tenant_id, content_hash, collection}.

    Pre-req: the Qdrant collection for the test tenant must already exist
    (run `make seed` once). If it doesn't, this fixture errors loudly.
    """
    file_bytes = FIXTURE_PDF.read_bytes()
    sha = crud.compute_sha256(file_bytes)
    collection = collection_name_for(TEST_TENANT_ID)

    qc = get_qdrant_client()
    try:
        qc.get_collection(collection)
    except Exception as exc:
        pytest.fail(
            f"Qdrant collection {collection!r} not found. "
            f"Run `make seed ORG_ID={TEST_TENANT_ID}` first.\n  underlying: {exc}"
        )

    async with async_session_maker() as session:
        existing = await crud.dedup_check(session, sha)
        if existing is None:
            # Fresh ingest — same pipeline as the Celery task, just inline.
            pages = parse_pdf(FIXTURE_PDF)
            parent_nodes, child_nodes = hierarchical_chunk(pages)
            embedded = await embed_chunks(child_nodes)

            doc = await crud.insert_document(
                session,
                tenant_id=TEST_TENANT_ID,
                filename="sample.pdf",
                content_hash=sha,
                file_path=str(FIXTURE_PDF),
                doc_type="paper",
            )
            await crud.bulk_insert_parent_chunks(
                session, parent_nodes, document_id=doc.id, tenant_id=TEST_TENANT_ID
            )
            await crud.update_document_counts(
                session,
                doc.id,
                page_count=len(pages),
                chunk_count=len(child_nodes),
            )
            await session.commit()
            doc_id = doc.id

            points = [
                build_point(
                    e,
                    tenant_id=TEST_TENANT_ID,
                    document_id=str(doc.id),
                    doc_type="paper",
                )
                for e in embedded
            ]
            upsert_to_qdrant(collection, points)
        else:
            doc_id = existing.id

    async with async_session_maker() as session:
        result = await session.execute(
            text(
                "SELECT parent_id FROM parent_chunks WHERE document_id = :d "
                "ORDER BY chunk_index LIMIT 1"
            ),
            {"d": doc_id},
        )
        parent_id = result.scalar_one()

    yield {
        "document_id": str(doc_id),
        "parent_id": parent_id,
        "tenant_id": TEST_TENANT_ID,
        "content_hash": sha,
        "collection": collection,
    }


# ── Scratch document (per-test) ───────────────────────────────────


@pytest.fixture
async def scratch_doc_id() -> AsyncIterator[str]:
    """Throwaway Document row for mutation tests; cleaned up on exit."""
    rand = uuid.uuid4().hex[:8]
    async with async_session_maker() as session:
        doc = await crud.insert_document(
            session,
            tenant_id=TEST_TENANT_ID,
            filename=f"scratch-{rand}.pdf",
            content_hash=f"scratch-hash-{rand}",
            file_path=f"data/uploads/scratch-{rand}.pdf",
            doc_type="test",
        )
        await session.commit()
        doc_id = str(doc.id)
    try:
        yield doc_id
    finally:
        async with async_session_maker() as session:
            await session.execute(
                text("DELETE FROM parent_chunks WHERE document_id = :d"),
                {"d": uuid.UUID(doc_id)},
            )
            await session.execute(
                text("DELETE FROM documents WHERE id = :d"),
                {"d": uuid.UUID(doc_id)},
            )
            await session.commit()
