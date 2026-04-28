"""Fetch the 1024-token parent windows behind the retrieved children.

For each unique parent_id in retrieved_chunks, look up the row in
parent_chunks. The order of the resulting list matches the order in which
parents first appear among the (already RRF-ranked) children — so the most
relevant context lands first in the LLM prompt.
"""

from __future__ import annotations

from app.agent.state import AgentState
from app.db import crud
from app.db.session import async_session_maker


async def parent_fetch(state: AgentState) -> dict:
    tenant_id = state["tenant_id"]
    children = state.get("retrieved_chunks") or []
    if not children:
        return {"parent_chunks": []}

    # Preserve the order in which each parent first appears.
    seen: set[str] = set()
    ordered_parent_ids: list[str] = []
    for c in children:
        pid = c.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            ordered_parent_ids.append(pid)

    async with async_session_maker() as session:
        rows = await crud.fetch_parent_chunks(
            session, tenant_id=tenant_id, parent_ids=ordered_parent_ids
        )

    by_id = {r.parent_id: r for r in rows}
    parents = []
    for pid in ordered_parent_ids:
        r = by_id.get(pid)
        if r is None:
            continue
        parents.append(
            {
                "parent_id": r.parent_id,
                "document_id": str(r.document_id),
                "tenant_id": r.tenant_id,
                "text": r.text,
                "page_num": r.page_num,
                "chunk_index": r.chunk_index,
            }
        )

    return {"parent_chunks": parents}
