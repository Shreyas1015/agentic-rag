"""Hierarchical chunking with LlamaIndex.

Two-level split:
  parents (1024-token windows) -> stored in PostgreSQL.parent_chunks; fetched
                                  AFTER retrieval to give the LLM coherent
                                  context.
  children (256-token chunks)  -> embedded and stored in Qdrant; the unit of
                                  retrieval.

Each child carries `parent_id` (the LlamaIndex node_id of its 1024-token
ancestor) so we can join back to PostgreSQL after Qdrant returns hits.

`page_num` is propagated from `PageText` metadata down through parents and
children so we keep page-level citations end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.schema import NodeRelationship

from app.ingestion.parser import PageText

PARENT_CHUNK_SIZE = 1024
CHILD_CHUNK_SIZE = 256


@dataclass
class ParentNode:
    parent_id: str  # LlamaIndex node_id (UUID-shaped)
    text: str
    page_num: int  # first page where this span starts
    chunk_index: int  # ordinal within document


@dataclass
class ChildNode:
    chunk_id: str  # LlamaIndex node_id (will be the Qdrant point id)
    parent_id: str  # join key back to ParentNode / parent_chunks table
    text: str
    page_num: int
    chunk_index: int


def hierarchical_chunk(
    pages: list[PageText],
) -> tuple[list[ParentNode], list[ChildNode]]:
    """Split per-page texts into 1024/256-token parents/children."""
    if not pages:
        return [], []

    # One Document per page so page_num survives in node metadata.
    documents = [
        Document(text=p.text, metadata={"page_num": p.page_num}) for p in pages
    ]

    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE]
    )
    nodes = parser.get_nodes_from_documents(documents)

    roots = get_root_nodes(nodes)  # 1024-token chunks
    leaves = get_leaf_nodes(nodes)  # 256-token chunks

    parents: list[ParentNode] = []
    parent_id_set: set[str] = set()
    for i, node in enumerate(roots):
        parent_id_set.add(node.node_id)
        parents.append(
            ParentNode(
                parent_id=node.node_id,
                text=node.get_content(),
                page_num=int(node.metadata.get("page_num", 1)),
                chunk_index=i,
            )
        )

    children: list[ChildNode] = []
    for i, node in enumerate(leaves):
        # Two-level hierarchy => leaf's PARENT relationship points at a root.
        parent_ref = node.relationships.get(NodeRelationship.PARENT)
        parent_id = parent_ref.node_id if parent_ref else node.node_id
        # Defensive fallback: if for some reason the parent isn't in the root
        # set (shouldn't happen with chunk_sizes=[1024, 256]), keep the leaf id.
        if parent_id not in parent_id_set:
            parent_id = node.node_id

        children.append(
            ChildNode(
                chunk_id=node.node_id,
                parent_id=parent_id,
                text=node.get_content(),
                page_num=int(node.metadata.get("page_num", 1)),
                chunk_index=i,
            )
        )

    return parents, children
