"""Create the Qdrant collection for a tenant.

Usage:
    uv run python scripts/create_collections.py --tenant-id myproject
    uv run python scripts/create_collections.py --tenant-id myproject --recreate

The collection is named `tenant_<tenant_id>` and holds 256-token *child*
chunks. Named vectors:
  - dense : `EMBEDDING_DIMS`-d cosine (default 1536, openai/text-embedding-3-small via OpenRouter)
  - bm25  : sparse, populated by FastEmbed BM25 (IDF modifier for proper scoring)

Payload indexes on tenant_id, is_active, document_id, parent_id to keep
filtered hybrid searches fast.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when running as `python scripts/create_collections.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.http.exceptions import UnexpectedResponse  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import settings  # noqa: E402
from app.core.qdrant_client import collection_name_for, get_qdrant_client  # noqa: E402


def create_collection(tenant_id: str, *, recreate: bool = False) -> str:
    client = get_qdrant_client()
    name = collection_name_for(tenant_id)

    exists = client.collection_exists(name)
    if exists and not recreate:
        print(f"[skip] collection {name!r} already exists; pass --recreate to drop and rebuild.")
        return name
    if exists and recreate:
        print(f"[drop] {name}")
        client.delete_collection(name)

    print(f"[create] {name}")
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=settings.EMBEDDING_DIMS, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(
                modifier=Modifier.IDF,
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )

    print("[index] payload fields")
    for field, schema in [
        ("tenant_id", PayloadSchemaType.KEYWORD),
        ("is_active", PayloadSchemaType.BOOL),
        ("document_id", PayloadSchemaType.KEYWORD),
        ("parent_id", PayloadSchemaType.KEYWORD),
        ("doc_type", PayloadSchemaType.KEYWORD),
        ("page_num", PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(name, field_name=field, field_schema=schema)
        print(f"  - {field} ({schema.value})")

    return name


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Qdrant collection for a tenant.")
    parser.add_argument("--tenant-id", required=True, help="Tenant slug (matches Logto organization_id).")
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild if it already exists.")
    args = parser.parse_args()

    try:
        name = create_collection(args.tenant_id, recreate=args.recreate)
    except UnexpectedResponse as exc:
        print(f"[error] Qdrant rejected the request: {exc}", file=sys.stderr)
        return 1

    info = get_qdrant_client().get_collection(name)
    print(f"\n[ok] collection={name}")
    print(f"     status={info.status.value}  points={info.points_count}")
    print(f"     vectors={list(info.config.params.vectors.keys())}")
    print(f"     sparse_vectors={list(info.config.params.sparse_vectors.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
