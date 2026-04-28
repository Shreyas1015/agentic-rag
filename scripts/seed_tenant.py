"""Insert / upsert a tenant row mapped to a Logto organization.

Usage:
    uv run python scripts/seed_tenant.py \\
        --logto-org-id <ORG_ID_FROM_LOGTO> \\
        --name "myproject"

The `tenant_id` we store is the Logto organization_id verbatim — that's
the value the API extracts from the access-token claims, so they must match.
The Qdrant collection name is derived as `tenant_<tenant_id>`; this script
does NOT create the Qdrant collection (run scripts/create_collections.py
afterwards).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make the repo root importable when running as `python scripts/seed_tenant.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.core.qdrant_client import collection_name_for  # noqa: E402
from app.db.models import Tenant  # noqa: E402
from app.db.session import async_session_maker  # noqa: E402


async def upsert_tenant(logto_org_id: str, name: str | None) -> Tenant:
    async with async_session_maker() as session:
        existing = await session.scalar(
            select(Tenant).where(Tenant.tenant_id == logto_org_id)
        )
        if existing is not None:
            print(f"[update] tenant_id={logto_org_id} (id={existing.id})")
            if name and existing.name != name:
                existing.name = name
            existing.qdrant_collection = collection_name_for(logto_org_id)
            await session.commit()
            await session.refresh(existing)
            return existing

        tenant = Tenant(
            tenant_id=logto_org_id,
            name=name,
            qdrant_collection=collection_name_for(logto_org_id),
        )
        session.add(tenant)
        await session.commit()
        await session.refresh(tenant)
        print(f"[create] tenant_id={logto_org_id} (id={tenant.id})")
        return tenant


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed a tenant row mapped to a Logto organization.")
    parser.add_argument(
        "--logto-org-id",
        required=True,
        help="Logto organization_id — used verbatim as our tenant_id.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Human-readable tenant name (optional).",
    )
    args = parser.parse_args()

    tenant = asyncio.run(upsert_tenant(args.logto_org_id, args.name))

    print()
    print("[ok]")
    print(f"  tenant.id           = {tenant.id}")
    print(f"  tenant.tenant_id    = {tenant.tenant_id}")
    print(f"  tenant.name         = {tenant.name}")
    print(f"  qdrant_collection   = {tenant.qdrant_collection}")
    print()
    print("Next:")
    print(f"  uv run python scripts/create_collections.py --tenant-id {tenant.tenant_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
