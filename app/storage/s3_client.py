"""S3-compatible object storage for tenant documents.

Wraps aioboto3 with our two real targets — MinIO in dev, AWS S3 in prod —
behind one async surface. The migration path is just env vars:
unset S3_ENDPOINT_URL and switch creds to AWS, the same code keeps working.

Object key layout: `<tenant_id>/<document_id>.pdf` so tenant isolation
shows up in the S3 console too. Browser-bound URLs swap the in-Docker
S3_ENDPOINT_URL host for S3_PUBLIC_URL so the presigned signature is still
valid against the publicly-reachable MinIO port.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aioboto3
from botocore.exceptions import ClientError
from botocore.config import Config

from app.core.config import settings

log = logging.getLogger(__name__)

_session = aioboto3.Session()


@asynccontextmanager
async def s3_client(*, public: bool = False) -> AsyncIterator:
    """Yield an aioboto3 S3 client.

    Default (`public=False`) targets the in-Docker endpoint we use for
    uploads/deletes (e.g. `http://minio:9000`). `public=True` targets the
    browser-reachable URL (`S3_PUBLIC_URL`) — used only for generating
    presigned URLs because the canonical signed request includes the Host
    header, and the browser will fetch from `localhost:9090` (or the
    public hostname in prod). Presigning never makes an HTTP call, so the
    signing endpoint doesn't need to be reachable from the container.
    """
    endpoint = (
        settings.S3_PUBLIC_URL if public else settings.S3_ENDPOINT_URL
    ) or None
    async with _session.client(
        "s3",
        endpoint_url=endpoint,
        region_name=settings.S3_REGION,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    ) as client:
        yield client


def document_key(tenant_id: str, document_id: str) -> str:
    """Object key for a tenant's uploaded document."""
    return f"{tenant_id}/{document_id}.pdf"


async def ensure_bucket() -> None:
    """Create the documents bucket if it doesn't exist. Idempotent.

    Called from the FastAPI lifespan so we fail loud at startup instead of
    on the first upload.
    """
    bucket = settings.S3_DOCUMENTS_BUCKET
    async with s3_client() as s3:
        try:
            await s3.head_bucket(Bucket=bucket)
            return
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
        log.info("Creating S3 bucket %s on %s", bucket, settings.S3_ENDPOINT_URL)
        await s3.create_bucket(Bucket=bucket)


async def upload_document(
    tenant_id: str, document_id: str, content: bytes
) -> str:
    """Upload a PDF and return its object key."""
    key = document_key(tenant_id, document_id)
    async with s3_client() as s3:
        await s3.put_object(
            Bucket=settings.S3_DOCUMENTS_BUCKET,
            Key=key,
            Body=content,
            ContentType="application/pdf",
        )
    return key


async def presigned_document_url(
    tenant_id: str,
    document_id: str,
    *,
    filename: str | None = None,
    inline: bool = True,
) -> str:
    """Generate a time-limited presigned GET URL for a tenant's PDF.

    `inline=True` makes browsers render the PDF in a tab instead of downloading.
    `filename` controls the suggested name when the user does download.
    The signature is bound to the in-Docker endpoint, but presigned URLs are
    portable: we rewrite the host to S3_PUBLIC_URL so the browser can reach it.
    """
    key = document_key(tenant_id, document_id)
    disposition = (
        f'{"inline" if inline else "attachment"}; filename="{filename or document_id}.pdf"'
    )
    # Sign against the PUBLIC URL — the canonical signed request includes
    # the Host header, and the browser will hit S3_PUBLIC_URL. A naive
    # string-replace on the URL after signing breaks the signature.
    async with s3_client(public=True) as s3:
        return await s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": settings.S3_DOCUMENTS_BUCKET,
                "Key": key,
                "ResponseContentDisposition": disposition,
                "ResponseContentType": "application/pdf",
            },
            ExpiresIn=settings.S3_PRESIGN_TTL_SECONDS,
        )


async def delete_document(tenant_id: str, document_id: str) -> None:
    """Best-effort delete. Caller already updated Postgres + Qdrant."""
    key = document_key(tenant_id, document_id)
    try:
        async with s3_client() as s3:
            await s3.delete_object(Bucket=settings.S3_DOCUMENTS_BUCKET, Key=key)
    except ClientError:
        log.exception("S3 delete failed for %s (non-fatal)", key)
