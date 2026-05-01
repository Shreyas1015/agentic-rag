import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    chat,
    documents,
    feedback,
    health,
    ingest,
    me,
    sessions,
    usage,
)
from app.core.config import settings
from app.storage.s3_client import ensure_bucket as ensure_documents_bucket

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Ensure the S3/MinIO documents bucket exists before serving any
    # uploads. Failure here surfaces at boot rather than on first upload.
    try:
        await ensure_documents_bucket()
    except Exception:
        log.exception("Object storage init failed — uploads will return 5xx")
    yield


app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG, lifespan=lifespan)

# CORS — the Next.js BFF proxies most calls server-side, but the SSE chat
# stream is consumed directly from the browser through that same proxy, so
# we still need permissive CORS for the configured frontend origin.
_origins = [
    o.strip()
    for o in (
        getattr(settings, "FRONTEND_ORIGINS", None)
        or "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(health.router)
app.include_router(me.router)
app.include_router(usage.router)
app.include_router(ingest.router)
app.include_router(documents.router)
app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(feedback.router)
