"""Health endpoints.

GET /health         cheap liveness check; never reaches dependencies.
GET /health/full    component check (postgres, redis, qdrant, celery, logto).
                    Used by the settings page infra panel.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
import redis.asyncio as redis
from fastapi import APIRouter
from sqlalchemy import text

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.qdrant_client import get_async_qdrant_client
from app.db.session import async_session_maker

log = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

_BOOT_TIME = time.time()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


async def _check_postgres() -> dict:
    started = time.perf_counter()
    try:
        async with async_session_maker() as s:
            await s.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": str(exc),
        }


async def _check_redis() -> dict:
    started = time.perf_counter()
    client = redis.from_url(settings.REDIS_URL)
    try:
        pong = await client.ping()
        return {
            "status": "ok" if pong else "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": str(exc),
        }
    finally:
        await client.aclose()


async def _check_qdrant() -> dict:
    started = time.perf_counter()
    try:
        client = get_async_qdrant_client()
        cols = await client.get_collections()
        return {
            "status": "ok",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": f"{len(cols.collections)} collections",
        }
    except Exception as exc:
        return {
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": str(exc),
        }


def _inspect_celery_sync() -> dict:
    started = time.perf_counter()
    try:
        active = celery_app.control.inspect(timeout=1.0).active() or {}
        worker_count = len(active)
        return {
            "status": "ok" if worker_count > 0 else "degraded",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": f"{worker_count} workers",
            "active_workers": worker_count,
        }
    except Exception as exc:
        return {
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": str(exc),
            "active_workers": 0,
        }


async def _check_celery() -> dict:
    return await asyncio.to_thread(_inspect_celery_sync)


async def _check_logto() -> dict:
    started = time.perf_counter()
    base = settings.LOGTO_INTERNAL_ENDPOINT or settings.LOGTO_ENDPOINT
    if not base:
        return {
            "status": "error",
            "latency_ms": 0.0,
            "detail": "LOGTO_ENDPOINT not configured",
            "issuer": None,
        }
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{base.rstrip('/')}/oidc/jwks")
        return {
            "status": "ok" if r.status_code == 200 else "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": f"HTTP {r.status_code}",
            "issuer": settings.logto_issuer,
        }
    except Exception as exc:
        return {
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": str(exc),
            "issuer": settings.logto_issuer,
        }


@router.get("/health/full", summary="Component-level health for the dashboard")
async def health_full() -> dict:
    pg, rd, qd, cl, lg = await asyncio.gather(
        _check_postgres(),
        _check_redis(),
        _check_qdrant(),
        _check_celery(),
        _check_logto(),
    )
    components = {
        "postgres": pg,
        "redis": rd,
        "qdrant": qd,
        "celery": cl,
        "logto": lg,
    }
    statuses = [c["status"] for c in components.values()]
    if all(s == "ok" for s in statuses):
        overall = "healthy"
    elif any(s == "error" for s in statuses):
        overall = "unhealthy"
    else:
        overall = "degraded"

    return {
        "status": overall,
        "components": components,
        "version": settings.APP_NAME,
        "uptime_seconds": round(time.time() - _BOOT_TIME, 1),
    }
