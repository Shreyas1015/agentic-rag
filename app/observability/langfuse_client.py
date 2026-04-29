"""Langfuse v4 SDK wrapper.

Self-hosted Langfuse runs at http://langfuse-web:3000 inside the Docker
network and http://localhost:3000 from the host. The SDK is initialized
on first use (lazy singleton) using LANGFUSE_PUBLIC_KEY / SECRET_KEY /
HOST from settings.

When keys are unset (CI without Langfuse, dev with it disabled), the
client is constructed with `tracing_enabled=False` — `@observe()` calls
become no-ops and `update_current_span(...)` is safe to call. We never
crash the request path because tracing is misconfigured.

Usage from agent nodes:

    from langfuse import observe
    from app.observability.langfuse_client import langfuse, current_trace_id

    @observe(name="retrieve_node")
    async def retrieve(state: AgentState) -> dict:
        ...
        langfuse.update_current_span(
            input={"query": state["query"]},
            output={"hits": len(hits)},
            metadata={"top_scores": [h.score for h in hits[:5]]},
        )
        return {"retrieved_chunks": [...]}

The endpoint reads the trace id via `current_trace_id()` to surface it
in the SSE `done` event so /feedback can target it.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langfuse import Langfuse, get_client

from app.core.config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    """Lazy-init singleton. When keys are missing we still return a client
    but with tracing disabled — every `@observe()` becomes a cheap no-op."""
    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        log.warning(
            "Langfuse keys not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY); "
            "tracing disabled."
        )
        return Langfuse(tracing_enabled=False)

    return Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
        tracing_enabled=True,
        environment=settings.APP_ENV,
        release=settings.APP_NAME,
    )


# Module-level alias so call sites can `from app.observability.langfuse_client
# import langfuse` and just call methods. We use a lazy proxy so import order
# doesn't trigger network I/O.
class _LazyLangfuse:
    """Defers .Langfuse() construction until the first attribute access."""

    def __getattr__(self, item):
        return getattr(get_langfuse(), item)


langfuse: Langfuse = _LazyLangfuse()  # type: ignore[assignment]


def current_trace_id() -> str | None:
    """Return the active trace id (or None if we're outside an @observe scope
    or tracing is disabled)."""
    try:
        client = get_client()
    except Exception:  # pragma: no cover  — never break the request
        return None
    try:
        return client.get_current_trace_id()
    except Exception:
        return None


def flush() -> None:
    """Flush any pending events. Safe to call on shutdown."""
    try:
        get_langfuse().flush()
    except Exception:
        log.exception("langfuse flush failed")


def usage_from_response(response) -> dict:
    """Pull usage + cost details out of a LangChain AIMessage in the shape
    Langfuse's update_current_generation expects.

    Returns a dict of kwargs to splat into update_current_generation:
        {"model": ..., "usage_details": ..., "cost_details": ...}
    Empty dict if the response carries no usage metadata (shouldn't happen
    with OpenRouter — they pass it through — but be defensive).
    """
    out: dict = {}
    md = getattr(response, "usage_metadata", None) or {}
    if md:
        out["usage_details"] = {
            "input": int(md.get("input_tokens", 0)),
            "output": int(md.get("output_tokens", 0)),
            "total": int(
                md.get(
                    "total_tokens",
                    md.get("input_tokens", 0) + md.get("output_tokens", 0),
                )
            ),
        }
    rm = getattr(response, "response_metadata", None) or {}
    name = rm.get("model_name")
    if name:
        out["model"] = name
    # OpenRouter returns the resolved upstream cost in token_usage.cost
    # (USD float). Pass it as a single-bucket cost_details so Langfuse's
    # cost dashboard sees the truth (not its own model-pricing estimate).
    tu = rm.get("token_usage") or {}
    if "cost" in tu and tu["cost"] is not None:
        out["cost_details"] = {"total": float(tu["cost"])}
    return out
