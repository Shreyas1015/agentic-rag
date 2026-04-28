"""Shared LLM clients pointed at OpenRouter.

Two singletons:
  - get_openai_client() / get_async_openai_client():
      `openai.OpenAI` / `openai.AsyncOpenAI` configured with OpenRouter base URL
      and our API key. Use directly for embeddings or simple chat completions
      (e.g. ingestion/embedder.py).
  - get_chat_model(model_id, ...):
      `langchain_openai.ChatOpenAI` configured the same way. Use inside
      LangGraph nodes that prefer LangChain idioms.

OpenRouter requires an `HTTP-Referer` and a friendly `X-Title` header for
attribution; both are sent automatically.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI

from app.core.config import settings


def _default_headers() -> dict[str, str]:
    return {
        "HTTP-Referer": settings.OPENROUTER_APP_REFERER,
        "X-Title": settings.OPENROUTER_APP_TITLE,
    }


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        default_headers=_default_headers(),
    )


@lru_cache(maxsize=1)
def get_async_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        default_headers=_default_headers(),
    )


def get_chat_model(
    model_id: str,
    *,
    temperature: float = 0.0,
    json_mode: bool = False,
    streaming: bool = False,
) -> ChatOpenAI:
    """Build a LangChain ChatOpenAI that talks to OpenRouter.

    Not cached because callers may want different models / temperatures /
    streaming flags. Construction is cheap.
    """
    kwargs: dict = {
        "model": model_id,
        "temperature": temperature,
        "openai_api_key": settings.OPENROUTER_API_KEY,
        "openai_api_base": settings.OPENROUTER_BASE_URL,
        "default_headers": _default_headers(),
        "streaming": streaming,
    }
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kwargs)
