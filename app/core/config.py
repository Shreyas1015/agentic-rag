from functools import cached_property

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────
    APP_NAME: str = "agentic-rag"
    APP_ENV: str = "dev"
    DEBUG: bool = True

    # ── LLM gateway (OpenRouter) ───────────────────────────
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_APP_REFERER: str = "http://localhost:8000"
    OPENROUTER_APP_TITLE: str = "agentic-rag"

    LLM_MODEL_CLASSIFY: str = "google/gemini-2.5-flash"
    LLM_MODEL_DECOMPOSE: str = "google/gemini-2.5-flash"
    LLM_MODEL_ASSESS: str = "google/gemini-2.5-flash"
    LLM_MODEL_REFORMULATE: str = "google/gemini-2.5-flash"
    LLM_MODEL_GENERATE: str = "openai/gpt-4o"
    LLM_MODEL_LONG_CONTEXT: str = "google/gemini-2.5-pro"
    EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
    EMBEDDING_DIMS: int = 1536

    # ── Long-context bypass (Phase 3) ──────────────────────
    # If the tenant's full active corpus fits within this many tokens, the
    # agent skips RAG (retrieve / rerank / parent_fetch) and feeds the
    # entire corpus to LLM_MODEL_LONG_CONTEXT alongside the question.
    # Defaults to 800k — leaves headroom inside Gemini 2.5 Pro's 1M ctx for
    # prompt + answer.
    LONG_CONTEXT_BYPASS_TOKEN_BUDGET: int = 800_000

    # ── Auth (Logto) ───────────────────────────────────────
    LOGTO_ENDPOINT: str = ""           # public/advertised URL — DEFINES the issuer (`iss` claim)
    # Optional service-to-service URL for JWKS fetch. Use this when the api
    # runs inside Docker and reaches Logto via the compose service name
    # (http://logto:3001) while tokens are issued/consumed using the public
    # http://localhost:3001 URL. If empty, we fall back to LOGTO_ENDPOINT.
    LOGTO_INTERNAL_ENDPOINT: str = ""
    LOGTO_RESOURCE: str = ""           # API resource indicator (becomes `aud`)
    LOGTO_APP_ID: str = ""             # M2M app ID (used by token-issuing clients)
    LOGTO_APP_SECRET: str = ""         # M2M app secret (only needed by clients)

    # ── PostgreSQL ─────────────────────────────────────────
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "postgres"    # in-Docker service name
    POSTGRES_PORT: int = 5432          # in-container port
    POSTGRES_HOST_PORT: int = 5433     # only used by docker-compose port mapping
    # Optional override. Set this when running FastAPI from host (uv run uvicorn)
    # against the Docker postgres on a non-default port:
    #   DATABASE_URL=postgresql+asyncpg://raguser:ragpassword@localhost:5433/ragdb
    DATABASE_URL: str = ""

    # ── Redis ──────────────────────────────────────────────
    REDIS_URL: str = "redis://redis:6379"
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"
    SEMANTIC_CACHE_REDIS_URL: str = "redis://redis:6379/2"
    CACHE_SIMILARITY_THRESHOLD: float = 0.88
    CACHE_TTL_SECONDS: int = 86400

    # ── Qdrant ─────────────────────────────────────────────
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333

    # ── Agent tuning ───────────────────────────────────────
    CONTEXT_SCORE_THRESHOLD: int = Field(default=7, ge=0, le=10)
    MAX_RETRIEVAL_ITERATIONS: int = Field(default=3, ge=1)

    # ── Ingestion ──────────────────────────────────────────
    # OCR pulls a Chinese-host model on first run that often fails behind
    # corporate proxies. Default off; in Docker we'll enable it (Step G uses
    # the bundled tesseract from the image so no download is needed).
    DOCLING_OCR_ENABLED: bool = False

    # ── Langfuse (Phase 2) ─────────────────────────────────
    # When unset, tracing is disabled (every @observe becomes a no-op).
    # In Docker, compose overrides LANGFUSE_HOST to http://langfuse-web:3000.
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"

    # ── Derived ────────────────────────────────────────────
    @cached_property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @cached_property
    def logto_jwks_uri(self) -> str:
        # Logto exposes JWKS at /oidc/jwks (the value is also discoverable
        # via /oidc/.well-known/openid-configuration → jwks_uri). Prefer
        # LOGTO_INTERNAL_ENDPOINT for the fetch since that's the
        # service-to-service URL inside Docker.
        base = self.LOGTO_INTERNAL_ENDPOINT or self.LOGTO_ENDPOINT
        return f"{base.rstrip('/')}/oidc/jwks"

    @cached_property
    def logto_issuer(self) -> str:
        return f"{self.LOGTO_ENDPOINT.rstrip('/')}/oidc"


settings = Settings()
