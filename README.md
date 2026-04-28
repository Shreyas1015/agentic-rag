# agentic-rag

Personal, full-complexity Agentic RAG — ingest any PDF, ask grounded questions, get streaming answers with inline citations. Self-hosted end-to-end (no managed cloud queues, no SaaS auth).

```
                    ┌─────────────┐                ┌─────────────┐
   ┌─ POST /ingest ─►│             │── Celery ────►│   worker    │
   │                │             │   (Redis)     │             │
   │                │   FastAPI   │               │  Docling    │
   │  ◄ task_id ────│             │               │  Chunk      │
   │                │             │               │  Embed      │
   │                │             │               │  Upsert ──┐ │
   │                └─────────────┘               └────────────┘
   │                       │                                  │
   │ POST /chat/stream     │ retrieve + parent_fetch          │
   │     SSE               ▼                                  ▼
   │                ┌─────────────┐               ┌─────────────┐
   │                │  LangGraph  │◄──── search ──┤   Qdrant    │
   │                │   7 nodes   │               │  (named     │
   │                │  classify…  │               │   vectors)  │
   │                │  generate   │               └─────────────┘
   │                └─────────────┘                       ▲
   │                       │                              │
   │                       └─── parent fetch / metadata ──┘
   │                                                      │
   └─────── tokens ◄── GPT-4o (OpenRouter) ───────────────┘
                              │
                              └── Redis: semantic cache ── 24 h TTL
```

## What it does

A user with a Logto access token can:

- **`POST /ingest`** — upload a PDF; SHA256-deduped, processed asynchronously by a Celery worker (Docling → hierarchical chunk → OpenAI embed via OpenRouter → Qdrant upsert with `is_active=true`).
- **`GET /ingest/status/{task_id}`** — poll the task until success.
- **`POST /chat/stream`** — ask a question; receive a Server-Sent Events stream:
  - `status` events as the LangGraph agent walks: classify → (decompose) → retrieve → parent_fetch → assess → (reformulate ↺) → generate
  - `token` events as GPT-4o streams the final answer
  - `done` event with structured citations `[{document_id, filename, page_num, parent_id}]`
- **Semantic cache hit** on a paraphrased re-ask — sub-second response.

Multi-tenant from day one: every Qdrant query and Postgres row is filtered by `tenant_id`, where `tenant_id` is the Logto `organization_id` claim extracted from the access token. No client can read another tenant's data.

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI · SSE · Pydantic v2 |
| Auth | **Logto OSS** self-hosted; PyJWT verifies access tokens against Logto JWKS (ES384) |
| Async worker | Celery (Redis broker) |
| Vector store | Qdrant 1.17 — named vectors `dense` (1536-d cosine) + `bm25` (sparse, IDF) |
| Relational store | PostgreSQL 16 + SQLAlchemy 2.0 async + asyncpg + Alembic |
| Cache | Redis 7 (DB 0 broker, DB 1 results, DB 2 semantic cache) |
| Document parser | Docling (layout-aware, tables → markdown) |
| Chunking | LlamaIndex `HierarchicalNodeParser` (1024-tok parents, 256-tok children) |
| Dense embeddings | `openai/text-embedding-3-small` via **OpenRouter** |
| Sparse embeddings | FastEmbed `Qdrant/bm25` (local) |
| Retrieval | Qdrant `Prefetch` (dense + bm25) + `Fusion.RRF` |
| Agent | LangGraph state machine, 7 nodes |
| LLMs | `google/gemini-2.5-flash` (orchestration) + `openai/gpt-4o` (generation) — both via OpenRouter |
| Container orchestration | Docker Compose (the same compose file Dokploy will deploy on a server) |
| Python | 3.12 · managed by `uv` |

A single `OPENROUTER_API_KEY` covers all chat + embedding traffic. Production ops live behind one Dokploy reverse proxy.

## Repo layout

```
app/
├── api/             FastAPI routers (health, ingest, chat)
├── agent/           LangGraph state + nodes + graph
├── core/            config, auth (Logto/PyJWT), llm (OpenRouter), celery_app, qdrant_client
├── db/              SQLAlchemy session + ORM models + async CRUD
├── ingestion/       parser, chunker, embedder, upserter, Celery tasks
├── retrieval/       hybrid_search (Qdrant RRF), cache (Redis semantic)
└── observability/   (Phase 2 — Langfuse, RAGAS)
docker-compose.yml   postgres, redis, qdrant, logto + (Phase G1) migrate, api, worker
docs/                source-of-truth artifacts (architecture decisions, build guide)
migrations/          Alembic
scripts/             seed_tenant, create_collections, run_eval (Phase 2)
tests/               (Phase G2)
```

## Quick start (local dev)

Prerequisites: Docker Desktop, Python 3.12 + `uv`.

```bash
# 1. Clone + bootstrap
git clone https://github.com/Shreyas1015/agentic-rag.git
cd agentic-rag
cp .env.example .env
# Edit .env: paste your OPENROUTER_API_KEY, LOGTO_APP_ID, LOGTO_APP_SECRET

# 2. Bring up the data plane (postgres + redis + qdrant + logto)
docker compose up -d postgres redis qdrant logto

# 3. (One-time) Open Logto admin and onboard
#    http://localhost:3002 → create admin user → API resource (id = LOGTO_RESOURCE)
#    → Machine-to-Machine app → copy app id/secret into .env
#    → Create an Organization, copy its id (this is your tenant_id)

# 4. Sync deps + run migrations
uv sync
uv run alembic upgrade head
uv run python scripts/seed_tenant.py --logto-org-id <ORG_ID> --name "My Project"
uv run python scripts/create_collections.py --tenant-id <ORG_ID>

# 5. Run api + worker (in two terminals)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
uv run celery -A app.core.celery_app worker --loglevel=info --pool=solo  # Windows: --pool=solo

# 6. Mint an M2M token, then upload a PDF
TOKEN=$(curl -s -X POST $LOGTO_ENDPOINT/oidc/token \
  -d grant_type=client_credentials \
  -d resource=$LOGTO_RESOURCE \
  -d organization_id=<ORG_ID> \
  -u "$LOGTO_APP_ID:$LOGTO_APP_SECRET" | jq -r .access_token)

curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@my.pdf;type=application/pdf"

curl --no-buffer -X POST http://localhost:8000/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "..."}'
```

The full Docker stack (api + worker containerized too) lands in commit *G1* — `docker compose up` becomes a single command that boots everything.

## Status

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | infra + ingest + retrieve + agent + chat | 🟡 ~95% (G1 + G2 in progress) |
| Phase 2 | BGE rerank · faithfulness · Langfuse · RAGAS · feedback API | planned |
| Phase 3 | Dokploy deploy · cost dashboard · long-context bypass | planned |

End-to-end smoke (host-mode) verified:
- Cold query: ~7 s, full agent traversal, streaming GPT-4o tokens with citations
- Warm query: ~0.9 s via semantic cache (cosine ≥ 0.85)
- Counts consistent: `documents.chunk_count` ↔ `parent_chunks` rows ↔ Qdrant active points

## Source-of-truth docs

Architecture decisions and the full build guide live in [`docs/`](docs/):

- [`docs/artifact1_architecture_decisions.html`](docs/artifact1_architecture_decisions.html) — final tech stack, ADRs, what changed and why
- [`docs/artifact2_build_guide.html`](docs/artifact2_build_guide.html) — phased build guide, folder tree, code skeletons, env vars, setup steps

Open them in a browser; they're self-contained HTML.
