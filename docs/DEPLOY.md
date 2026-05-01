# Production deploy via Dokploy

Runbook for deploying agentic-rag to a Dokploy-managed EC2 host. Same `docker-compose.yml` Dokploy already reads in dev, plus `docker-compose.prod.yml` which strips host port mappings (Traefik routes by Host header), forces every secret to be set, and persists uploads to a named volume.

## 1. Prod architecture

| Component           | Where it lives                                             | Why                                                                   |
| ------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------- |
| Postgres            | **AWS RDS** (single instance, three databases)             | Managed backups, no container disk pressure, durable across redeploys |
| Qdrant              | container, persistent volume                               | Tenant collections — small enough to live with the app                |
| Redis               | container                                                  | Celery broker + semantic cache — ephemeral by design                  |
| MinIO               | container, persistent volume (or swap to AWS S3 via env)   | PDF object storage; presigned URLs for browser preview                |
| ClickHouse          | container, persistent volume                               | Required by Langfuse for trace storage                                |
| Logto               | container, talks to RDS `logto` DB                         | OIDC + admin console                                                  |
| Langfuse web/worker | containers, talk to RDS `langfuse` DB + ClickHouse + MinIO | Trace UI + ingest worker                                              |
| Traefik             | managed by Dokploy on the host                             | TLS + subdomain routing — no service binds host ports                 |

The three RDS databases (`agentic_rag`, `logto`, `langfuse`) live on one Postgres instance. RDS user must have `CREATE ROLE` (Logto creates its own roles at startup — Neon is incompatible because of this).

## 2. Prerequisites

- EC2 host with Dokploy installed. Security group must allow `22`, `80`, `443`, and `3000` (Dokploy admin UI).
- Five subdomains pointing at the host:
  - `api.example.com` → FastAPI
  - `auth.example.com` → Logto OIDC
  - `auth-admin.example.com` → Logto admin console
  - `traces.example.com` → Langfuse UI
  - `files.example.com` → MinIO (browser fetches presigned PDF URLs)
- An RDS Postgres instance reachable from the host. Three databases pre-created: `agentic_rag`, `logto`, `langfuse`. The DB user must have `CREATE ROLE`.
- An OpenRouter API key with budget for production traffic.
- ~12 GB RAM free, ~10 GB disk for `qdrant` + `clickhouse` + `minio` + `uploads_data` + HF model cache.

## 3. First-time deploy

### 3.1 Generate secrets

On a workstation (not the shared host) so secrets don't end up in shell history:

```bash
echo "LOGTO_SECRET_VAULT_KEK=$(openssl rand -base64 32)"
echo "LANGFUSE_SALT=$(openssl rand -hex 32)"
echo "LANGFUSE_ENCRYPTION_KEY=$(openssl rand -hex 32)"
echo "LANGFUSE_NEXTAUTH_SECRET=$(openssl rand -hex 32)"
echo "LANGFUSE_INIT_USER_PASSWORD=$(openssl rand -base64 24)"
echo "LANGFUSE_PUBLIC_KEY=pk-lf-$(openssl rand -hex 16)"
echo "LANGFUSE_SECRET_KEY=sk-lf-$(openssl rand -hex 32)"
echo "MINIO_ROOT_USER=ragadmin"
echo "MINIO_ROOT_PASSWORD=$(openssl rand -base64 24)"
```

Save the output in your password manager — Dokploy needs all of it in step 3.3.

### 3.2 Point Dokploy at the compose files

In Dokploy → **Create Application** → **Docker Compose**:

- **Source:** GitHub → `Shreyas1015/agentic-rag`, branch `main`
- **Compose file:** `docker-compose.yml`
- **Compose override:** `docker-compose.prod.yml`

Equivalent to running (from the repo root):

```
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

### 3.3 Production env

Paste the block below into Dokploy's environment editor, replacing every `<...>`. The overlay uses `${VAR:?msg}` for required secrets — if anything is missing, `docker compose config` fails loudly at deploy time with the variable name.

```ini
# ── App ──────────────────────────────────────────
APP_NAME=agentic-rag
APP_ENV=prod
DEBUG=false
FRONTEND_ORIGINS=https://app.example.com

# ── OpenRouter ───────────────────────────────────
OPENROUTER_API_KEY=<sk-or-v1-...>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_APP_REFERER=https://api.example.com
OPENROUTER_APP_TITLE=agentic-rag

LLM_MODEL_CLASSIFY=google/gemini-2.5-flash
LLM_MODEL_DECOMPOSE=google/gemini-2.5-flash
LLM_MODEL_ASSESS=google/gemini-2.5-flash
LLM_MODEL_REFORMULATE=google/gemini-2.5-flash
LLM_MODEL_GENERATE=anthropic/claude-sonnet-4.6
LLM_MODEL_LONG_CONTEXT=google/gemini-3.1-pro-preview:nitro
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIMS=1536
LONG_CONTEXT_BYPASS_TOKEN_BUDGET=800000

# ── Logto (public URLs behind Traefik) ───────────
LOGTO_ENDPOINT=https://auth.example.com
LOGTO_ADMIN_ENDPOINT=https://auth-admin.example.com
LOGTO_RESOURCE=https://api.example.com
LOGTO_APP_ID=<filled in step 4.1>
LOGTO_APP_SECRET=<filled in step 4.1>
LOGTO_SECRET_VAULT_KEK=<from 3.1>

# ── Postgres on RDS (single source of truth) ─────
# asyncpg uses ?ssl=require ; Node pg uses ?sslmode=require
DATABASE_URL=postgresql+asyncpg://<user>:<pw>@<rds-host>:5432/agentic_rag?ssl=require
LOGTO_DB_URL=postgres://<user>:<pw>@<rds-host>:5432/logto?sslmode=require
LANGFUSE_DATABASE_URL=postgresql://<user>:<pw>@<rds-host>:5432/langfuse?sslmode=require

# Legacy POSTGRES_* — still required by pydantic-settings validators.
# Values unused once DATABASE_URL is set.
POSTGRES_USER=<user>
POSTGRES_PASSWORD=<pw>
POSTGRES_DB=agentic_rag
POSTGRES_HOST=<rds-host>
POSTGRES_PORT=5432
POSTGRES_HOST_PORT=5432

# ── Redis (in-stack) ─────────────────────────────
REDIS_URL=redis://redis:6379
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
SEMANTIC_CACHE_REDIS_URL=redis://redis:6379/2
CACHE_SIMILARITY_THRESHOLD=0.85
CACHE_TTL_SECONDS=86400

# ── Qdrant (in-stack) ────────────────────────────
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# ── Object storage (MinIO in-stack; S3-compatible) ───
# S3_ENDPOINT_URL is set in the prod overlay (in-Docker, http://minio:9000).
# S3_PUBLIC_URL is what the BROWSER fetches presigned URLs from — must
# resolve to the Traefik-fronted MinIO subdomain.
S3_PUBLIC_URL=https://files.example.com
MINIO_ROOT_USER=<from 3.1>
MINIO_ROOT_PASSWORD=<from 3.1>

# ── Langfuse (public URL behind Traefik) ─────────
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_NEXTAUTH_URL=https://traces.example.com
LANGFUSE_PUBLIC_KEY=<from 3.1>
LANGFUSE_SECRET_KEY=<from 3.1>
LANGFUSE_SALT=<from 3.1>
LANGFUSE_ENCRYPTION_KEY=<from 3.1>
LANGFUSE_NEXTAUTH_SECRET=<from 3.1>
LANGFUSE_INIT_USER_EMAIL=<your admin email>
LANGFUSE_INIT_USER_PASSWORD=<from 3.1>
LANGFUSE_INIT_USER_NAME=Admin
LANGFUSE_INIT_ORG_ID=agentic-rag-prod
LANGFUSE_INIT_ORG_NAME=Agentic RAG
LANGFUSE_INIT_PROJECT_ID=agentic-rag
LANGFUSE_INIT_PROJECT_NAME=agentic-rag

# ── Agent tuning ─────────────────────────────────
CONTEXT_SCORE_THRESHOLD=7
MAX_RETRIEVAL_ITERATIONS=3
DOCLING_OCR_ENABLED=true
```

### 3.4 Map domains in Dokploy

In the Dokploy app's **Domains** tab — one row per service. Dokploy injects the corresponding Traefik labels.

| Domain                   | Service        | Container port |
| ------------------------ | -------------- | -------------- |
| `api.example.com`        | `api`          | `8000`         |
| `auth.example.com`       | `logto`        | `3001`         |
| `auth-admin.example.com` | `logto`        | `3002`         |
| `traces.example.com`     | `langfuse-web` | `3000`         |
| `files.example.com`      | `minio`        | `9000`         |

Enable Let's Encrypt on each. Trigger the deploy. Watch logs — `migrate` runs `alembic upgrade head` against RDS once and exits 0.

> **No host port `3000` conflict.** The prod overlay strips every `ports:` mapping from the base compose, so Dokploy's own admin UI on host port 3000 keeps working. Langfuse-web's `3000` is reachable only inside the Docker network.

## 4. Post-deploy onboarding

### 4.1 Logto admin

Open `https://auth-admin.example.com` and create the admin account, then in the admin console:

1. **API resources** → **Create**
   - Name: `agentic-rag`
   - Identifier: must equal `LOGTO_RESOURCE` (e.g. `https://api.example.com`)
2. **Applications** → **Create application** → **Machine-to-machine**
   - Name: `agentic-rag-m2m`
   - Attach the API resource you just created
3. Copy the M2M app's **App ID** and **App Secret** into Dokploy env (`LOGTO_APP_ID`, `LOGTO_APP_SECRET`). Redeploy `api` + `worker` so they pick the new values up.
4. **Organizations** → **Create organization** → use the auto-assigned ID as `tenant_id` downstream.
5. In the org's **Machine-to-machine apps** tab, add `agentic-rag-m2m`. Without this the M2M token won't carry an `organization_id` claim and `/chat/stream` returns 403.

### 4.2 Tenant + Qdrant collection

Tenant rows still need to be seeded. Qdrant collections are created lazily on first ingest now (see `qdrant_client.ensure_tenant_collection`), so the explicit `create_collections.py` step is optional.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  exec api uv run python scripts/seed_tenant.py \
  --logto-org-id <ORG_ID> --name "<Tenant display name>"
```

### 4.3 Langfuse first run

Open `https://traces.example.com`. Sign in with `LANGFUSE_INIT_USER_EMAIL` + `LANGFUSE_INIT_USER_PASSWORD`. Org + project + API keys are auto-created from the `LANGFUSE_INIT_*` env vars at first start — no clicks required.

### 4.4 MinIO console (optional)

The MinIO container exposes its admin console at `:9001` _internally_. To inspect uploads, either map a sixth subdomain (`storage.example.com → minio:9001`) or `docker exec` into the container.

## 5. Smoke test

```bash
# Mint a token
TOKEN=$(curl -s -X POST https://auth.example.com/oidc/token \
  -d grant_type=client_credentials \
  -d resource=https://api.example.com \
  -d organization_id=<ORG_ID> \
  -u "$LOGTO_APP_ID:$LOGTO_APP_SECRET" | jq -r .access_token)

curl https://api.example.com/health
# {"status": "ok"}

# Ingest a PDF
curl -X POST https://api.example.com/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@my.pdf;type=application/pdf"

# Chat (SSE stream)
curl --no-buffer -X POST https://api.example.com/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is in this document?"}'
```

`https://traces.example.com` — every `/chat/stream` request shows up as a trace tree with per-node tokens + cost.

## 6. Updates

Standard flow: `git push origin main` → Dokploy auto-deploys. To roll back:

```bash
# On the Dokploy server:
git fetch --tags
git checkout <tag-or-sha>
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Migrations are forward-only — if you roll back across a schema change, run `alembic downgrade` against RDS first or restore RDS from a snapshot.

## 7. Common ops

```bash
# Tail logs
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api worker

# Restart after env change (compose `restart` does NOT re-read .env — must use `up -d`)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --force-recreate api worker

# Re-run migrations after a schema PR
docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm migrate

# Read-only Postgres shell (RDS)
psql "postgresql://<user>:<pw>@<rds-host>:5432/agentic_rag?sslmode=require"

# Flush semantic cache (after threshold change)
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec redis \
  redis-cli -n 2 FLUSHDB

# Recreate a tenant's Qdrant collection (e.g. after embedding model change)
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  exec api uv run python scripts/create_collections.py --tenant-id <ORG_ID> --recreate
```

## 8. Migrating MinIO → AWS S3

The S3 client is endpoint-agnostic. To switch:

1. Create an S3 bucket + IAM user with `PutObject` / `GetObject` / `DeleteObject` on it.
2. In Dokploy env: unset `S3_ENDPOINT_URL` (or set it to empty), set `S3_REGION`, replace `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` with the IAM access key/secret, set `S3_PUBLIC_URL=https://<bucket>.s3.<region>.amazonaws.com`.
3. Optionally `aws s3 sync` existing MinIO contents into the bucket.
4. Redeploy `api` + `worker`. Remove the `minio` service from the compose stack.

## 9. Troubleshooting

| Symptom                                                            | Likely cause                                                                                                                                                                          |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docker compose config` errors with `<VAR> must be set in prod`    | A required secret is missing in Dokploy env. Re-check section 3.3.                                                                                                                    |
| Deploy fails: `password authentication failed for user "postgres"` | RDS connection string wrong — double check `DATABASE_URL`, `LOGTO_DB_URL`, `LANGFUSE_DATABASE_URL`. All three must be reachable.                                                      |
| Logto crashloops with `permission denied to create role`           | RDS user lacks `CREATE ROLE`. Grant it (or use the master user).                                                                                                                      |
| Logto / Langfuse can't reach RDS but psql from host works          | Containerized Node `pg` doesn't trust the AWS RDS CA bundle. The base compose sets `NODE_TLS_REJECT_UNAUTHORIZED=0` on those services — verify it survived your overlay edits.        |
| `/chat/stream` returns 401                                         | Logto access token expired (1h TTL) or `aud` mismatch. `LOGTO_RESOURCE` must equal the API resource Identifier in Logto admin.                                                        |
| `/chat/stream` returns 403                                         | Token has no `organization_id` claim. Add the M2M app to the org's "Machine-to-machine apps" (4.1 step 5).                                                                            |
| Langfuse traces don't appear                                       | `LANGFUSE_PUBLIC_KEY` / `SECRET_KEY` mismatch between api/worker env and `LANGFUSE_INIT_PROJECT_*`. They must be identical.                                                           |
| Logto admin console redirect loop                                  | `LOGTO_ENDPOINT` / `LOGTO_ADMIN_ENDPOINT` don't match the actual public URLs (Traefik domains).                                                                                       |
| `/documents/{id}/url` returns a URL the browser can't open         | `S3_PUBLIC_URL` not Traefik-routed. Confirm the `files.<domain>` row in 3.4 and that DNS is live.                                                                                     |
| Worker crashloops with `Event loop is closed`                      | Stale connection in the SQLAlchemy pool from a prior worker process. Restart worker; pool gets disposed in the new process.                                                           |
| `/ingest` succeeds but status stays pending forever                | Worker isn't running. `docker compose ps` should show `agentic-rag-worker` Up.                                                                                                        |
| Ingestion crashes in OCR                                           | Set `DOCLING_OCR_ENABLED=false` if you don't need scanned-PDF support. RapidOCR's first-use model download sometimes fails behind firewalls; bundled `tesseract-ocr` is the fallback. |
