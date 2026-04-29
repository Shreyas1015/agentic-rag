# Production deploy via Dokploy

This is the runbook for deploying agentic-rag to a Dokploy-managed server. Same `docker-compose.yml` Dokploy reads, plus the `docker-compose.prod.yml` overlay that strips host port mappings, requires every secret to be set, and persists uploads to a named volume.

> **Phase 1 + 2 + 3 are the deployable artifact** — Phase 3 just adds the production overlay + this runbook. The `phase-2-shipped` tag (or any commit on `main`) is fine to deploy.

---

## 1. Prerequisites

- A server with Dokploy installed and running (Dokploy's docs: https://docs.dokploy.com).
- A domain pointing at the server. Three subdomains are typical:
  - `api.example.com`         → FastAPI (`/health`, `/ingest`, `/chat/stream`, `/feedback`)
  - `auth.example.com`        → Logto OIDC (`:3001` internally)
  - `auth-admin.example.com`  → Logto admin console (`:3002` internally)
  - `traces.example.com`      → Langfuse UI + ingest (`:3000` internally)
- An OpenRouter API key with budget for production traffic.
- ~16 GB RAM, ~10 GB disk for the data volumes (postgres + qdrant + clickhouse + minio + uploads + HF model cache).

## 2. First-time deploy

### 2.1 Clone & generate secrets

In a workstation shell (not the server) so secrets never live in your terminal history on a shared host:

```bash
git clone https://github.com/Shreyas1015/agentic-rag.git
cd agentic-rag

# Generate all the secrets at once. Save the output somewhere secure.
echo "LOGTO_SECRET_VAULT_KEK=$(openssl rand -base64 32)"
echo "LANGFUSE_SALT=$(openssl rand -hex 32)"
echo "LANGFUSE_ENCRYPTION_KEY=$(openssl rand -hex 32)"
echo "LANGFUSE_NEXTAUTH_SECRET=$(openssl rand -hex 32)"
echo "LANGFUSE_INIT_USER_PASSWORD=$(openssl rand -base64 24)"   # admin login
echo "LANGFUSE_PUBLIC_KEY=pk-lf-$(openssl rand -hex 16)"
echo "LANGFUSE_SECRET_KEY=sk-lf-$(openssl rand -hex 32)"
echo "POSTGRES_PASSWORD=$(openssl rand -base64 24)"
```

### 2.2 Production `.env`

In Dokploy's project settings, paste the env vars below — replacing every `<...>` placeholder. The compose overlay refuses to start if any required secret is unset (`${VAR:?msg}` pattern), so you'll see clear errors at boot if something is missing.

```ini
# ── App ──────────────────────────────────────────
APP_NAME=agentic-rag
APP_ENV=prod
DEBUG=false

# ── OpenRouter ───────────────────────────────────
OPENROUTER_API_KEY=<sk-or-v1-...>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_APP_REFERER=https://api.example.com
OPENROUTER_APP_TITLE=agentic-rag

# Models — defaults are sensible; override if your eval scores demand it.
LLM_MODEL_CLASSIFY=google/gemini-2.5-flash
LLM_MODEL_DECOMPOSE=google/gemini-2.5-flash
LLM_MODEL_ASSESS=google/gemini-2.5-flash
LLM_MODEL_REFORMULATE=google/gemini-2.5-flash
LLM_MODEL_GENERATE=openai/gpt-4o
LLM_MODEL_LONG_CONTEXT=google/gemini-2.5-pro
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIMS=1536
LONG_CONTEXT_BYPASS_TOKEN_BUDGET=800000

# ── Logto (public URLs behind Traefik) ───────────
LOGTO_ENDPOINT=https://auth.example.com
LOGTO_ADMIN_ENDPOINT=https://auth-admin.example.com
LOGTO_RESOURCE=https://api.example.com
LOGTO_APP_ID=<from the M2M app you'll create in step 4>
LOGTO_APP_SECRET=<from the M2M app>
LOGTO_SECRET_VAULT_KEK=<from step 2.1>

# ── Langfuse (public URL behind Traefik) ─────────
LANGFUSE_HOST=http://langfuse-web:3000     # internal only — api hits this
LANGFUSE_NEXTAUTH_URL=https://traces.example.com
LANGFUSE_PUBLIC_KEY=<from step 2.1>
LANGFUSE_SECRET_KEY=<from step 2.1>
LANGFUSE_SALT=<from step 2.1>
LANGFUSE_ENCRYPTION_KEY=<from step 2.1>
LANGFUSE_NEXTAUTH_SECRET=<from step 2.1>
LANGFUSE_INIT_USER_EMAIL=<your admin email>
LANGFUSE_INIT_USER_PASSWORD=<from step 2.1>
LANGFUSE_INIT_USER_NAME=Admin
LANGFUSE_INIT_ORG_ID=agentic-rag-prod
LANGFUSE_INIT_ORG_NAME=Agentic RAG
LANGFUSE_INIT_PROJECT_ID=agentic-rag
LANGFUSE_INIT_PROJECT_NAME=agentic-rag

# ── PostgreSQL (internal — Dokploy talks to it via service name) ──
POSTGRES_USER=raguser
POSTGRES_PASSWORD=<from step 2.1>
POSTGRES_DB=ragdb
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_HOST_PORT=5432

# ── Redis (internal) ─────────────────────────────
REDIS_URL=redis://redis:6379
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
SEMANTIC_CACHE_REDIS_URL=redis://redis:6379/2
CACHE_SIMILARITY_THRESHOLD=0.85
CACHE_TTL_SECONDS=86400

# ── Qdrant (internal) ────────────────────────────
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# ── Agent tuning ─────────────────────────────────
CONTEXT_SCORE_THRESHOLD=7
MAX_RETRIEVAL_ITERATIONS=3
DOCLING_OCR_ENABLED=true        # tesseract is in the image; safe to enable in prod
```

### 2.3 Point Dokploy at the compose files

In Dokploy's app config, set the compose command to apply both files:

```
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Then in Dokploy's domain settings, route:
- `api.example.com`        → service `api`, port `8000`
- `auth.example.com`       → service `logto`, port `3001`
- `auth-admin.example.com` → service `logto`, port `3002`
- `traces.example.com`     → service `langfuse-web`, port `3000`

Dokploy's Traefik handles TLS termination via Let's Encrypt — enable that on each domain.

Trigger the first deploy. Watch the logs in the Dokploy UI; the `migrate` and `langfuse-db-create` / `logto-db-create` services run once and exit 0.

## 3. Post-deploy: onboarding

### 3.1 Logto admin (one-time)

Open `https://auth-admin.example.com` in a browser. Create the admin account, then in the admin console:

1. Sidebar → **API resources** → **Create API resource**
   - Name: `agentic-rag`
   - Identifier: must match `LOGTO_RESOURCE` in your env (e.g. `https://api.example.com`)
2. Sidebar → **Applications** → **Create application** → **Machine-to-machine**
   - Name: `agentic-rag-m2m`
   - On the next screen attach the API resource you just created
3. Copy the M2M app's **App ID** and **App Secret** into Dokploy env: `LOGTO_APP_ID`, `LOGTO_APP_SECRET`. Trigger a redeploy of `api` + `worker` so they pick up the new values.
4. Sidebar → **Organizations** → **Create organization**
   - Use the org's auto-assigned ID as `tenant_id` everywhere downstream. (Or create the org via the Management API.)
5. In the org's **Machine-to-machine apps** tab, **Add applications** → pick `agentic-rag-m2m`. Without this, the M2M app cannot mint tokens with `organization_id` claims and `/chat/stream` will 403.

### 3.2 Tenant + Qdrant collection

```bash
# Once on the server (Dokploy provides shell access via the UI):
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  exec api uv run python scripts/seed_tenant.py \
  --logto-org-id <ORG_ID> --name "<Tenant display name>"

docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  exec api uv run python scripts/create_collections.py \
  --tenant-id <ORG_ID>
```

### 3.3 Langfuse first run

Open `https://traces.example.com`. Sign in with the email + password you set as `LANGFUSE_INIT_USER_*`. The org and project were auto-created from the env vars at first start; no clicks required. Confirm the project's **Public key** + **Secret key** match your env (they will — that's what `LANGFUSE_INIT_PROJECT_*` does).

## 4. Smoke test from the public URL

```bash
# Mint a token
TOKEN=$(curl -s -X POST https://auth.example.com/oidc/token \
  -d grant_type=client_credentials \
  -d resource=https://api.example.com \
  -d organization_id=<ORG_ID> \
  -u "$LOGTO_APP_ID:$LOGTO_APP_SECRET" | jq -r .access_token)

# Health
curl https://api.example.com/health
# {"status": "ok"}

# Ingest a PDF
curl -X POST https://api.example.com/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@my.pdf;type=application/pdf"

# Chat
curl --no-buffer -X POST https://api.example.com/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is in this document?"}'
```

Open `https://traces.example.com` — every `/chat/stream` request shows up as a trace tree with per-node tokens + cost.

## 5. Updates

The standard flow is just `git push origin main` followed by Dokploy's auto-deploy. If a release breaks something, roll back to the prior tag:

```bash
# On the Dokploy server:
git fetch --tags
git checkout phase-2-shipped     # or any earlier tag/SHA
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Migrations are forward-only — if you roll back across a schema change, run `alembic downgrade` manually first or restore from Dokploy's volume backup.

## 6. Common ops

```bash
# Tail logs (single service)
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api worker

# Restart a service after env change
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --force-recreate api

# Re-run migrations after a schema PR
docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm migrate

# Postgres shell (read-only inspection — be careful)
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec postgres \
  psql -U raguser -d ragdb

# Redis cache flush (after switching cache threshold or ingestion contract)
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec redis \
  redis-cli -n 2 FLUSHDB

# Drop + recreate a tenant's Qdrant collection (e.g. after embedding model change)
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  exec api uv run python scripts/create_collections.py \
  --tenant-id <ORG_ID> --recreate
```

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `docker compose config` errors with `<VAR> must be set in prod` | A required secret is missing in Dokploy env. Check section 2.2. |
| `/chat/stream` returns 401 | Logto access token expired (1h TTL) or aud mismatch. Check `LOGTO_RESOURCE` matches the API resource identifier in Logto admin. |
| `/chat/stream` returns 403 | Token has no `organization_id` claim. Ensure the M2M app is added to the org's "Machine-to-machine apps" (3.1 step 5). |
| Langfuse traces don't appear | `LANGFUSE_PUBLIC_KEY` / `SECRET_KEY` mismatch between api/worker env and `LANGFUSE_INIT_PROJECT_*`. They must be identical (auto-onboarding uses the same values). |
| Logto admin console redirect loop | `LOGTO_ENDPOINT` / `LOGTO_ADMIN_ENDPOINT` don't match the actual public URLs (Traefik domains). |
| Worker crashloops with `Event loop is closed` | Stale connection in the SQLAlchemy pool from a prior worker process. Restart the worker; the pool gets disposed in the new process. |
| `/ingest` succeeds but `/ingest/status` stays pending forever | Worker isn't running. `docker compose ps` should show `agentic-rag-worker` Up. Logs in Dokploy will show the crash reason. |
| Ingestion task crashes in OCR | Set `DOCLING_OCR_ENABLED=false` if you don't need scanned-PDF support; the Chinese-host model RapidOCR pulls on first use sometimes fails behind firewalls. The bundled `tesseract-ocr` is a fallback if you do need OCR. |
