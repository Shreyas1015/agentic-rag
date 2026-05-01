# Dokploy quickstart — agentic-rag

One-page Dokploy UI walkthrough. Full reference + troubleshooting in [DEPLOY.md](./DEPLOY.md).

## Before you click

- EC2 host with Dokploy installed (admin UI on `:3000`).
- Security group: `22`, `80`, `443`, `3000`.
- DNS: 5 subdomains pointing at the host — `api`, `auth`, `auth-admin`, `traces`, `files`.
- AWS RDS Postgres reachable, with three databases (`agentic_rag`, `logto`, `langfuse`) and a user that has `CREATE ROLE`.
- Secrets generated (see [DEPLOY.md §3.1](./DEPLOY.md#31-generate-secrets)).
- OpenRouter key.

## Steps

1. **Create Application** → **Docker Compose**.
2. **Source** → GitHub → repo `Shreyas1015/agentic-rag`, branch `main`.
3. **Compose file:** `server/docker-compose.yml`
   **Compose override:** `server/docker-compose.prod.yml`
4. **Environment** tab → paste the env block from [DEPLOY.md §3.3](./DEPLOY.md#33-production-env). Fill every `<...>`. Save.
5. **Domains** tab → add five rows (Let's Encrypt on each):

   | Domain                   | Service        | Port |
   |--------------------------|----------------|------|
   | `api.example.com`        | `api`          | 8000 |
   | `auth.example.com`       | `logto`        | 3001 |
   | `auth-admin.example.com` | `logto`        | 3002 |
   | `traces.example.com`     | `langfuse-web` | 3000 |
   | `files.example.com`      | `minio`        | 9000 |

6. **Deploy**. Watch logs — `migrate` runs once and exits 0. `langfuse-web`, `langfuse-worker`, `logto`, `api`, `worker` all reach steady state.
7. Open `https://auth-admin.example.com`. Follow [DEPLOY.md §4.1](./DEPLOY.md#41-logto-admin) to create the API resource + M2M app + organization. Paste M2M ID/secret into Dokploy env, redeploy `api` + `worker`.
8. Seed the tenant ([§4.2](./DEPLOY.md#42-tenant--qdrant-collection)).
9. Smoke test ([§5](./DEPLOY.md#5-smoke-test)).

## Common gotchas

- **Dokploy admin UI on host port 3000 conflicting with langfuse-web.** Doesn't happen: prod overlay strips every host port mapping. Langfuse's `3000` is internal only, reached through Traefik on `traces.<domain>`.
- **`<VAR> must be set in prod` at deploy time.** A required secret is missing in step 4. The variable name is in the error.
- **Logto crashloops with `permission denied to create role`.** RDS user lacks `CREATE ROLE`. Grant it or use the master user.
- **`/documents/{id}/url` returns a URL that won't open in the browser.** `S3_PUBLIC_URL` doesn't resolve, or the `files.<domain>` row in step 5 is missing.
- **`/chat/stream` returns 403.** M2M app isn't added to the org. Logto admin → Org → Machine-to-machine apps → add `agentic-rag-m2m`.
