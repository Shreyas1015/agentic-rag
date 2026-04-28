# agentic-rag — convenience targets for local dev.
# Tabs only (Make is sensitive). Works in Git Bash on Windows + Linux/macOS.

# Defaults — override at invocation: `make seed ORG_ID=foo NAME="My Project"`.
ORG_ID  ?= jzi706s5503k
NAME    ?= Local Dev Project

.PHONY: help up down restart logs logs-app ps build rebuild \
        migrate seed reseed test test-fast lint fmt fix \
        shell-api shell-worker shell-db redis-cli qdrant-ui \
        clean nuke

help:
	@echo "Common targets:"
	@echo "  make up              start the full stack (postgres, redis, qdrant, logto, migrate, api, worker)"
	@echo "  make down            stop and remove containers"
	@echo "  make logs            tail logs from all services"
	@echo "  make logs-app        tail just api + worker"
	@echo "  make build           build the agentic-rag-app image"
	@echo "  make rebuild         force a no-cache rebuild"
	@echo "  make migrate         run alembic upgrade head inside the api container"
	@echo "  make seed            seed tenant + Qdrant collection (ORG_ID=... NAME=...)"
	@echo "  make reseed          recreate Qdrant collection (drops existing data)"
	@echo "  make test            run pytest against the live stack"
	@echo "  make test-fast       skip slow tests (Docling + full agent runs)"
	@echo "  make lint / fmt      ruff check / ruff format + autofix"
	@echo "  make shell-api       bash inside the api container"
	@echo "  make shell-worker    bash inside the worker container"
	@echo "  make shell-db        psql inside the postgres container"
	@echo "  make clean           docker compose down (containers go, volumes stay)"
	@echo "  make nuke            docker compose down -v (drops ALL volumes — destructive)"

# ── Stack lifecycle ────────────────────────────────────────────────

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart api worker

logs:
	docker compose logs -f --tail=200

logs-app:
	docker compose logs -f --tail=200 api worker

ps:
	docker compose ps

build:
	docker compose build api

rebuild:
	docker compose build --no-cache api

# ── DB / fixtures ──────────────────────────────────────────────────

migrate:
	docker compose exec api uv run alembic upgrade head

seed:
	docker compose exec api uv run python scripts/seed_tenant.py --logto-org-id "$(ORG_ID)" --name "$(NAME)"
	docker compose exec api uv run python scripts/create_collections.py --tenant-id "$(ORG_ID)"

reseed:
	docker compose exec api uv run python scripts/create_collections.py --tenant-id "$(ORG_ID)" --recreate

# ── Quality gates ──────────────────────────────────────────────────

test:
	uv run pytest -v

test-fast:
	uv run pytest -v -m "not slow"

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

fix:
	uv run ruff check --fix .
	uv run ruff format .

# ── Shells ─────────────────────────────────────────────────────────

shell-api:
	docker compose exec api bash

shell-worker:
	docker compose exec worker bash

shell-db:
	docker compose exec postgres psql -U raguser -d ragdb

redis-cli:
	docker compose exec redis redis-cli

qdrant-ui:
	@echo "Qdrant dashboard: http://localhost:6333/dashboard"

# ── Cleanup ────────────────────────────────────────────────────────

clean:
	docker compose down

# Drops ALL named volumes — postgres, redis, qdrant, logto, model_cache.
# Destructive. Confirms manually so a stray Tab doesn't wipe a real env.
nuke:
	@echo "About to remove ALL data volumes (postgres + redis + qdrant + model cache)."
	@read -p "Type 'yes' to confirm: " ans && [ "$$ans" = "yes" ] || (echo aborted; exit 1)
	docker compose down -v
