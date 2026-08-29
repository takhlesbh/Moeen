.PHONY: dev stop test lint eval docker clean install discord

install:
	cd packages/core && uv sync
	cd packages/ui && npm install

dev:
	@echo "Starting Open Executive..."
	@cd packages/core && uv run uvicorn openexecutive.api.main:app --reload --port 8000 &
	@cd packages/ui && npm run dev

stop:
	@lsof -ti :8000 -ti :3000 2>/dev/null | xargs kill -9 2>/dev/null || true
	@echo "Stopped."

test:
	cd packages/core && uv run pytest tests/ -v --tb=short

lint:
	cd packages/core && uv run ruff check openexecutive/ && uv run mypy openexecutive/

# No --scenarios: the packaged loader (openexecutive/evals/scenarios.py) is the
# single source of discovery and resolves the directory itself. The old flag
# pointed at evals/scenarios/, which has not existed for some time — the runner
# then found 0 scenarios and still exited 0.
eval:
	cd packages/core && uv run python ../../evals/run_evals.py \
		--output ../../evals/results/

# Validate inventory + knowledge store and write a run manifest without making
# a single model call. Run this before spending money on a baseline.
eval-preflight:
	cd packages/core && uv run python ../../evals/run_evals.py \
		--output ../../evals/results/ --preflight-only

docker:
	docker compose -f docker/docker-compose.yml up --build

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf packages/core/.venv packages/core/.mypy_cache packages/core/.ruff_cache
	rm -rf packages/ui/node_modules packages/ui/.next

discord:
	cd packages/core && uv run python -m openexecutive.integrations.discord_bot

seed-knowledge:
	cd packages/core && uv run python -c "from openexecutive.knowledge.loader import seed_builtin_knowledge; import asyncio; asyncio.run(seed_builtin_knowledge())"
