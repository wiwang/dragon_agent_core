.PHONY: install test demo api worker ui infra-up infra-down fmt

install:          ## install all extras
	uv sync --extra dev --extra service --extra infra --extra obs --extra llm

test:             ## run the hermetic test suite (mock LLM, in-memory adapters)
	uv run pytest -q

demo:             ## run the agent once on the mock LLM and print the node trace
	uv run python -m mini_lang.demo "explain the refund policy and add 2 + 2"

api:              ## serve the FastAPI app
	uv run uvicorn service.app:app --reload --port 8000

worker:           ## run a standalone queue worker (for Kafka mode)
	uv run python -m service.worker

ui:               ## launch the Gradio UI (needs the API running)
	uv run python -m ui.gradio_app

infra-up:         ## start redis, postgres, qdrant, redpanda, jaeger, mlflow
	docker compose -f infra/docker-compose.yml up -d

infra-down:       ## stop infra
	docker compose -f infra/docker-compose.yml down
