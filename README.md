# mini-lang — a LangGraph-style agent runtime, built from scratch

A from-scratch implementation of the core ideas behind LangGraph in **pure Python**
(asyncio · dataclasses · pydantic · FastAPI · queues/events · retry+timeout), then
extended with the production stack a senior ML engineer is expected to reason about:
state checkpointing, conversation storage, tracing, distributed events, vector RAG,
eval tracking, and horizontal scaling on Kubernetes.

**Design principle:** every external dependency sits behind a Protocol with an
in-memory/mock default, so the whole system — and the entire test suite — runs with
**zero API keys and zero services**. Flip an env var to swap in a real backend.

## Concept → implementation → production

| LangGraph concept | Mini implementation | Production adapter |
|---|---|---|
| State | `AgentState` ([state.py](src/mini_lang/state.py)) | — |
| Nodes | `Node` + Planner/Tool/Retriever/Evaluator/Answer ([nodes/](src/mini_lang/nodes/)) | — |
| Edges / routing | `Graph` + `Router`/`RouteMap` ([graph.py](src/mini_lang/graph.py), [router.py](src/mini_lang/router.py)) | — |
| Parallel branches | `asyncio.gather` fan-out/join in `Graph.astream` | — |
| Checkpointing | `InMemoryCheckpointer` ([checkpoint/](src/mini_lang/checkpoint/)) | **Redis** |
| Conversation storage | `InMemoryConversationStore` ([persistence/](src/mini_lang/persistence/)) | **PostgreSQL** (SQLAlchemy async) |
| Tool calling | `ToolNode` + registry ([tools/](src/mini_lang/tools/)) | — |
| RAG | `InMemoryVectorStore` ([rag/](src/mini_lang/rag/)) | **Qdrant** |
| Evaluation | `EvaluatorNode` | **MLflow** |
| Async execution | `asyncio` throughout | — |
| Distributed events / queue | `InMemoryEventBus` ([events/](src/mini_lang/events/)) | **Kafka** (Redpanda) |
| Tracing | no-op `span` ([observability/](src/mini_lang/observability/)) | **OpenTelemetry** → Jaeger |
| API serving | **FastAPI** + SSE ([service/app.py](src/service/app.py)) | — |
| Reliability | `with_retry` + `with_timeout` ([reliability.py](src/mini_lang/reliability.py)) | — |
| UI | **Gradio** ([ui/gradio_app.py](src/ui/gradio_app.py)) | — |
| Horizontal scaling | — | **Kubernetes** + HPA ([infra/k8s/](infra/k8s/)) |

## The agent graph

```
                       ┌─ tool ─────┐
   planner ─ route ───┤              ├─ answer ─ evaluator ─┬─ score<thr ─→ planner
                       └─ retriever ─┘                      └─ END
```

The planner picks a route (`tool`, `retrieve`, `both` in parallel, or `answer`); the
evaluator scores the answer and loops back to the planner while the score is below
threshold (bounded by `AGENT_MAX_ITERATIONS`).

## Quick start (no infra, no keys)

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Use the Python version pinned in `.python-version`; uv can install it for you.

```bash
git clone https://github.com/wiwang/dragon_agent_core.git
cd dragon_agent_core
uv python install
uv sync --locked --extra dev --extra service
uv run pytest -q
uv run python -m mini_lang.demo "explain the refund policy and add 2 + 2"   # CLI trace
uv run uvicorn service.app:app --port 8000 # API
```

With the API running, open another terminal in the checkout and start the UI.

```bash
uv run python -m ui.gradio_app             # Gradio UI at http://localhost:7860
```

### API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | run synchronously, return the result |
| POST | `/chat/async` | enqueue on the bus; a worker processes it |
| POST | `/chat/stream` | SSE stream of node-by-node events |
| GET | `/runs/{thread}` | checkpointed state history |
| GET | `/runs/by-id/{id}` | poll an async run's result |
| GET | `/conversations/{thread}` | durable run history |
| GET | `/healthz` | liveness + queue depth |

## Running on real infra

```bash
make infra-up                              # redis, postgres, qdrant, redpanda, jaeger, mlflow
cp .env.example .env                        # then uncomment the backends you want
uv sync --extra dev --extra service --extra infra --extra obs
```

Flip env vars (see [.env.example](.env.example)) and the *same code* runs on real
backends — e.g. `AGENT_CHECKPOINTER=redis`, `AGENT_VECTORSTORE=qdrant`,
`AGENT_EVENTBUS=kafka`, `AGENT_TRACING_ENABLED=true`, `AGENT_MLFLOW_ENABLED=true`.

- Jaeger UI → http://localhost:16686 · MLflow UI → http://localhost:5001 · Qdrant → http://localhost:6333

Integration tests against live infra:

```bash
RUN_INFRA_TESTS=1 uv run pytest tests/test_integration_infra.py -q
```

## Kubernetes

```bash
docker build -f infra/Dockerfile -t agentic-agent:latest .   # one image: api/worker/ui
kind create cluster && kind load docker-image agentic-agent:latest
kubectl apply -f infra/k8s
kubectl -n agentic port-forward svc/ui 7860:7860
```

The **worker** Deployment consumes the Kafka bus in a shared consumer group, so the
**HPA** ([06-hpa.yaml](infra/k8s/06-hpa.yaml)) scales it on CPU under load. The API runs
`AGENT_API_INPROC_WORKERS=0` and delegates async runs to the worker pods. (HPA needs
`metrics-server` installed in the cluster.)

## Layout

```
src/mini_lang/   core framework (state, graph, nodes, llm, tools, rag, checkpoint,
                 persistence, events, observability, eval) + agent.py builder
src/service/     FastAPI app, queue worker, runner, dependency factories
src/ui/          Gradio chat UI
infra/           docker-compose, Dockerfile, k8s manifests
tests/           hermetic suite (mock LLM + in-memory) + opt-in infra integration tests
```
