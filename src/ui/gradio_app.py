"""Gradio chat UI.

Talks to the FastAPI service over SSE (``/chat/stream``) and renders each graph
super-step as it arrives, then the final grounded answer. Run the API first::

    uv run uvicorn service.app:app --port 8000
    uv run python -m ui.gradio_app
"""

from __future__ import annotations

import json
import os

import gradio as gr
import httpx

SERVICE_URL = os.environ.get("AGENT_SERVICE_URL", "http://localhost:8000")


async def chat_fn(message: str, history: list, thread_id: str):
    """Stream node events from the service, yielding a live transcript."""
    trace_lines: list[str] = []
    answer = ""
    payload = {"query": message, "thread_id": thread_id or "ui"}

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{SERVICE_URL}/chat/stream", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = json.loads(line[len("data:") :].strip())
                node = data.get("node")
                if data["event"] in ("node", "parallel") and node:
                    label = node if isinstance(node, str) else " ‖ ".join(node)
                    trace_lines.append(
                        f"• **{label}** — route=`{data.get('route')}` "
                        f"score=`{data.get('score')}`"
                    )
                if data.get("answer"):
                    answer = data["answer"]
                trace = "\n".join(trace_lines)
                yield f"{trace}\n\n---\n**Answer:** {answer or '…'}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="mini-lang agent") as demo:
        gr.Markdown("# mini-lang agent\nA from-scratch LangGraph-style agent. Watch the graph think.")
        thread = gr.Textbox(value="ui", label="thread_id", scale=1)
        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[thread],
            examples=[["explain the refund policy"], ["compute 2 * 21"], ["what is 12 / 4 about shipping"]],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
