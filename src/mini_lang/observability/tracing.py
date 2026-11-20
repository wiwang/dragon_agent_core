"""Tracing seam.

Phase 1 ships a no-op ``span`` so the graph engine can be instrumented without
any dependency on OpenTelemetry. Phase 5 swaps in a real OTel tracer via
:func:`setup_tracing` — the graph code never changes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

# Set by setup_tracing() in Phase 5. When None, span() is a no-op.
_tracer: Any = None


def setup_tracing(service_name: str = "mini-lang", endpoint: str | None = None) -> None:
    """Configure OpenTelemetry. Imported lazily so core stays dependency-free."""
    global _tracer
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    # A plain "http://host:port" endpoint to Jaeger's OTLP receiver is insecure (no TLS).
    insecure = bool(endpoint and endpoint.startswith("http://"))
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure) if endpoint else OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


def instrument_fastapi(app) -> None:
    """Auto-instrument a FastAPI app so each request is a root span."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Open a tracing span. No-op unless setup_tracing() has been called."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as sp:
        for key, value in attributes.items():
            sp.set_attribute(key, value)
        yield
