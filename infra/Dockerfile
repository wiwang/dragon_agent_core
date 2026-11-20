# Multi-purpose image for the api, worker, and ui (the command is overridden
# per Deployment). Built once, run three ways.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy metadata + source, then install the package with the runtime extras.
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system ".[service,infra,obs]"

EXPOSE 8000 7860

# Default to the API; worker/ui Deployments override `command`.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
