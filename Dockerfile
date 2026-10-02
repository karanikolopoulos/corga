FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface
WORKDIR /app

# deps first in their own cached layer (torch is the CPU build pinned in uv.lock)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

COPY corga ./corga
COPY conf ./conf
COPY data/corfu_dataset.json data/corfu_synthetic_rain.json ./data/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

# let any user id write to the cache volume
RUN useradd --uid 1000 corga && mkdir -p /app/.cache && chmod 777 /app/.cache
USER corga
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "corga"]
