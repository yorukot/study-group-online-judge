# syntax=docker/dockerfile:1.7

FROM tailscale/tailscale:latest AS tailscale

FROM node:22-alpine AS ui-build
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui ./
RUN npm run build

FROM python:3.14-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app


FROM base AS build

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM base AS judge

COPY --from=docker:29.4.0-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=tailscale /usr/local/bin/tailscale /usr/local/bin/tailscale

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends git openssh-client ca-certificates fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 judge \
    && useradd --uid 10001 --gid 10001 --create-home judge \
    && mkdir -p /data /home/judge/.cache/huggingface /home/judge/.cache/uv \
    && chown -R judge:judge /data /home/judge/.cache

COPY --from=build --chown=judge:judge /app/.venv /app/.venv
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN /app/.venv/bin/playwright install --with-deps --only-shell chromium \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ui-build --chown=judge:judge /ui/dist /app/ui/dist
COPY --chown=judge:judge ui/src/lab-names.json /app/lab-names.json

ENV PATH="/app/.venv/bin:${PATH}" \
    JUDGE_LAB_NAMES_PATH=/app/lab-names.json \
    JUDGE_UI_DIST=/app/ui/dist \
    JUDGE_DATABASE_PATH=/data/judge.db \
    HF_HOME=/home/judge/.cache/huggingface \
    UV_CACHE_DIR=/home/judge/.cache/uv \
    PYTHONUNBUFFERED=1

USER 10001:10001

CMD ["uvicorn", "judge.main:app", "--host", "0.0.0.0", "--port", "8000"]
