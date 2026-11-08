FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim
WORKDIR /app
COPY pyproject.toml /app/
COPY uv.lock /app/
RUN uv sync --locked --no-install-project
COPY server.py /app/
COPY glossary.json /app/
