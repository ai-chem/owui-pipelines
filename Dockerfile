FROM ghcr.io/open-webui/pipelines:main

# uv dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PIPELINES_REQUIREMENTS_PATH=/app/pipeline_requirements.txt

WORKDIR /app/

COPY ./pyproject.toml .
COPY ./.python-version .
COPY ./uv.lock .

RUN uv sync --locked

COPY pipelines/ ./pipelines/

RUN uv pip freeze > ${PIPELINES_REQUIREMENTS_PATH}
