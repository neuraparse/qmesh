FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY qmesh ./qmesh

RUN pip install --no-cache-dir -e .[service]

EXPOSE 8788

CMD ["uvicorn", "qmesh.service.app:app", "--host", "0.0.0.0", "--port", "8788"]
