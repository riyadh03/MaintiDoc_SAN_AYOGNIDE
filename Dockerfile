FROM python:3.12-slim

# System tools & libraries for PDF rendering, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install project dependencies into system Python
RUN uv pip install --system \
    pymupdf \
    pandas \
    numpy \
    pillow \
    tabulate \
    matplotlib \
    ipython \
    pydantic \
    llama-index-core \
    llama-index-embeddings-huggingface \
    sentence-transformers \
    qdrant-client \
    ollama \
    ipykernel \
    jupyterlab

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

