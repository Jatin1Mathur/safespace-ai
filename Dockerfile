FROM python:3.14-slim

# System tools + Ollama
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates zstd \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://ollama.com/install.sh | sh \
 && pip install --no-cache-dir uv

# Hugging Face Spaces run the app as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    OLLAMA_MODELS=/home/user/.ollama/models \
    OLLAMA_HOST=127.0.0.1:11434 \
    UV_PYTHON_PREFERENCE=only-system \
    PYTHONUNBUFFERED=1 \
    PATH=/home/user/app/.venv/bin:$PATH
WORKDIR /home/user/app

# Download both AI models into the image (done once, at build time)
RUN ollama serve & sleep 8 \
 && ollama pull llama3.2 \
 && ollama pull alibayram/medgemma:4b

# Install Python packages
COPY --chown=user pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the app
COPY --chown=user . .
RUN chmod +x start.sh

EXPOSE 7860
CMD ["./start.sh"]
