#!/bin/bash
set -e

echo "Starting Ollama..."
ollama serve &
until curl -s http://127.0.0.1:11434/api/tags > /dev/null; do sleep 1; done

echo "Starting FastAPI backend..."
cd /home/user/app/backend
uvicorn main:app --host 127.0.0.1 --port 8000 &
until curl -s http://127.0.0.1:8000/ > /dev/null; do sleep 1; done

echo "Starting Gradio frontend..."
cd /home/user/app
exec python gradio_frontend.py
