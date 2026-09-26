#!/bin/bash
# Starts SafeSpace AI with a public demo link. Press Ctrl+C to stop everything.
cd "$(dirname "$0")"
source .venv/bin/activate

cleanup() { echo "Stopping..."; kill 0; }
trap cleanup EXIT

echo "Starting backend (emergency calls disabled)..."
(cd backend && TWILIO_ACCOUNT_SID="" python main.py) &

echo "Starting frontend..."
python gradio_frontend.py &
until curl -s http://localhost:7860 > /dev/null; do sleep 2; done

echo "Starting public link..."
cloudflared tunnel --url http://localhost:7860 2>&1 | grep --line-buffered -o "https://[a-z0-9-]*\.trycloudflare\.com"
