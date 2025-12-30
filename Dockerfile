# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
# Check if files are in ai-based-training subdirectory or root
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for Hugging Face Spaces
ENV PYTHONUNBUFFERED=1

# Create a startup script to run both FastAPI and Streamlit
RUN cat > /app/start.sh <<'SH' && chmod +x /app/start.sh
#!/bin/bash
set -e
echo "🟢 Starting FastAPI backend..."
cd /app
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 54300 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"
# Wait briefly for backend to be ready
sleep 5
echo "🟢 Starting Streamlit frontend..."
# Use PORT env var from Hugging Face Spaces if provided, otherwise default to 7860
PORT=${PORT:-7860}
export STREAMLIT_SERVER_PORT="$PORT"
# Run Streamlit headless and disable CORS so HF routing works
streamlit run frontend/app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true --server.enableCORS false
# Wait for background processes (uvicorn)
wait $BACKEND_PID
SH

# Expose Streamlit port
EXPOSE 7860

# Start both services
CMD ["/app/start.sh"]