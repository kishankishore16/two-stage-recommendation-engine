FROM python:3.12-slim

# Install system dependencies for FAISS and LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code + model artifacts + frontend
COPY . .

# Railway injects PORT; default to 8000 for local Docker use
ENV PORT=8000
EXPOSE ${PORT}

# Run FastAPI server — use shell form so $PORT is expanded at runtime
CMD uvicorn src.serving.main:app --host 0.0.0.0 --port $PORT
