# Production Recommendation Engine

A two-stage recommendation engine using the Amazon Clothing, Shoes & Jewelry dataset.

## Architecture

1.  **Stage 1: Retrieval (PyTorch + FAISS)**
    A Two-Tower Neural Network embeds users and items into a shared vector space.
    FAISS provides sub-millisecond Approximate Nearest Neighbor (ANN) search to quickly retrieve the top 200 candidates from millions of items.
2.  **Stage 2: Ranking (LightGBM)**
    A gradient boosted tree classifier takes the 200 candidates and scores them using rich, non-linear cross-features.
3.  **Serving (FastAPI + Redis)**
    An asynchronous FastAPI backend handles requests. User profiles are cached in Redis for fast retrieval. Model inference is offloaded to a thread pool to prevent blocking the event loop.

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Data Pipeline
```bash
# Downloads data, cleans it, engineers features, and creates train/test splits
python scripts/run_preprocessing.py
```

### 3. Model Training
```bash
# Train the PyTorch Two-Tower model
python scripts/train_retrieval.py

# Build the FAISS index for fast retrieval
python scripts/build_faiss_index.py

# Train the LightGBM ranker
python scripts/train_ranker.py
```

### 4. Serving (Docker)
```bash
# Start Redis and the FastAPI application
docker-compose up -d

# Populate Redis with user/item features
python scripts/populate_redis.py
```

### 5. Test API
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "A1234", "num_results": 10}'
```

## Tech Stack
*   **Data Processing:** pandas, pyarrow
*   **Deep Learning:** PyTorch
*   **Vector Database:** FAISS
*   **Machine Learning:** LightGBM, scikit-learn
*   **API Framework:** FastAPI, Uvicorn, Pydantic
*   **Caching:** Redis (redis.asyncio)
*   **Containerization:** Docker, Docker Compose
