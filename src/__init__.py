"""
Production Recommendation Engine — Amazon Clothing, Shoes & Jewelry

A two-stage recommendation system:
  Stage 1: Two-Tower Neural Network (PyTorch) + FAISS for candidate retrieval
  Stage 2: LightGBM for re-ranking

Served via FastAPI with Redis caching, containerized with Docker Compose.
"""
