# CricOptima - Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p ml_models data

# Expose ports
EXPOSE 8000 8501

# Default: API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
