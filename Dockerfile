# Multi-stage Dockerfile for MartialMind AI Backend - Optimized for size

# Stage 1: Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime stage - using smaller image
FROM python:3.9-slim

LABEL maintainer="Dojo Republic"
LABEL description="MartialMind AI - Performance Analysis and Injury Risk Detection"

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Ensure Python can find the packages
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/root/.local/lib/python3.9/site-packages

# Copy application code
COPY . .

# Create directory for temporary files
RUN mkdir -p /tmp/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]