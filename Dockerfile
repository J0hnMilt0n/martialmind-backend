# Dockerfile for MartialMind AI Backend
# Uses MediaPipe Pose for robust pose estimation (no external model downloads needed)

FROM python:3.9-slim

LABEL maintainer="Dojo Republic"
LABEL description="MartialMind AI - Performance Analysis and Injury Risk Detection"

WORKDIR /app

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/*

# Copy application code
COPY . .

# Create directory for temporary files
RUN mkdir -p /tmp/uploads

# Expose port
EXPOSE 8000

# Health check with start period for model download
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application with single worker
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]