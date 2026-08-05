# Dockerfile for Web-Intelligence-Agent (Python sidecar)
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency specifications
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Copy application files
COPY . .

# Default port. Render will overwrite PORT env and bind to 0.0.0.0
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
