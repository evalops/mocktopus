# Multi-stage build for smaller image
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./
COPY src/ ./src/

# Install package
RUN pip install --no-cache-dir -e .

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed package from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/mocktopus /usr/local/bin/mocktopus

# Copy source code (for editable install)
COPY src/ ./src/
COPY pyproject.toml ./
COPY examples/ ./examples/

# Create directories for recordings and scenarios
RUN mkdir -p /data/recordings /data/scenarios

# Install runtime package
RUN pip install --no-cache-dir -e .

# Expose default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Default scenario location
ENV SCENARIO_PATH=/data/scenarios/scenario.yaml
ENV RECORDINGS_DIR=/data/recordings

# Run server by default
CMD ["mocktopus", "serve", "-s", "${SCENARIO_PATH}", "--host", "0.0.0.0"]