FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/sklearn_model.joblib

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import joblib; joblib.load('/app/models/sklearn_model.joblib')" || exit 1

# Default command to run prediction
CMD ["python", "src/predict.py"]
