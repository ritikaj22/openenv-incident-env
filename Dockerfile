FROM python:3.11-slim

WORKDIR /app

# Install system deps (needed for healthcheck)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure package structure
RUN mkdir -p env graders && touch env/__init__.py graders/__init__.py

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]