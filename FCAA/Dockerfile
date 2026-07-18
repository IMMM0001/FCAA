FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY data/ ./data/
COPY results/ ./results/

# Create results directories
RUN mkdir -p /app/results/logs /app/results/figures

# Default command: run the comparison experiment
CMD ["python", "-m", "experiments.run_comparison"]
