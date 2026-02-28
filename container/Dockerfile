# Use PyTorch with CUDA for GPU support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/container
RUN mkdir -p /app/ace_experiments.py

# Copy application code
COPY container/ /app/container/
COPY ace_experiments.py /app/

# Create a non-root user for security (optional but recommended for Azure)
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["uvicorn", "container.app:app", "--host", "0.0.0.0", "--port", "8000"]
