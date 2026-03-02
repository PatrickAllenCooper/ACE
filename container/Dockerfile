FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install torch with CUDA 12.1 support as a separate layer so it is
# cached independently from application requirements. PyTorch wheels
# bundle the CUDA runtime libraries they need, so no system CUDA install
# is required - only the GPU driver on the host (provided by Azure).
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy and install remaining application requirements.
# torch is already installed above; pip will not reinstall it.
COPY container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/container

# Copy application code
COPY container/ /app/container/
COPY ace_experiments.py /app/

# Non-root user for security
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "container.app:app", "--host", "0.0.0.0", "--port", "8000"]
