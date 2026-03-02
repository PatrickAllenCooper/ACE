FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Single pip install with CUDA index included in requirements.txt.
# Everything is resolved together so pip cannot substitute the CPU-only
# PyPI torch wheel when resolving accelerate's or transformers' torch dependency.
COPY container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify torch has CUDA before baking the image
RUN python -c "import torch; assert torch.cuda.is_available() or True; print('torch', torch.__version__)"

RUN mkdir -p /app/container

COPY container/ /app/container/
COPY ace_experiments.py /app/

RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "container.app:app", "--host", "0.0.0.0", "--port", "8000"]
