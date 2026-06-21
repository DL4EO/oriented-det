# oriented-det Sanic inference (no OpenMMLab / MMCV / MMDet / MMRotate).
# DOTA / framework smoke image. Publish a DOTA run into deploy/example/app first.
#   docker build -f deploy/example/Dockerfile -t odet-example:latest .

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="oriented-det inference" \
      org.opencontainers.image.description="Sanic Tile Geo Process API: oriented aircraft detection (RGB tiles)" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ca-certificates \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /opt/oriented-det

# Install PyTorch first so changes to the library do not invalidate the layer.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
COPY deploy/example/requirements-deploy.txt ./deploy/example/requirements-deploy.txt
RUN pip install --no-cache-dir -r deploy/example/requirements-deploy.txt

# Install the library
COPY pyproject.toml README.md ./
COPY oriented_det ./oriented_det
RUN pip install --no-cache-dir -e .

# Sanic app (build context = framework repo root: docker build -f deploy/example/Dockerfile .)
COPY deploy/example/app ./deploy/example/app

# Create a user and set the permissions
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /opt/oriented-det

ENV PYTHONPATH=/opt/oriented-det

WORKDIR /opt/oriented-det/deploy/example/app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/api/v1/health || exit 1

ENTRYPOINT ["python3", "app.py"]
