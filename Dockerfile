# syntax=docker/dockerfile:1.7
#
# Image de production de l'API MedXRay.
#   docker build -t medxray-api:1.0.0 .
#   docker run --rm -p 8000:8000 medxray-api:1.0.0
#
# Multi-stage : l'étage `builder` installe les dépendances dans un venv, l'étage `runtime`
# ne copie que ce venv. Ni compilateur, ni cache pip, ni code d'entraînement dans l'image.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------- builder ---
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN python -m venv /opt/venv

# 1) PyTorch *CPU* depuis l'index officiel : la roue CUDA par défaut pèse ~2,5 Go de plus,
#    inutile sur un serveur sans GPU. Couche séparée => mise en cache tant que la version
#    ne change pas.
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/cpu

# 2) Dépendances d'exécution, versions figées par le fichier de contraintes.
COPY requirements/constraints.txt /tmp/constraints.txt
COPY pyproject.toml README.md /src/
COPY src /src/src
RUN pip install --constraint /tmp/constraints.txt /src

# ---------------------------------------------------------------------------- runtime ---
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="medxray-api" \
      org.opencontainers.image.description="Détection de pneumonie sur radiographie + Grad-CAM" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    MEDXRAY_MODEL_PATH=/app/models/model.pt \
    MEDXRAY_TORCH_THREADS=2 \
    OMP_NUM_THREADS=2

# Utilisateur non-root : si l'application est compromise, l'attaquant n'est pas root
# dans le conteneur.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
# Les poids font partie de l'image : une image = une version de modèle, immuable et
# traçable (sha256 exposé par GET /v1/model).
COPY --chown=app:app models/model.pt /app/models/model.pt

WORKDIR /app
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# Un seul worker par conteneur : on scale horizontalement (réplicas), et chaque worker
# chargerait sinon sa propre copie du modèle en mémoire.
CMD ["uvicorn", "--factory", "medxray.api.main:create_app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--no-access-log", "--proxy-headers", "--timeout-graceful-shutdown", "20"]
