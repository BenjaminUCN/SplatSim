# Etapa de entrenamiento del orquestador (splatsim/cloud): sin submódulos de render.
# Se construye en GitHub Actions (.github/workflows/images.yml):
#     docker build -f docker/train.Dockerfile -t splatsim-train .
FROM ubuntu:24.04

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends ca-certificates curl ffmpeg gcc git libc6-dev unzip \
    && rm -rf /var/lib/apt/lists/*

# La 1.60 de Ubuntu falla contra R2 (501 Not Implemented); esta es la misma versión del lab.
RUN curl -fsSL https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip -o /tmp/rclone.zip \
    && unzip -j /tmp/rclone.zip '*/rclone' -d /usr/local/bin && chmod 755 /usr/local/bin/rclone && rm /tmp/rclone.zip

ENV UV_NO_CACHE=1 VENV=/opt/venv WORKDIR=/workspace PYTHONPATH=/app PATH=/root/.local/bin:$PATH
WORKDIR /app

# Las dependencias van en su propia capa: cambiar el código no vuelve a instalar ni descargar torch.
# runpod_setup.sh instala el repo como editable, así que aquí basta un paquete vacío.
COPY requirements-train.txt setup.py ./
COPY scripts/runpod_setup.sh scripts/runpod_setup.sh
RUN mkdir splatsim && touch splatsim/__init__.py && bash scripts/runpod_setup.sh

COPY . .

ENTRYPOINT ["/opt/venv/bin/python", "-u", "-m", "splatsim.cloud.runner"]
