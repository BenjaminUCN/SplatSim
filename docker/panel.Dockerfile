# Panel del orquestador (scripts/xarm5/cloud_server.py), para correrlo en cualquier PC sin torch ni CUDA:
# solo encola, alquila pods en Vast y mueve archivos con rclone. Se usa con docker/panel.compose.yml.
#     docker build -f docker/panel.Dockerfile -t splatsim-panel .
FROM python:3.12-slim AS rclone

# La 1.60 de Ubuntu falla contra R2 (501 Not Implemented); esta es la misma versión del lab y de los pods.
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends ca-certificates curl unzip \
    && curl -fsSL https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip -o /tmp/rclone.zip \
    && unzip -j /tmp/rclone.zip '*/rclone' -d /out && chmod 755 /out/rclone

FROM python:3.12-slim

COPY --from=rclone /out/rclone /usr/local/bin/rclone

# La CLI de Vast en su propio venv, igual que su instalador: sus dependencias no chocan con las del panel.
RUN python -m venv /opt/vastai \
    && /opt/vastai/bin/pip install --no-cache-dir -q vastai==1.11.0 \
    && ln -s /opt/vastai/bin/vastai /usr/local/bin/vastai

WORKDIR /app
COPY requirements-panel.txt ./
RUN pip install --no-cache-dir -q -r requirements-panel.txt

COPY . .

# El contenedor corre con el usuario y el HOME del anfitrión (ver el compose), así que las rutas de la
# cola (~/data/..., ~/lerobot/...) son las mismas dentro y fuera. Ese HOME solo tiene las carpetas
# montadas: lo demás que se escriba en caché va a /tmp.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/.cache
EXPOSE 8083

# Sin --no-update-registry reescribiría configs/registry.yaml dentro de la imagen, donde nadie lo lee.
ENTRYPOINT ["python", "scripts/xarm5/cloud_server.py", "--no-update-registry"]
