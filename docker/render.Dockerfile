# Etapas de render y espejado del orquestador (splatsim/cloud): la de entrenamiento más el rasterizador.
# Se construye en GitHub Actions (.github/workflows/images.yml):
#     docker build -f docker/render.Dockerfile --build-arg BASE=splatsim-train -t splatsim-render .
ARG BASE=ghcr.io/benjaminucn/splatsim-train:latest

# El toolkit de CUDA pesa varios GB y solo hace falta para compilar: las extensiones salen como wheels.
FROM ${BASE} AS build
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends g++ libxml2 make swig \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run -o /tmp/cuda.run \
    && sh /tmp/cuda.run --silent --toolkit --toolkitpath=/opt/cuda --no-opengl-libs --no-man-page --no-drm \
    && rm /tmp/cuda.run
COPY submodules/gaussian-splatting-wrapper/gaussian_splatting/submodules/diff-gaussian-rasterization /src/diff-gaussian-rasterization
COPY submodules/simple-knn /src/simple-knn
# Sin GPU al construir, torch no puede elegir la arquitectura: se compila para las que ofrecen los pods.
# El setup.py de ghalton pide libc++, que es de clang; con g++ compila igual sin esa opción.
# rasterizer_impl.h usa uint32_t sin incluirlo, y este compilador no lo trae por otro lado.
RUN sed -i '1i #include <cstdint>' /src/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h \
    && uv pip install --python /opt/venv/bin/python pip setuptools wheel \
    && /opt/venv/bin/python -m pip download --no-deps --no-binary :all: -d /src ghalton==0.6.2 \
    && tar xzf /src/ghalton-0.6.2.tar.gz -C /src \
    && sed -i "s/\[\"-stdlib=libc++\"\]/[]/; s/\['-stdlib=libc++'\]/[]/" /src/ghalton-0.6.2/setup.py \
    && TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0+PTX" CUDA_HOME=/opt/cuda PATH=/opt/cuda/bin:$PATH \
        /opt/venv/bin/python -m pip wheel --no-deps --no-build-isolation \
        -w /wheels /src/diff-gaussian-rasterization /src/simple-knn pybullet==3.2.7 /src/ghalton-0.6.2

FROM ${BASE}
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 libsm6 libx11-6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
COPY requirements-render.txt requirements-train.txt /tmp/
RUN uv pip install --python /opt/venv/bin/python --index-strategy unsafe-best-match --find-links /wheels \
        -c /tmp/requirements-train.txt -r /tmp/requirements-render.txt \
    && uv pip install --python /opt/venv/bin/python --no-deps /wheels/diff_gaussian_rasterization*.whl /wheels/simple_knn*.whl \
    && rm -rf /wheels

COPY . .
ENV SPLATSIM_HEADLESS=1 \
    PYTHONPATH=/app:/app/submodules/gaussian-splatting-wrapper:/app/submodules/gaussian-splatting-wrapper/gaussian_splatting:/app/submodules/pybullet-URDF-models
