#!/usr/bin/env bash
# Instala solo lo necesario para entrenar (sin submódulos de render) en un pod de RunPod:
#     git clone --depth 1 https://github.com/BenjaminUCN/SplatSim.git /workspace/SplatSim
#     bash /workspace/SplatSim/scripts/runpod_setup.sh
#
# Todo queda en WORKDIR (el disco de volumen, sobrevive a detener el pod). ~/data y ~/lerobot
# apuntan ahí para que sirvan las rutas de las configs; si ya existen, no se tocan.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-/workspace}"
VENV="${VENV:-$WORKDIR/venv}"

if ! command -v ffmpeg >/dev/null; then    # torchcodec usa las bibliotecas de FFmpeg del sistema
    apt-get update -qq
    apt-get install -y -qq ffmpeg
fi
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --allow-existing --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match -r "$REPO/requirements-train.txt"
uv pip install --python "$VENV/bin/python" --no-deps -e "$REPO"

for name in data lerobot; do
    mkdir -p "$WORKDIR/$name"
    [ -e "$HOME/$name" ] || ln -s "$WORKDIR/$name" "$HOME/$name"
done

echo "── Pod ──────────────────────────────────────────"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "Sin GPU visible"
echo "vCPU (límite cgroup): $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo '?')  ·  nproc: $(nproc)"
echo "RAM (límite cgroup): $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo '?')"
df -h /dev/shm "$WORKDIR" | sed 1d
"$VENV/bin/python" -c "import torch, torchcodec, lerobot, viser, wandb, tyro; print('torch', torch.__version__, '· CUDA', torch.cuda.is_available())"
echo "Listo. Panel: cd $REPO && $VENV/bin/python scripts/xarm5/train_server.py"
