#!/bin/bash
# Limpieza de espacio que no pierde nada (25-nov). Pensado para el final de cada paso de una cola; se puede repetir.
#   bash scripts/xarm5/disk_upkeep.sh [--dry-run]
#
# 1. PNG de renders (~/data/bc_data/*_10hz con DONE) cuyo dataset LeRobot está completo: los verifica y los borra
#    (scripts/xarm5/prune_render_images.py). El revisor muestra los cuadros desde el dataset.
# 2. PNG de ensayos (~/data/policy_trials) a mp4 sin pérdida, verificado cuadro a cuadro
#    (scripts/xarm5/pack_trial_images.py). Salta las corridas modificadas hace menos de 1 h.
# 3. checkpoints/ intermedios de modelos terminados (con model.safetensors) sin cambios en 6 h; queda el modelo final.
# Variables: BC_DATA, TRIALS, MODELS y DATA, por si las rutas son otras (p. ej. en lab2).
DRY=""; [ "$1" = "--dry-run" ] && DRY=1
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$HOME/miniconda3/envs/fork-splatsim/bin/python}
BC_DATA=${BC_DATA:-$HOME/data/bc_data}
TRIALS=${TRIALS:-$HOME/data/policy_trials}
MODELS=${MODELS:-$HOME/lerobot/examples/tutorial/diffusion/outputs/robot_learning_tutorial}
DATA=${DATA:-$HOME/lerobot/splatsim_test_data}
free() { df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9'; }
before=$(free)

renders=()
for r in "$BC_DATA"/*_10hz*; do
    [ -f "$r/DONE" ] || continue
    # ya podado y sin PNG nuevos: no hace falta volver a verificar
    [ -f "$r/dataset_images.yaml" ] && [ -z "$(find "$r" -mindepth 3 -maxdepth 3 -name '*_rgb_*.png' -print -quit)" ] && continue
    [ -n "$(find "$r" -mindepth 3 -maxdepth 3 -name '*_rgb_*.png' -print -quit)" ] && renders+=("$r")
done
if [ ${#renders[@]} -gt 0 ]; then
    nice -n 19 "$PY" "$REPO/scripts/xarm5/prune_render_images.py" "${renders[@]}" --roots "$DATA" \
        $([ -z "$DRY" ] && echo --delete) 2>&1 | grep -v Warning
fi

[ -d "$TRIALS" ] && nice -n 19 "$PY" "$REPO/scripts/xarm5/pack_trial_images.py" "$TRIALS" --workers 4 \
    $([ -n "$DRY" ] && echo --dry_run) 2>&1 | tail -3

n=0
for m in "$MODELS"/*/; do
    m=${m%/}
    [ -f "$m/model.safetensors" ] && [ -d "$m/checkpoints" ] && [ -n "$(ls -A "$m/checkpoints")" ] || continue
    [ -z "$(find "$m" -newermt '-6 hours' -print -quit)" ] || continue
    n=$((n + 1))
    [ -z "$DRY" ] && rm -rf "${m:?}/checkpoints"
done
echo "checkpoints intermedios: $n modelos$([ -n "$DRY" ] && echo ' (dry-run)')"
echo "disco: $before → $(free) GB libres"
