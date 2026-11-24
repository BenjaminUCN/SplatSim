"""Borra los PNG de renders ya convertidos a dataset LeRobot (splatsim/utils/data/dataset_frames.py).

    python scripts/xarm5/prune_render_images.py ~/data/bc_data/cut_400tr_10hz            # solo verifica y cuenta
    python scripts/xarm5/prune_render_images.py ~/data/bc_data/*_10hz --delete

Para cada render busca el dataset convertido desde él: el que lo nombra en meta/splatsim_sources.yaml (si hay varios,
el más reciente con DONE).
Luego verifica:
- que cada episodio tenga tantos cuadros como .pkl;
- que haya un PNG por cuadro;
- que el primer, el del medio y el último cuadro de 5 episodios se parezcan a los del dataset.

Solo si todo eso pasa, borra los PNG y deja dataset_images.yaml para el revisor:
    python scripts/xarm5/review_recordings.py
Los .pkl, meta.json y render_info.yaml se quedan. Un render sin dataset, o con un problema, no se toca.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from splatsim.utils.data import dataset_frames as df  # noqa: E402


@dataclass
class Args:
    renders: tyro.conf.Positional[tuple[str, ...]]
    roots: tuple[str, ...] = ("~/lerobot/splatsim_test_data",)
    delete: bool = False
    samples: int = 5


def main(args: Args) -> int:
    roots = [Path(r).expanduser() for r in args.roots]
    total, failed = 0, 0
    for r in args.renders:
        render = Path(r).expanduser()
        if not (render / "DONE").exists() and not any(render.glob("*/meta.json")):
            print(f"{render.name}: no parece un render terminado, lo salto")
            failed += 1
            continue
        found = df.find_dataset(render, roots)
        if not found:
            print(f"{render.name}: ningún dataset lo nombra en meta/splatsim_sources.yaml, no se toca")
            failed += 1
            continue
        dataset, mapping = found
        problems = df.check(render, dataset, mapping, args.samples)
        if problems:
            print(f"{render.name}: {len(problems)} problemas con {dataset.name}, no se toca:")
            for p in problems[:10]:
                print(f"    {p}")
            failed += 1
            continue
        size, kept, extra = df.prune(render, dataset, mapping, delete=args.delete)
        total += size
        verb = "borrados" if args.delete else "se pueden borrar"
        print(f"{render.name} → {dataset.name}: {len(mapping)} episodios verificados, {size / 1e9:.1f} GB {verb}"
              + (f"; {kept} episodios fuera del dataset conservan sus PNG" if kept else "")
              + (f"; cámaras que no están en el dataset (se conservan): {', '.join(extra)}" if extra else ""))
    print(f"total: {total / 1e9:.1f} GB {'borrados' if args.delete else '(sin --delete no se borra nada)'}")
    return 1 if failed else 0


if __name__ == "__main__":
    code = main(tyro.cli(Args))
    sys.stdout.flush()
    os._exit(code)
