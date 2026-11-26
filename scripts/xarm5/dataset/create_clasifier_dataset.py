import argparse
from pathlib import Path
import shutil
from typing import Sequence
import warnings

def find_image_dir(episode_dir: Path) -> Path:
    """
    Detecta el nombre de la carpeta de imágenes del episodio.

    - SplatSim usa 'image_1/'
    - Episodios reales (gello) usan 'images/'

    Si ninguna existe, se asume que no hay carpeta de imágenes separada
    (las imágenes podrían venir embebidas en el pkl, caso no soportado
    aún por este conversor para escritura a video).
    """
    for candidate in ("image_1", "images"):
        d = episode_dir / candidate
        if d.exists():
            return d

    raise RuntimeError(
        f"No se encontró carpeta de imágenes ('image_1/' ni 'images/') en {episode_dir}"
    )

def copy_images(
    images: Sequence[Path],
    label: str,
    episode_number: str,
    output_dir: Path,
    next_idx: int,
) -> int:
    for src in images:
        frame_number = src.stem.rsplit("_", 1)[1]

        dst = output_dir / (
            f"{label}_{episode_number}_{frame_number}_{next_idx:06d}.png"
        )

        shutil.copy2(src, dst)
        next_idx += 1

    return next_idx


def load_episode_images(
    episode_dir: Path,
) -> tuple[list[Path], list[Path]]:
    image_path = find_image_dir(episode_dir)

    wrist_images = sorted(image_path.glob("wrist_*.png"))
    
    if len(wrist_images) < 16:
        warnings.warn(f"Episodio {episode_dir.name} tiene solo {len(wrist_images)} imágenes. El episodio será ignorado.")
        return [], []

    valid_grasp_images = wrist_images[-5:]
    invalid_grasp_images = wrist_images[10:16]
    
    return valid_grasp_images, invalid_grasp_images

def discover_episode_dirs(input_dir: Path) -> list[Path]:
    """Encuentra subcarpetas de episodio dentro de input_dir, ordenadas por nombre.

    Se asume que cada subcarpeta inmediata de input_dir es un episodio
    (formato observado: '000', '001', '002', ...). No se asume ningún
    patrón de nombre específico más allá de "es un directorio".
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"El directorio de entrada no existe: {input_dir}")
    episode_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not episode_dirs:
        warnings.warn(f"No se encontraron subcarpetas de episodio en {input_dir}")
    return episode_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfiere imágenes de la cámara en mano (wrist) desde demostraciones SplatSim/GELLO para crear un dataset de clasificación de grasping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directorio raíz con subcarpetas de episodio (ej. ~/data/bc_data/gello/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visual_grasp_clasifier_dataset",
        help="Directorio donde se guardarán las imágenes del dataset.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce la verbosidad de las advertencias impresas en consola.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.quiet:
        warnings.filterwarnings("once")
    else:
        warnings.filterwarnings("always")

    episode_dirs = discover_episode_dirs(input_dir)
    print(f"Encontrados {len(episode_dirs)} episodios en {input_dir}")

    output_initial_lenght = len(list(output_dir.glob("*.png"))) 
    next_idx = output_initial_lenght
    
    for episode_dir in episode_dirs:
        print(f"Cargando episodio: {episode_dir}")
        valid_grasp_images, invalid_grasp_images = load_episode_images(episode_dir)
        
        if len(valid_grasp_images) < 1 or len(invalid_grasp_images) < 1:
            continue

        episode_number = episode_dir.name  # "000", "001", ...    

        next_idx = copy_images(
            valid_grasp_images,
            "grasped",
            episode_number,
            output_dir,
            next_idx,
        )

        next_idx = copy_images(
            invalid_grasp_images,
            "not_grasped",
            episode_number,
            output_dir,
            next_idx,
        )
        
    print(f"\nSe han copiado {next_idx-output_initial_lenght} imágenes a {output_dir}.")

if __name__ == "__main__":
    main()