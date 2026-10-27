"""
lerobot_transform.py
=========================

Convierte episodios de demostración a un dataset LeRobot v3 VÁLIDO,
usando la API oficial de LeRobot.

Soporta DOS esquemas de entrada, detectados automáticamente por episodio:

  1. SIM (SplatSim):
       - pkl con clave 'action' (7,) ya combinada
       - carpeta de imágenes 'image_1/'
       - convención de índices: base_idx = 2*(idx-1), wrist_idx = 2*(idx-1)+1
       - nombres de pkl numéricos: 00001.pkl, 00002.pkl, ...

     Dos formas de numerar las imágenes, detectadas por episodio:
       - por frame (servidor xArm5 actual): <cam>_00001.png va con 00001.pkl,
         para cualquier cámara (base_rgb, base2_rgb, wrist_rgb, ...)
       - intercalada (grabaciones antiguas): ver convención de arriba

  2. REAL (gello, brazo físico):
       - pkl SIN 'action': trae 'control' (6,) + 'gripper_position' (escalar)
         -> action_real = concat(control, gripper_position)  (7,)
       - carpeta de imágenes 'images/' (no 'image_1/')
       - convención de índices: base_rgb empieza en 2, sube de 2 en 2
         (base_idx = 2*posición + 2, wrist_idx = 2*posición + 1)
       - nombres de pkl con timestamp: '2026-06-18T17:23:43.608734.pkl'
         (no son enteros, se ordenan alfabéticamente / cronológicamente)

El esquema se detecta inspeccionando las claves del primer pkl y el
nombre de la carpeta de imágenes presente en el episodio. Cada episodio
puede tener un esquema distinto dentro de la misma corrida de conversión
(útil para mezclar sim + real en un solo dataset si así lo decides).

Requirements
------------
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .

pip install imageio imageio-ffmpeg

Usage
-----
python lerobot_transform.py \
    --input_dir ~/data/bc_data_real/gello \
    --output_dir ~/datasets/splatsim_real_v3
"""

from __future__ import annotations

import argparse
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================
# LeRobot imports
# ============================================================

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============================================================


@dataclass
class ConversionConfig:
    input_dirs: list[Path]
    output_dir: Path

    fps: int = 10

    robot_type: str = "xarm5_lite"

    wrist_camera: bool = True
    cameras: list[str] | None = None  # subconjunto de cámaras a convertir (por defecto todas)
    wrist_offset: int | None = None
    use_videos: bool = True


# ============================================================
# Utils
# ============================================================


def safe_to_numpy(obj: Any) -> np.ndarray:

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()

    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, (list, tuple)):
        return np.asarray(obj)

    if np.isscalar(obj):
        return np.asarray([obj])

    raise TypeError(f"Unsupported type: {type(obj)}")


def load_pickle(path: Path) -> dict:

    with open(path, "rb") as f:

        try:
            data = pickle.load(f, encoding="latin1")
        except Exception:
            f.seek(0)
            data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}")

    return data


def load_image(path: Path) -> np.ndarray:
    """
    Returns:
        uint8 CHW image
    """

    img = Image.open(path).convert("RGB")

    arr = np.asarray(img, dtype=np.uint8)

    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))

    return arr


def find_image(image_dir: Path, prefix: str, index: int) -> Path | None:

    candidates = [
        f"{prefix}_{index:05d}.png",
        f"{prefix}_{index:04d}.png",
        f"{prefix}_{index}.png",
    ]

    for c in candidates:
        p = image_dir / c

        if p.exists():
            return p

    return None


def pkl_sort_key(p: Path):
    """
    Clave de ordenamiento robusta para nombres de archivo .pkl.

    - Si el stem es un entero puro ('00001') -> ordena numéricamente
      (caso SplatSim).
    - Si no (timestamps como '2026-06-18T17:23:43.608734') -> ordena
      alfabéticamente por el nombre completo, que para timestamps con
      formato ISO-like consistente da el orden cronológico correcto
      (caso real/gello).

    Se usa una tupla (0, n, "") / (1, 0, nombre) para que nunca se
    comparen directamente un int con un str.
    """
    try:
        return (0, int(p.stem), "")
    except ValueError:
        return (1, 0, p.stem)


def discover_episodes(input_dirs: list[Path]) -> list[Path]:
    """
    Descubre episodios (subcarpetas con .pkl) dentro de una o más
    carpetas raíz.

    Permite mezclar fuentes en una sola conversión, p.ej.:
        input_dirs = [Path("~/bc_data/gello"), Path("~/bc_data/gello_real")]
    para generar un único dataset LeRobot con episodios de simulación
    y episodios reales juntos.
    """

    episodes: list[Path] = []

    for input_dir in input_dirs:

        if not input_dir.exists():
            raise RuntimeError(f"La carpeta no existe: {input_dir}")

        found = sorted(
            d for d in input_dir.iterdir()
            if d.is_dir() and any(d.glob("*.pkl"))
        )

        if not found:
            warnings.warn(f"No se encontraron episodios (.pkl) en {input_dir}")

        print(f"  {input_dir}: {len(found)} episodios encontrados")
        episodes.extend(found)

    if not episodes:
        raise RuntimeError(f"No episodes found in {input_dirs}")

    return episodes


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


def detect_episode_schema(episode_dir: Path) -> str:
    """
    Detecta si un episodio es 'sim' (SplatSim) o 'real' (gello),
    inspeccionando las claves del primer pkl.

    Returns:
        "sim"  si el primer pkl tiene clave 'action'
        "real" si el primer pkl tiene 'control' (+ opcionalmente
               'gripper_position') pero no 'action'
    """
    pkl_files = sorted(episode_dir.glob("*.pkl"), key=pkl_sort_key)

    if not pkl_files:
        raise RuntimeError(f"No hay .pkl en {episode_dir}")

    data = load_pickle(pkl_files[0])

    if "action" in data:
        return "sim"

    if "control" in data:
        return "real"

    raise RuntimeError(
        f"No se pudo determinar el esquema de {episode_dir}: "
        f"no tiene 'action' ni 'control'. Claves encontradas: {list(data.keys())}"
    )


def detect_wrist_offset(
    episode_dir: Path,
    image_dir_name: str,
    num_samples: int = 5,
) -> int | None:

    image_dir = episode_dir / image_dir_name

    if not image_dir.exists():
        return None

    pkls = sorted(episode_dir.glob("*.pkl"), key=pkl_sort_key)[:num_samples]

    matches = []

    for pos, pkl_path in enumerate(pkls):

        if find_image(image_dir, "wrist_rgb", pos):
            matches.append(0)

        elif find_image(image_dir, "wrist_rgb", pos + 1):
            matches.append(1)

    if not matches:
        return None

    if all(m == matches[0] for m in matches):
        return matches[0]

    return None


# ============================================================
# Dynamic feature inference
# ============================================================


def infer_image_keys(
    episode_dir: Path,
    image_dir_name: str,
    wrist_camera: bool,
    cameras: list[str] | None = None,
) -> list[str]:
    """Cámaras presentes en el episodio (<cam>_rgb_<idx>.png), base_rgb primero y wrist al final."""

    image_dir = episode_dir / image_dir_name

    found = {
        m.group(1)
        for p in image_dir.glob("*.png")
        if (m := re.match(r"^(.+_rgb)_\d+\.png$", p.name))
    }

    if cameras is not None:
        missing = set(cameras) - found
        if missing:
            raise RuntimeError(f"Cámaras {sorted(missing)} no están en {image_dir} (hay {sorted(found)})")
        found = set(cameras)

    if not wrist_camera:
        found.discard("wrist_rgb")

    if not found:
        raise RuntimeError(f"No image streams found in {image_dir}")

    return sorted(found, key=lambda k: (k.startswith("wrist"), k != "base_rgb", k))


def uses_per_frame_naming(
    episode_dir: Path,
    image_dir_name: str,
    image_keys: list[str],
) -> bool:
    """True si cada pkl tiene una imagen por cámara con su mismo número (<cam>_00001.png
    junto a 00001.pkl). En la numeración intercalada antigua cada índice es de una sola
    cámara, así que esta prueba falla."""

    image_dir = episode_dir / image_dir_name
    pkls = sorted(episode_dir.glob("*.pkl"), key=pkl_sort_key)[:5]

    return bool(pkls) and all(
        (image_dir / f"{key}_{pkl.stem}.png").exists()
        for pkl in pkls
        for key in image_keys
    )


def infer_image_shape(
    episode_dir: Path,
    image_dir_name: str,
    image_key: str,
) -> tuple[int, int, int]:

    image_dir = episode_dir / image_dir_name

    img_path = next(
        image_dir.glob(f"{image_key}*.png")
    )

    img = Image.open(img_path).convert("RGB")

    w, h = img.size

    # CHW
    return (3, h, w)


def build_action_from_pkl(pkl_data: dict, schema: str) -> np.ndarray:
    """
    Construye el vector de acción (7,) a partir del pkl, según el esquema
    del episodio.

    - sim:  usa directamente pkl_data['action']  (ya viene (7,))
    - real: concatena pkl_data['control'] (6,) + pkl_data['gripper_position']
            (escalar 0-d) -> (7,)
    """
    if schema == "sim":
        return safe_to_numpy(pkl_data["action"]).astype(np.float32).flatten()

    if schema == "real":
        # control = safe_to_numpy(pkl_data["control"]).astype(np.float32).flatten()
        # gripper = safe_to_numpy(pkl_data["gripper_position"]).astype(np.float32).flatten()
        # return np.concatenate([control, gripper])
        return safe_to_numpy(pkl_data["state"]).astype(np.float32).flatten()

    raise ValueError(f"Esquema desconocido: {schema}")


def infer_state_action_dims(episode_dir: Path, schema: str) -> tuple[int, int]:

    pkl_path = sorted(
        episode_dir.glob("*.pkl"),
        key=pkl_sort_key,
    )[0]

    data = load_pickle(pkl_path)

    state = safe_to_numpy(data["state"]).flatten()
    action = build_action_from_pkl(data, schema)

    return len(state), len(action)


# ============================================================
# LeRobot feature spec
# ============================================================


def build_lerobot_features(
    image_keys: list[str],
    image_shapes: dict[str, tuple[int, int, int]],
    state_dim: int,
    action_dim: int,
    frame_type: str = "video",
):

    features = {}

    for key in image_keys:

        features[f"observation.images.{key}"] = {
            "dtype": frame_type,
            "shape": image_shapes[key],
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": [f"state_{i}" for i in range(state_dim)],
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": [f"action_{i}" for i in range(action_dim)],
    }

    return features


# ============================================================
# Dataset creation
# ============================================================


def create_dataset(
    config: ConversionConfig,
    image_keys: list[str],
    image_shapes: dict[str, tuple[int, int, int]],
    state_dim: int,
    action_dim: int,
    repo_suffix: str,
):

    features = build_lerobot_features(
        image_keys=image_keys,
        image_shapes=image_shapes,
        state_dim=state_dim,
        action_dim=action_dim,
        frame_type="video" if config.use_videos else "image",
    )

    hf_user = "LuEduSoHu"
    repo_id = f"{hf_user}/splatsim_{config.robot_type}_{config.fps}fps_{repo_suffix}"
    root_dir = config.output_dir / repo_id.split("/")[1]
    print(
        f"Creating LeRobot dataset with config: \nrepo_id: '{repo_id}'\n"
        f"root_dir: '{root_dir}'\nrobot_type: {config.robot_type}\n"
        f"fps: {config.fps}\nfeatures: {features}\nuse_videos: {config.use_videos}"
    )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root_dir,
        robot_type=config.robot_type,
        fps=config.fps,
        features=features,
        use_videos=config.use_videos,
    )

    return dataset


def push_lerobot_to_hub(dataset: "LeRobotDataset") -> None:
    """Push dataset to hub, retrying with a new repo_id on failure.

    Loops until the push succeeds or the user presses Enter to skip.
    """
    while True:
        repo_id = dataset.repo_id
        print(f"[LeRobot] Pushing dataset to hub as '{repo_id}'...")
        try:
            dataset.push_to_hub()
            print(f"[LeRobot] Successfully pushed to hub as '{repo_id}'.")
            return
        except Exception as e:
            print(f"[LeRobot] ERROR: Failed to push to hub: {e}")
            print("[LeRobot] Repo ID should be in 'username/dataset_name' format.")
            print("[LeRobot] Make sure you are authenticated with `huggingface-cli login`.")
            new_repo_id = input("[LeRobot] Enter a new repo_id to retry (or press Enter to skip): ").strip()
            if not new_repo_id:
                print("[LeRobot] Skipping push to hub. Dataset is saved locally.")
                return
            dataset.repo_id = new_repo_id


# ============================================================
# Frame builder
# ============================================================


def build_frame(
    pkl_data: dict,
    images: dict[str, np.ndarray],
    image_keys: list[str],
    schema: str,
    task: str = "",
) -> dict:
    """
    Build a frame compatible with LeRobotDataset.add_frame().

    Expected conventions:
    - observation.state -> float32
    - action            -> float32 (7,), construido según `schema`
    - images            -> float32 CHW in [0, 1]
    """

    state = safe_to_numpy(
        pkl_data["state"]
    ).astype(np.float32).flatten()

    action = build_action_from_pkl(pkl_data, schema)

    frame: dict = {
        "observation.state": state,
        "action": action,
        "task": task,
    }

    for key in image_keys:
        if key not in images:
            continue

        img = images[key]
        img = np.asarray(img)  # Ensure numpy

        # Convert HWC -> CHW if necessary
        if img.ndim != 3:
            raise ValueError(f"Image '{key}' must be 3D, got shape {img.shape}")

        # HWC -> CHW
        if img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))

        # Convert uint8 -> float32 [0,1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)

        # Final validation
        if img.shape[0] != 3:
            raise ValueError(f"Image '{key}' must be CHW with 3 channels, got {img.shape}")

        frame[f"observation.images.{key}"] = img

    return frame


# ============================================================
# Episode processing
# ============================================================


def process_episode(
    dataset: LeRobotDataset,
    episode_dir: Path,
    image_keys: list[str],
    image_dir_name: str,
    schema: str,
    config: ConversionConfig,
):

    image_dir = episode_dir / image_dir_name

    pkl_files = sorted(episode_dir.glob("*.pkl"), key=pkl_sort_key)

    if not pkl_files:
        return

    # Offset base, en imágenes, del primer base_rgb del episodio.
    # SplatSim (image_1/): primer base_rgb en índice 0 -> base_idx = 2*pos
    # Real (images/):       primer base_rgb en índice 2 -> base_idx = 2*pos + 2
    base_start_offset = 0 if image_dir_name == "image_1" else 2

    wrist_offset = config.wrist_offset

    if "wrist_rgb" in image_keys and wrist_offset is None:
        detected = detect_wrist_offset(episode_dir, image_dir_name)

        wrist_offset = (
            detected
            if detected is not None
            else 0
        )

    per_frame = uses_per_frame_naming(episode_dir, image_dir_name, image_keys)

    n_missing = 0

    for position, pkl_path in enumerate(pkl_files):

        try:
            pkl_data = load_pickle(pkl_path)
        except Exception as e:
            warnings.warn(f"Failed loading {pkl_path}: {e}")
            continue

        images = {}
        valid = True

        # Convención de índices, por posición (0-indexed) dentro del
        # episodio en orden cronológico:
        #
        #   base_idx  = 2*position + base_start_offset
        #   wrist_idx = base_idx + 1  (con offset adicional si se detectó)
        for key in image_keys:

            if per_frame:
                img_path = image_dir / f"{key}_{pkl_path.stem}.png"
                if not img_path.exists():
                    img_path = None

            elif key == "base_rgb":
                base_idx = 2 * position + base_start_offset
                img_path = find_image(image_dir, key, base_idx)
                if img_path is None:
                    img_path = find_image(image_dir, key, base_idx + 1)

            elif key == "wrist_rgb":
                wrist_idx = 2 * position + base_start_offset + 1
                img_path = find_image(image_dir, key, wrist_idx)
                if img_path is None:
                    img_path = find_image(image_dir, key, wrist_idx + 1)

            else:
                img_path = None

            if img_path is None:
                valid = False
                n_missing += 1
                break

            try:
                images[key] = load_image(img_path)
            except Exception as e:
                warnings.warn(f"Failed image {img_path}: {e}")
                valid = False
                break

        if not valid:
            continue

        try:
            frame = build_frame(
                pkl_data,
                images,
                list(images.keys()),
                schema=schema,
                task="aproach_apple",
            )
            dataset.add_frame(frame)
        except Exception as e:
            warnings.warn(f"Failed building frame for {pkl_path}: {e}")

    if n_missing:
        print(
            f"  [{episode_dir.name}] {n_missing} frames descartados por "
            f"imágenes faltantes (de {len(pkl_files)} pkls)"
        )

    dataset.save_episode()


# ============================================================
# Main conversion
# ============================================================


def run_conversion(config: ConversionConfig, repo_suffix: str):

    print("=" * 70)
    print("SplatSim / Real -> LeRobot v3")
    print("=" * 70)

    episode_dirs = discover_episodes(config.input_dirs)
    print(f"\nFound {len(episode_dirs)} episodes total\n")

    # Las features del dataset (image_keys, shapes, state/action dims) se
    # infieren SIEMPRE del primer episodio de esquema 'sim' encontrado,
    # sin importar el orden de las carpetas pasadas en --input_dir.
    #
    # Esto es crítico cuando se está EXTENDIENDO un dataset que ya existe
    # en Hugging Face (creado originalmente a partir de episodios de
    # simulación): las features nuevas deben coincidir exactamente con
    # las que ya tiene ese repo, o LeRobot rechazará la combinación o se
    # corromperá el dataset existente.
    reference_episode = None
    for ep in episode_dirs:
        if detect_episode_schema(ep) == "sim":
            reference_episode = ep
            break

    if reference_episode is None:
        warnings.warn(
            "No se encontró ningún episodio con esquema 'sim' entre las "
            "carpetas dadas. Se usará el primer episodio encontrado para "
            "definir las features, sea cual sea su esquema. Si tu intención "
            "era extender un dataset de simulación ya existente en HF, "
            "verifica que la carpeta de simulación esté incluida en "
            "--input_dir."
        )
        reference_episode = episode_dirs[0]

    schema = detect_episode_schema(reference_episode)
    image_dir_name = find_image_dir(reference_episode).name

    print(f"Episodio de referencia para definir features: '{reference_episode}'")
    print(f"Esquema detectado: '{schema}'")
    print(f"Carpeta de imágenes detectada: '{image_dir_name}'")

    image_keys = infer_image_keys(
        reference_episode, image_dir_name, config.wrist_camera, config.cameras
    )

    image_shapes = {
        k: infer_image_shape(reference_episode, image_dir_name, k)
        for k in image_keys
    }

    state_dim, action_dim = infer_state_action_dims(reference_episode, schema)

    print("\nDetected configuration:")
    print(f"Image keys : {image_keys}")

    for k, shape in image_shapes.items():
        print(f"{k:12s}: {shape}")

    print(f"State dim  : {state_dim}")
    print(f"Action dim : {action_dim}")

    dataset = create_dataset(
        config=config,
        image_keys=image_keys,
        image_shapes=image_shapes,
        state_dim=state_dim,
        action_dim=action_dim,
        repo_suffix=repo_suffix,
    )

    for ep_dir in tqdm(episode_dirs):

        ep_schema = detect_episode_schema(ep_dir)
        ep_image_dir_name = find_image_dir(ep_dir).name

        if ep_schema != schema:
            warnings.warn(
                f"Episodio {ep_dir.name} tiene esquema '{ep_schema}', distinto "
                f"al esquema '{schema}' usado para definir las features del "
                f"dataset. Se procesará igual, pero verifica que la acción "
                f"resultante tenga sentido."
            )

        process_episode(
            dataset=dataset,
            episode_dir=ep_dir,
            image_keys=image_keys,
            image_dir_name=ep_image_dir_name,
            schema=ep_schema,
            config=config,
        )

    print("\nFinalizing dataset...")

    try:
        dataset.clear_episode_buffer()
    except Exception:
        pass
    dataset.finalize()

    print("\nDone.")
    print(
        f"\nDataset saved to:\n"
        f"{config.output_dir.resolve()}"
    )

    push_lerobot_to_hub(dataset)


# ============================================================
# CLI
# ============================================================


def build_parser():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        nargs="+",
        help="Una o más carpetas raíz con episodios (subcarpetas con .pkl). "
        "Ej: --input_dir ~/bc_data/gello ~/bc_data/gello_real "
        "para combinar simulación y real en un solo dataset.",
    )
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--robot_type", type=str, default="xarm5_lite")
    p.add_argument("--wrist_offset", type=int, default=None, choices=[0, 1])
    p.add_argument("--no_wrist_camera", action="store_true")
    p.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Cámaras a incluir, ej. --cameras base_rgb base2_rgb wrist_rgb "
        "(por defecto todas las que tenga el primer episodio).",
    )
    p.add_argument("--use_images", action="store_true")
    p.add_argument(
        "--repo_suffix",
        type=str,
        default=None,
        help="Sufijo para el repo_id, ej. '260619_1723' o 'fewshot_260622'. "
        "Si no se especifica, se usa la fecha/hora actual. IMPORTANTE: usa "
        "un sufijo distinto al del dataset original para crear un repo "
        "nuevo y no sobrescribirlo.",
    )

    return p


def main():

    args = build_parser().parse_args()

    config = ConversionConfig(
        input_dirs=args.input_dir,
        output_dir=args.output_dir,

        fps=args.fps,
        robot_type=args.robot_type,

        wrist_offset=args.wrist_offset,
        wrist_camera=not args.no_wrist_camera,
        cameras=args.cameras,
        use_videos=not args.use_images,
    )

    repo_suffix = args.repo_suffix or datetime.now().strftime("%y%m%d_%H%M")

    run_conversion(config, repo_suffix=repo_suffix)


if __name__ == "__main__":
    main()