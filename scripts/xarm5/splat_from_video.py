"""De un video (o una carpeta de fotos) a una escena de SplatSim: fotos → COLMAP → gaussian splat.

    python scripts/xarm5/splat_from_video.py ~/Videos/robot.mp4 --out ~/data/Colmaps/261109_1530

Deja lo mismo que el flujo del README (ffmpeg, convert.py y train.py del submódulo):
    <out>/input/           fotos del video
    <out>/distorted/       base de datos y modelo de COLMAP antes de corregir la distorsión
    <out>/images/          fotos sin distorsión          ← source_path en objects.yaml
    <out>/sparse/0/        cámaras de COLMAP
    <out>/output/          el splat (point_cloud/iteration_<n>/point_cloud.ply) ← model_path

COLMAP corre con pycolmap-cuda12, que trae su propio CUDA: la de apt (la del lab) no lo usa, y emparejar
1300 fotos en la CPU tarda horas. Los pasos de convert.py son los mismos, con sus mismas opciones.
Después falta alinear el splat con el URDF en CloudCompare: eso es a mano (README, "Align the simulator
and the splat"), y el panel de la nube (cloud_server.py) lo espera antes de dar la escena por buena.
"""
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import tyro

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "submodules/gaussian-splatting-wrapper/gaussian_splatting/train.py"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
STEPS = ("frames", "colmap", "train")
TQDM = re.compile(r"Training progress:.*?(\d+)/(\d+) \[")
WARN_REGISTERED = 0.8         # menos fotos con pose que esto suele ser un modelo partido en dos
MIN_REGISTERED = 0.5          # menos, y entrenar el splat es pagar por una escena que no sirve


@dataclass
class Args:
    source: tyro.conf.Positional[str]
    """Video, o carpeta con las fotos ya extraídas."""
    out: str
    """Carpeta de la escena (p.ej. ~/data/Colmaps/<fecha>)."""
    fps: float = 0.0
    """Fotos por segundo de video; 0 = todas (el lab usó todas: 1290 fotos de 43 s)."""
    matcher: str = "exhaustive"
    """exhaustive (como convert.py) o sequential (más rápido; solo sirve para fotos de un video)."""
    iterations: int = 30_000
    data_device: str = "cpu"
    """Dónde guarda train.py las fotos: 1300 fotos de 1600 px son ~20 GB, más que la GPU de muchos pods."""
    steps: list[str] = field(default_factory=lambda: list(STEPS))
    """Pasos a correr, para repetir uno: frames, colmap, train."""
    clean: bool = False
    """Al terminar, borrar input/ y distorted/: el renderizador solo usa images/, sparse/ y output/."""
    python: str = sys.executable
    """Intérprete para train.py (necesita diff_gaussian_rasterization y simple_knn)."""


_last_log = time.monotonic()


def log(text: str) -> None:
    """Con lo que tardó el paso anterior: cada fase crece distinto con las fotos (emparejar todas con
    todas, con su cuadrado; train.py, con las iteraciones), así que una prueba corta se extrapola por fase."""
    global _last_log
    now = time.monotonic()
    print(f"[splat] (+{now - _last_log:.0f} s) {text}", flush=True)
    _last_log = now


def extract_frames(source: Path, folder: Path, fps: float) -> int:
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    if source.is_dir():
        photos = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for photo in photos:
            shutil.copy2(photo, folder / photo.name)
    else:
        rate = ["-vf", f"fps={fps}"] if fps > 0 else []
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source), *rate, "-qscale:v", "1",
                        "-qmin", "1", "-start_number", "1", str(folder / "%05d.jpg")], check=True)
    count = sum(1 for _ in folder.iterdir())
    if count < 3:
        raise SystemExit(f"Solo {count} fotos en {folder}: COLMAP necesita muchas más")
    return count


def run_colmap(scene: Path, matcher: str) -> None:
    """feature_extractor → exhaustive_matcher → mapper → image_undistorter, como convert.py."""
    import pycolmap

    device = pycolmap.Device.cuda if pycolmap.has_cuda else pycolmap.Device.cpu
    if not pycolmap.has_cuda:
        log("⚠️ pycolmap sin CUDA: COLMAP correrá en la CPU")
    inputs, distorted = scene / "input", scene / "distorted"
    shutil.rmtree(distorted, ignore_errors=True)
    (distorted / "sparse").mkdir(parents=True)
    database = distorted / "database.db"
    photos = sum(1 for _ in inputs.iterdir())

    log(f"COLMAP: features de {photos} fotos")
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    pycolmap.extract_features(database, inputs, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              device=device)
    log(f"COLMAP: emparejar ({matcher})")
    match = pycolmap.match_exhaustive if matcher == "exhaustive" else pycolmap.match_sequential
    match(database, device=device)
    log("COLMAP: reconstruir")
    options = pycolmap.IncrementalPipelineOptions()
    options.ba_global_function_tolerance = 1e-6       # --Mapper.ba_global_function_tolerance de convert.py
    models = pycolmap.incremental_mapping(database, inputs, distorted / "sparse", options)
    if not models:
        raise SystemExit("COLMAP no pudo reconstruir la escena: faltan fotos con textura o con solape")
    best, model = max(models.items(), key=lambda kv: kv[1].num_reg_images())
    registered = model.num_reg_images()
    log(f"COLMAP: {len(models)} modelo(s); el mayor tiene {registered}/{photos} fotos y "
        f"{model.num_points3D()} puntos")
    if registered < MIN_REGISTERED * photos:
        raise SystemExit(f"COLMAP ubicó solo {registered / photos:.0%} de las fotos: prueba --matcher exhaustive, "
                         f"más fotos por segundo, o un video con más textura y solape")
    if registered < WARN_REGISTERED * photos:
        log(f"⚠️ solo {registered / photos:.0%} de las fotos tienen pose: revisa la escena antes de alinearla")
    if best != 0:        # convert.py y quien lo lea después esperan el modelo en sparse/0
        shutil.rmtree(distorted / "sparse" / "0", ignore_errors=True)
        model.write(distorted / "sparse" / "0")

    log("COLMAP: quitar la distorsión")
    for name in ("images", "sparse", "stereo"):
        shutil.rmtree(scene / name, ignore_errors=True)
    pycolmap.undistort_images(scene, distorted / "sparse" / "0", inputs, output_type="COLMAP")
    sparse = scene / "sparse"
    (sparse / "0").mkdir(exist_ok=True)
    for item in sparse.iterdir():
        if item.name != "0":
            shutil.move(item, sparse / "0" / item.name)


def train_splat(scene: Path, iterations: int, data_device: str, python: str) -> None:
    """train.py del submódulo. Su barra (tqdm, con \\r) se reescribe como [hecho/total] para el runner."""
    output = scene / "output"
    shutil.rmtree(output, ignore_errors=True)
    command = [python, "-u", str(TRAIN_SCRIPT), "-s", str(scene), "-m", str(output), "--iterations", str(iterations),
               "--save_iterations", str(iterations), "--test_iterations", str(iterations),
               "--data_device", data_device, "--disable_viewer"]
    log(f"train.py: {iterations} iteraciones")
    proc = subprocess.Popen(command, cwd=TRAIN_SCRIPT.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    shown, buffer = -1, b""
    while chunk := proc.stdout.read1(65536):
        buffer += chunk
        *pieces, buffer = re.split(rb"[\r\n]", buffer)
        for piece in pieces:
            text = piece.decode(errors="replace").strip()
            if match := TQDM.search(text):
                done, total = int(match.group(1)), int(match.group(2))
                if done * 100 // total != shown:
                    shown = done * 100 // total
                    print(f"[{done}/{total}] train.py", flush=True)
            elif text:
                print(text, flush=True)
    if proc.wait() != 0:
        raise SystemExit(f"train.py terminó con código {proc.returncode}")
    ply = output / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if not ply.is_file():
        raise SystemExit(f"train.py no dejó {ply}")
    log(f"splat: {ply} ({ply.stat().st_size / 1e6:.0f} MB)")


def main(args: Args) -> None:
    unknown = set(args.steps) - set(STEPS)
    if unknown or args.matcher not in ("exhaustive", "sequential"):
        raise SystemExit(f"Pasos desconocidos {sorted(unknown)} o matcher {args.matcher!r}")
    scene, source = Path(args.out).expanduser(), Path(args.source).expanduser()
    scene.mkdir(parents=True, exist_ok=True)
    if "frames" in args.steps:
        log(f"{extract_frames(source, scene / 'input', args.fps)} fotos de {source.name}")
    if "colmap" in args.steps:
        run_colmap(scene, args.matcher)
    if "train" in args.steps:
        train_splat(scene, args.iterations, args.data_device, args.python)
    if args.clean:
        for name in ("input", "distorted", "stereo"):
            shutil.rmtree(scene / name, ignore_errors=True)
        for script in scene.glob("run-colmap-*.sh"):
            script.unlink()
    log(f"listo: {scene}")


if __name__ == "__main__":
    main(tyro.cli(Args))
