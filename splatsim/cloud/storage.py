"""Almacenamiento de objetos con rclone, igual en el lab y en los pods.

remote es un remoto de rclone con su bucket ("r2:splatsim") o una carpeta local (en las pruebas).
Las credenciales van en variables RCLONE_CONFIG_*, nunca en archivos ni en la línea de comandos.
"""
import fnmatch
import itertools
import os
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ~/.local/bin no siempre está en el PATH (p.ej. si el panel se abrió desde otro proceso)
os.environ["PATH"] = f"{os.environ.get('PATH', '')}:{Path.home() / '.local/bin'}"

SECRETS_FILE = "~/.config/splatsim/cloud.env"
REMOTE_NAME = "r2"
FOLDER_FLAGS = ("--transfers", "32", "--checkers", "32", "--checksum")
PACK_SUFFIX = ".tar"
PACK_WORKERS = 8


def load_secrets(path: str = SECRETS_FILE) -> dict[str, str]:
    """Lee KEY=valor (ver configs/cloud.env.example). Rechaza un archivo legible por otros."""
    file = Path(path).expanduser()
    if not file.exists():
        raise SystemExit(f"Falta {file}; copia configs/cloud.env.example y complétalo")
    if file.stat().st_mode & 0o077:
        raise SystemExit(f"{file} lo pueden leer otros usuarios: corre chmod 600 {file}")
    pairs = (line.split("=", 1) for line in file.read_text().splitlines()
             if "=" in line and not line.lstrip().startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs if v.strip()}


def storage_env(secrets: dict[str, str]) -> dict[str, str]:
    """Variables con que rclone encuentra el bucket R2, en este proceso y en el pod."""
    prefix = f"RCLONE_CONFIG_{REMOTE_NAME.upper()}_"
    return {
        f"{prefix}TYPE": "s3",
        f"{prefix}PROVIDER": "Cloudflare",
        f"{prefix}ACCESS_KEY_ID": secrets["R2_ACCESS_KEY_ID"],
        f"{prefix}SECRET_ACCESS_KEY": secrets["R2_SECRET_ACCESS_KEY"],
        f"{prefix}ENDPOINT": f"https://{secrets['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        f"{prefix}NO_CHECK_BUCKET": "true",   # el token solo tiene acceso al bucket, no a listarlos
        "SPLATSIM_REMOTE": f"{REMOTE_NAME}:{secrets['R2_BUCKET']}",
    }


def item_label(index: int, count: int, key: str, done: int | None = None, total: int | None = None) -> str:
    """Qué carpeta se está moviendo y, si va en paquetes, por cuál episodio va."""
    return f"{index}/{count} {Path(key).name}" + (f": {done}/{total} episodes" if done is not None else "")


def is_excluded(relative: str, patterns: list[str]) -> bool:
    """Los filtros de rclone que usan las etapas: "/episodio/**" y "**/nombre_*.png"."""
    for pattern in patterns:
        if pattern.startswith("/") and pattern.endswith("/**"):
            if relative.startswith(pattern[1:-2]):
                return True
        elif pattern.startswith("**/"):
            if fnmatch.fnmatch(relative.rsplit("/", 1)[-1], pattern[3:]):
                return True
        else:
            raise ValueError(f"filtro no soportado en un paquete: {pattern}")
    return False


def write_pack(episode: Path, target: Path, exclude: list[str] | None = None) -> bool:
    """Tar sin comprimir (los PNG ya lo están). Mismo contenido, mismos bytes: así rclone reconoce por su
    hash el paquete que ya está en el bucket. False si no queda ningún archivo."""
    names = [p for p in sorted(episode.rglob("*")) if p.is_file()
             and not is_excluded(f"{episode.name}/{p.relative_to(episode).as_posix()}", exclude or [])]
    if not names:
        return False
    with tarfile.open(target, "w", format=tarfile.GNU_FORMAT, dereference=True) as tar:
        for path in names:
            info = tar.gettarinfo(path, arcname=f"{episode.name}/{path.relative_to(episode).as_posix()}")
            info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, "", "", 0o644
            info.mtime = int(info.mtime)
            with open(path, "rb") as f:
                tar.addfile(info, f)
    return True


class EpisodePacks:
    """Un paquete por episodio en vez de un objeto por archivo: 127 mil archivos de una grabación tardan
    casi una hora por la latencia de cada uno, y un corte solo repite los episodios que faltaban."""

    def _upload_packed(self, local: Path, key: str, exclude: list[str] | None, progress) -> None:
        items = sorted(Path(local).iterdir())
        counter = itertools.count(1)

        def send(item: Path) -> None:
            if item.is_dir():
                with tempfile.TemporaryDirectory(dir=Path(local).parent, prefix=".pack-") as tmp:
                    pack = Path(tmp) / f"{item.name}{PACK_SUFFIX}"
                    if write_pack(item, pack, exclude):
                        self.upload_file(pack, f"{key}/{pack.name}")
            elif not is_excluded(item.name, exclude or []):
                self.upload_file(item, f"{key}/{item.name}")
            if progress:
                progress(next(counter), len(items))

        with ThreadPoolExecutor(PACK_WORKERS) as pool:
            list(pool.map(send, items))

    def _unpack(self, key: str, local: Path, progress) -> None:
        packs = [name for name in self.list_files(key) if name.endswith(PACK_SUFFIX)]
        if not packs:
            return
        Path(local).mkdir(parents=True, exist_ok=True)
        counter = itertools.count(1)

        def fetch(name: str) -> None:
            with tempfile.TemporaryDirectory(dir=Path(local).parent, prefix=".pack-") as tmp:
                pack = Path(tmp) / name
                self.download_file(f"{key}/{name}", pack)
                with tarfile.open(pack) as tar:
                    tar.extractall(local, filter="data")
            if progress:
                progress(next(counter), len(packs))

        with ThreadPoolExecutor(PACK_WORKERS) as pool:
            list(pool.map(fetch, packs))


class Storage(EpisodePacks):
    def __init__(self, remote: str, env: dict[str, str] | None = None):
        if shutil.which("rclone") is None:
            raise SystemExit("Falta rclone (https://rclone.org/install/)")
        self.remote = remote.rstrip("/")
        self.env = {**os.environ, **(env or {})}

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["rclone", *args], env=self.env, text=True, capture_output=True)

    def _checked(self, *args: str) -> str:
        done = self._run(*args)
        if done.returncode:
            raise RuntimeError(f"rclone {args[0]} falló: {done.stderr.strip()[-500:]}")
        return done.stdout

    def _path(self, key: str) -> str:
        return f"{self.remote}/{key}"

    def read(self, key: str) -> str | None:
        """Contenido de un objeto de texto, o None si no existe."""
        done = self._run("cat", self._path(key))
        return done.stdout if done.returncode == 0 and done.stdout else None

    def write(self, key: str, text: str) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
            f.write(text)
            f.flush()
            self._checked("copyto", f.name, self._path(key))

    def upload(self, local: str | Path, key: str, exclude: list[str] | None = None, packed: bool = False,
               progress=None) -> None:
        """Copia una carpeta; lo que ya está igual (tamaño y hash) no se vuelve a subir. Con packed, cada
        subcarpeta llega como un paquete."""
        if packed:
            self._upload_packed(Path(local), key, exclude, progress)
            return
        filters = [arg for pattern in exclude or [] for arg in ("--exclude", pattern)]
        self._checked("copy", str(local), self._path(key), *FOLDER_FLAGS, *filters)

    def upload_file(self, local: str | Path, key: str) -> None:
        self._checked("copyto", str(local), self._path(key), "--checksum")

    def download(self, key: str, local: str | Path, progress=None) -> None:
        """Deja la carpeta igual se haya subido suelta o en paquetes."""
        self._checked("copy", self._path(key), str(local), *FOLDER_FLAGS, "--exclude", f"/*{PACK_SUFFIX}")
        self._unpack(key, Path(local), progress)

    def download_file(self, key: str, local: str | Path) -> None:
        self._checked("copyto", self._path(key), str(local))

    def list_files(self, key: str) -> list[str]:
        """Los archivos directos de una carpeta, sin entrar a las subcarpetas."""
        return self._checked("lsf", self._path(key), "--files-only", "--max-depth", "1").splitlines()

    def copy(self, source_key: str, target_key: str) -> None:
        """Copia dentro del bucket, sin pasar por este PC."""
        self._checked("copy", self._path(source_key), self._path(target_key), *FOLDER_FLAGS)

    def delete(self, key: str) -> None:
        self._checked("purge", self._path(key))
