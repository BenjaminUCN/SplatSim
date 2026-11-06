"""Almacenamiento de objetos con rclone, igual en el lab y en los pods.

remote es un remoto de rclone con su bucket ("r2:splatsim") o una carpeta local (en las pruebas).
Las credenciales van en variables RCLONE_CONFIG_*, nunca en archivos ni en la línea de comandos.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# ~/.local/bin no siempre está en el PATH (p.ej. si el panel se abrió desde otro proceso)
os.environ["PATH"] = f"{os.environ.get('PATH', '')}:{Path.home() / '.local/bin'}"

SECRETS_FILE = "~/.config/splatsim/cloud.env"
REMOTE_NAME = "r2"
# Las grabaciones son ~100k archivos pequeños: con pocas transferencias, la latencia manda sobre el ancho de banda.
FOLDER_FLAGS = ("--transfers", "32", "--checkers", "32", "--checksum")


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


class Storage:
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

    def upload(self, local: str | Path, key: str, exclude: list[str] | None = None) -> None:
        """Copia una carpeta; lo que ya está igual (tamaño y hash) no se vuelve a subir."""
        filters = [arg for pattern in exclude or [] for arg in ("--exclude", pattern)]
        self._checked("copy", str(local), self._path(key), *FOLDER_FLAGS, *filters)

    def upload_file(self, local: str | Path, key: str) -> None:
        self._checked("copyto", str(local), self._path(key))

    def download(self, key: str, local: str | Path) -> None:
        self._checked("copy", self._path(key), str(local), *FOLDER_FLAGS)

    def copy(self, source_key: str, target_key: str) -> None:
        """Copia dentro del bucket, sin pasar por este PC."""
        self._checked("copy", self._path(source_key), self._path(target_key), *FOLDER_FLAGS)

    def delete(self, key: str) -> None:
        self._checked("purge", self._path(key))
