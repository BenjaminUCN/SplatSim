"""Reparte los episodios de un script de renderizado entre varios procesos (--workers N).

render_recordings.py y mirror_recordings.py renderizan un episodio tras otro en un proceso, y su
cuello de botella es la CPU (leer los .pkl, la física, comprimir los .png): la GPU queda a ~25 %.
Con --workers N el proceso principal lanza N copias del mismo script, cada una con su parte de
los episodios (--episodes ...) y su propia escena cargada en la GPU (~3 GB cada una), y reimprime
su salida con el avance total ([k/total]) para que el panel de la cola lo siga como siempre.
"""
import re
import subprocess
import sys
import threading

PROGRESS = re.compile(r"^\[(\d+)/(\d+)\]")


def own_argv(argv: list[str]) -> list[str]:
    """sys.argv sin --episodes ni --workers (cada proceso recibe los suyos)."""
    out, skip = [], False
    for arg in argv:
        if arg.startswith("--"):
            skip = arg.split("=")[0] in ("--episodes", "--workers")
            if skip:
                continue
        if not skip:
            out.append(arg)
    return out


def split(names: list[str], workers: int) -> list[list[str]]:
    """Reparto intercalado (0, N, 2N... al primero), así cada uno recibe largos parecidos."""
    return [part for part in (names[i::workers] for i in range(workers)) if part]


def run_workers(argv: list[str], names: list[str], workers: int, total: int) -> int:
    """Corre `python argv --episodes <parte> --workers 1` por cada parte; total = líneas de
    avance esperadas entre todos. Devuelve 0 si todos terminan bien."""
    parts = split(names, workers)
    print(f"{len(names)} episodios en {len(parts)} procesos", flush=True)
    lock = threading.Lock()
    done = [0]

    def forward(k: int, proc: subprocess.Popen):
        for line in proc.stdout:
            line = line.rstrip("\n")
            with lock:
                if PROGRESS.match(line):
                    done[0] += 1
                    line = PROGRESS.sub(f"[{done[0]}/{total}]", line)
                print(f"{line} (w{k})" if PROGRESS.match(line) else f"[w{k}] {line}", flush=True)

    procs, threads = [], []
    for k, part in enumerate(parts):
        proc = subprocess.Popen([sys.executable, "-u", *argv, "--episodes", *part, "--workers", "1"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", stdin=subprocess.DEVNULL)
        t = threading.Thread(target=forward, args=(k, proc), daemon=True)
        t.start()
        procs.append(proc)
        threads.append(t)
    codes = [p.wait() for p in procs]
    for t in threads:
        t.join()
    failed = [k for k, c in enumerate(codes) if c != 0]
    if failed:
        print(f"Fallaron los procesos {failed} (códigos {[codes[k] for k in failed]})", flush=True)
    return 1 if failed else 0
