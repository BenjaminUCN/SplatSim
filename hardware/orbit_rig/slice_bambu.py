"""Lamina las placas del soporte orbital con Bambu Studio (CLI) para la A1 mini y escribe los .3mf.

    python3 hardware/orbit_rig/slice_bambu.py [--stl ~/data/cad/orbit_rig] [--bambu ~/.local/opt/bambu]

Necesita el AppImage de Bambu Studio extraído en <bambu>/squashfs-root y, si el sistema no tiene
WebKitGTK 4.1, sus .deb extraídos en <bambu>/libs (apt-get download + dpkg-deb -x, sin sudo).
Los perfiles de sistema heredan unos de otros ("inherits") y el CLI los quiere planos: aquí se
aplanan y se les aplican los cambios de PLATES. Escribe <stl>/bambu/<placa>.3mf (con los ajustes
dentro, listo para abrir en Bambu Studio) y una tabla con el tiempo y los gramos de cada placa.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

MACHINE = "Bambu Lab A1 mini 0.4 nozzle"
FILAMENT = "Bambu PLA Basic @BBL A1M"
FAST = {"top_shell_layers": "3", "top_shell_thickness": "0.6", "bottom_shell_layers": "2",
        "sparse_infill_density": "10%"}
PROCESS = "0.28mm Extra Draft @BBL A1M"
PLATES = {   # placa: (piezas, cambios al perfil)
    "1_base": (["base"], FAST),
    "2_rotor": (["rotor"], FAST),
    "3_brazo": (["brazo"], FAST),
    "4_arcos": (["arco_1", "arco_2", "arco_3", "arco_4"], FAST),
    "5_carro_poste": (["carro", "poste", "punta_pedestal"], {**FAST, "wall_loops": "3"}),
    "opcional_colgante": (["punta_colgante"], FAST),
    "opcional_pedestal_mano": (["pedestal_mano", "punta_pedestal"], {**FAST, "wall_loops": "3"}),
}


def flat(profiles: Path, kind: str, name: str) -> dict:
    for path in (profiles / kind).rglob("*.json"):
        data = json.loads(path.read_text())
        if data.get("name") == name:
            base = flat(profiles, kind, data["inherits"]) if data.get("inherits") else {}
            base.update(data)
            base.pop("inherits", None)
            return base
    raise SystemExit(f"No está el perfil {kind}/{name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", type=Path, default=Path.home() / "data" / "cad" / "orbit_rig")
    parser.add_argument("--bambu", type=Path, default=Path.home() / ".local" / "opt" / "bambu")
    parser.add_argument("--plates", nargs="+", choices=list(PLATES), help="solo estas placas (por defecto, todas)")
    args = parser.parse_args()
    stl, bambu = args.stl.expanduser(), args.bambu.expanduser()
    profiles = bambu / "squashfs-root" / "resources" / "profiles" / "BBL"
    out = stl / "bambu"
    out.mkdir(exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=f"{bambu}/libs/usr/lib/x86_64-linux-gnu:{bambu}/squashfs-root/bin")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for kind, name, file in (("machine", MACHINE, "machine.json"), ("filament", FILAMENT, "filament.json")):
            (tmp / file).write_text(json.dumps({**flat(profiles, kind, name), "from": "system"}))
        total_min = total_g = 0.0
        for plate, (parts, changes) in PLATES.items():
            if args.plates and plate not in args.plates:
                continue
            process = {**flat(profiles, "process", PROCESS), **changes, "from": "system", "name": f"orbit_rig {plate}"}
            (tmp / "process.json").write_text(json.dumps(process))
            work = tmp / plate
            work.mkdir()
            cmd = [str(bambu / "squashfs-root" / "bin" / "bambu-studio"), "--slice", "0",
                   "--load-settings", f"{tmp}/machine.json;{tmp}/process.json", "--load-filaments", f"{tmp}/filament.json",
                   "--arrange", "1", "--orient", "0", "--outputdir", str(work), "--export-3mf", f"{plate}.3mf",
                   *[str(stl / f"{p}.stl") for p in parts]]
            subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1200)
            gcode = (work / "plate_1.gcode").read_text(errors="ignore")[:20000]
            time = re.search(r"total estimated time: (.+)", gcode).group(1)
            grams = float(re.search(r"total filament weight \[g\] : ([\d.]+)", gcode).group(1))
            minutes = sum(int(v) * {"d": 1440, "h": 60, "m": 1, "s": 1 / 60}[u] for v, u in re.findall(r"(\d+)([dhms])", time))
            if (work / f"{plate}.3mf").exists():
                (work / f"{plate}.3mf").replace(out / f"{plate}.3mf")
            if not plate.startswith("opcional"):
                total_min, total_g = total_min + minutes, total_g + grams
            print(f"{plate:18s} {time:>12s} {grams:6.1f} g  {', '.join(parts)}")
    print(f"{'total':18s} {int(total_min // 60)}h {int(total_min % 60):02d}m {total_g:6.1f} g  (sin la opcional)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
