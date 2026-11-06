"""Proveedores de GPU en la nube. El orquestador solo usa la interfaz de VastProvider; para agregar
otro proveedor basta otra clase con los mismos métodos."""
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["PATH"] = f"{os.environ.get('PATH', '')}:{Path.home() / '.local/bin'}"
READ_ATTEMPTS = 3
READ_RETRY_S = float(os.environ.get("SPLATSIM_API_RETRY_S", 5))


@dataclass
class Requirements:
    gpu_name: str = "RTX 5090"    # "" = cualquiera: para trabajos que solo usan CPU
    min_cores: int = 16           # el cargador de datos limita más que la GPU (6.30 vs 4.42 pasos/s)
    min_ram_gb: int = 48
    min_pcie_gbps: float = 0.0
    min_reliability: float = 0.97
    min_inet_down_mbps: int = 200  # la imagen pesa varios GB: una conexión lenta se paga por hora


@dataclass
class Offer:
    id: int
    machine_id: int
    gpu_name: str
    cpu_name: str
    cores: float
    ram_gb: float
    pcie_gbps: float
    dph: float                    # $/h con el disco pedido
    down_cost_gb: float           # $/GB de bajada (cada host pone su tarifa)
    up_cost_gb: float
    inet_down_mbps: float
    inet_up_mbps: float
    reliability: float
    location: str = ""


@dataclass
class Instance:
    id: int
    status: str                   # loading | running | exited | ...
    label: str = ""


class VastProvider:
    name = "vast"

    def _cli(self, *args: str) -> str:
        command = ["vastai", *args]
        # todo lo que sigue a --args va al contenedor, así que --raw debe ir antes
        command.insert(command.index("--args") if "--args" in command else len(command), "--raw")
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"vastai {args[0]} {args[1] if len(args) > 1 else ''}: sin respuesta en 120 s") from e
        if done.returncode:
            raise RuntimeError(f"vastai {args[0]} {args[1] if len(args) > 1 else ''}: "
                               f"{(done.stderr or done.stdout).strip()[-400:]}")
        return done.stdout

    def _read(self, *args: str):
        """Una consulta se puede repetir: la API a veces responde vacío o con un error pasajero."""
        for attempt in range(READ_ATTEMPTS):
            try:
                return json.loads(self._cli(*args))
            except (RuntimeError, json.JSONDecodeError) as e:
                if attempt == READ_ATTEMPTS - 1:
                    raise RuntimeError(f"vastai {args[0]} {args[1] if len(args) > 1 else ''}: {e}") from e
                time.sleep(READ_RETRY_S)

    def balance(self) -> float:
        return float(self._read("show", "user")["credit"])

    def search(self, req: Requirements, disk_gb: int) -> list[Offer]:
        query = (f"num_gpus=1 rentable=true verified=true "
                 f"cpu_cores_effective>={req.min_cores} cpu_ram>={req.min_ram_gb} "
                 f"disk_space>={disk_gb} reliability2>={req.min_reliability} "
                 f"inet_down>={req.min_inet_down_mbps}")
        if req.gpu_name:
            query += f" gpu_name={req.gpu_name.replace(' ', '_')} cuda_max_good>=12.8"
        if req.min_pcie_gbps:
            query += f" pcie_bw>={req.min_pcie_gbps}"
        raw = self._read("search", "offers", query, "-o", "dph_total", "--storage", str(disk_gb))
        return [Offer(id=o["id"], machine_id=o["machine_id"], gpu_name=o["gpu_name"], cpu_name=o["cpu_name"],
                      cores=o["cpu_cores_effective"], ram_gb=round(o["cpu_ram"] / 1024, 1),
                      pcie_gbps=o.get("pcie_bw") or 0.0, dph=o["dph_total"],
                      down_cost_gb=o["inet_down_cost"], up_cost_gb=o["inet_up_cost"],
                      inet_down_mbps=o["inet_down"], inet_up_mbps=o["inet_up"],
                      reliability=o["reliability2"], location=o.get("geolocation") or "")
                for o in raw]

    def create(self, offer: Offer, image: str, args: list[str], env: dict[str, str], disk_gb: int,
               label: str) -> int:
        """Lanza la imagen tal cual con su ENTRYPOINT: al terminar, la instancia se detiene."""
        env_arg = " ".join(f"-e {k}={v}" for k, v in env.items())
        reply = self._cli("create", "instance", str(offer.id), "--image", image, "--disk", str(disk_gb),
                          "--label", label, "--cancel-unavail", "--env", env_arg, "--args", *args)
        try:                 # no se repite: podría crear dos pods
            out = json.loads(reply)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"vastai create instance: respuesta inválida {reply[:200]!r}") from e
        if not out.get("success"):
            raise RuntimeError(f"vastai create instance: {out}")
        return int(out["new_contract"])

    def instances(self) -> list[Instance]:
        return [Instance(id=i["id"], status=i.get("actual_status") or "loading", label=i.get("label") or "")
                for i in self._read("show", "instances")]

    def destroy(self, instance_id: int) -> None:
        for attempt in range(READ_ATTEMPTS):      # repetirlo no hace daño, y mientras falle el pod cobra
            try:
                self._cli("destroy", "instance", str(instance_id), "-y")
                return
            except RuntimeError:
                if attempt == READ_ATTEMPTS - 1:
                    raise
                time.sleep(READ_RETRY_S)
