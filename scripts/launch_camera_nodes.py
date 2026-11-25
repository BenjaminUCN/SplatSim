from dataclasses import dataclass
from multiprocessing import Process

import tyro

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.zmq_core.camera_node import ZMQServerCamera


@dataclass
class Args:
    hostname: str = "127.0.0.1"
    # hostname: str = "128.32.175.167"
    # configs/xarm5_real.yaml: cada cámara con serial y puerto fijos (si no, puertos desde 5000
    # en el orden de detección, que no es estable)
    config: str | None = None


def launch_server(port: int, camera_id: int, args: Args, size: tuple[int, int] = (640, 480)):
    camera = RealSenseCamera(camera_id, width=size[0], height=size[1])
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera server on port {port}")
    server.serve()


def config_ports(path: str, ids: list[str]) -> list[tuple[str, int, tuple[int, int]]]:
    """(serial, puerto, (ancho, alto)) de cada cámara del archivo; sin serial: las detectadas que sobren, en orden.
    El tamaño: el "size" de la cámara o camera_size del archivo (por defecto 640x480)."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cameras = cfg.get("cameras") or {}
    default_size = tuple(cfg.get("camera_size") or (640, 480))
    free = [i for i in ids if i not in {c.get("serial") for c in cameras.values()}]
    out = []
    for name, cam in cameras.items():
        serial = cam.get("serial") or (free.pop(0) if free else None)
        if serial not in ids:
            raise SystemExit(f"Cámara {name}: serial {serial} no conectado (detectadas: {ids})")
        size = tuple(cam.get("size") or default_size)
        print(f"{name}: serial {serial} -> puerto {cam['port']}, {size[0]}x{size[1]}")
        out.append((serial, cam["port"], size))
    return out


def main(args):
    ids = get_device_ids()
    if args.config:
        ports = config_ports(args.config, ids)
    else:
        ports = [(camera_id, 5000 + i, (640, 480)) for i, camera_id in enumerate(ids)]
    camera_servers = []
    for camera_id, camera_port, size in ports:
        # start a python process for each camera
        print(f"Launching camera {camera_id} on port {camera_port}")
        camera_servers.append(
            Process(target=launch_server, args=(camera_port, camera_id, args, size))
        )

    for server in camera_servers:
        server.start()


if __name__ == "__main__":
    main(tyro.cli(Args))
