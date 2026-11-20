"""Mallas de la RealSense D435i para el URDF y el panel, desde el modelo de Intel.

    curl -LO https://raw.githubusercontent.com/IntelRealSense/realsense-ros/ros2-development/realsense2_description/meshes/d435.dae
    python scripts/xarm5/build_d435i_mesh.py d435.dae

La D435i usa la misma carcasa que la D435 (realsense2_description la dibuja con d435.dae). Escribe
d435i_body.stl (aluminio) y d435i_front.stl (vidrio frontal y lentes) en metros, en el marco óptico de
la cámara RGB, que es el que usan las cámaras del sim (x a la derecha de la imagen, y hacia abajo, z
hacia donde mira): un modelo puesto con la pose de una cámara queda con su lente RGB en esa pose.
"""
import argparse
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

OUT_DIR = "splatsim/robot_definitions/meshes/wrist_camera"
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
# en d435.dae (m): x hacia la izquierda de la cámara, y hacia arriba, z hacia adelante, vidrio en z = 0.
# Las lentes son geometrías propias: los dos IR (a 50 mm, la base estéreo), la RGB y el proyector
OPTICS = ("ID23", "ID29", "ID71", "ID89", "ID95", "ID101")
RGB_CENTER_DAE = (0.0325, 0.0, -0.0043)     # centro de la RGB; el cero de profundidad está 4.2 mm tras el vidrio
BODY_TRIANGLES = 30000
FRONT_TRIANGLES = 6000


def read_dae(path: str) -> dict:
    """Triángulos (N, 3, 3) por geometría; d435.dae no tiene transformaciones en los nodos."""
    root = ET.parse(path).getroot()
    if root.find(".//c:library_visual_scenes//c:matrix", NS) is not None:
        raise ValueError(f"{path} has node transforms; this reader ignores them")
    out = {}
    for geometry in root.iterfind(".//c:library_geometries/c:geometry", NS):
        mesh = geometry.find("c:mesh", NS)
        sources = {s.get("id"): s.find("c:float_array", NS) for s in mesh.findall("c:source", NS)}
        position = mesh.find("c:vertices/c:input[@semantic='POSITION']", NS).get("source")[1:]
        vertices = np.array(sources[position].text.split(), float).reshape(-1, 3)
        faces = []
        for triangles in mesh.findall("c:triangles", NS):
            inputs = triangles.findall("c:input", NS)
            stride = max(int(i.get("offset")) for i in inputs) + 1
            offset = next(int(i.get("offset")) for i in inputs if i.get("semantic") == "VERTEX")
            p = np.array(triangles.find("c:p", NS).text.split(), int).reshape(-1, stride)[:, offset]
            faces.append(p.reshape(-1, 3))
        out[geometry.get("id")] = vertices[np.concatenate(faces)]
    return out


def to_rgb_optical(triangles: np.ndarray) -> np.ndarray:
    """Del marco del .dae al óptico de la RGB: giro de 180° en z (x e y cambian de signo)."""
    p = triangles - np.array(RGB_CENTER_DAE)
    return p * np.array([-1.0, -1.0, 1.0])


def decimate(triangles: np.ndarray, target: int) -> np.ndarray:
    import open3d as o3d
    vertices, faces = np.unique(triangles.reshape(-1, 3).round(7), axis=0, return_inverse=True)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(faces.reshape(-1, 3)))
    mesh = mesh.simplify_quadric_decimation(target)
    return np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]


def write_stl(path: str, triangles: np.ndarray) -> None:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    data = np.zeros(len(triangles), np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    data["n"], data["v"] = normals, triangles.reshape(-1, 9)
    with open(path, "wb") as f:
        f.write(b"RealSense D435i, RGB optical frame, meters".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(triangles)))
        data.tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dae", help="d435.dae from IntelRealSense/realsense-ros (realsense2_description)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    parts = read_dae(args.dae)
    optics = np.concatenate([parts[k] for k in OPTICS])
    body = np.concatenate([t for k, t in parts.items() if k not in OPTICS])
    # la cara frontal de la carcasa es el vidrio negro: triángulos en z = 0 mirando hacia adelante
    normals = np.cross(body[:, 1] - body[:, 0], body[:, 2] - body[:, 0])
    front = (normals[:, 2] > 0.9 * np.linalg.norm(normals, axis=1)) & (body[:, :, 2].min(axis=1) > -0.0008)
    shell = decimate(body[~front], BODY_TRIANGLES)
    glass = decimate(np.concatenate([body[front], optics]), FRONT_TRIANGLES)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, triangles in (("d435i_body.stl", shell), ("d435i_front.stl", glass)):
        triangles = to_rgb_optical(triangles)
        write_stl(os.path.join(args.out_dir, name), triangles)
        low, high = triangles.reshape(-1, 3).min(0), triangles.reshape(-1, 3).max(0)
        print(f"{name}: {len(triangles)} triangles, {np.round(low, 4)} .. {np.round(high, 4)}")


if __name__ == "__main__":
    main()
