"""Malla con color de un objeto a partir de su splat, para verlo en PyBullet y en el panel.

Las gaussianas (ya en el marco del cuerpo, en metros) se acumulan en una grilla de vóxeles pesadas por su
opacidad; la densidad suavizada se umbraliza, se rellena por dentro (el splat de un fruto es una cáscara) y
se queda la componente más grande. marching cubes da la superficie, que se suaviza y se simplifica. El color
de cada vértice es el de las gaussianas más cercanas.

PyBullet no usa los colores por vértice de un .obj, solo texturas: cada cara se pinta con un color plano,
el de una paleta de PALETTE_LEVELS³ colores guardada como textura (sin desdoblar la malla en UV).
"""
from pathlib import Path

import numpy as np

PALETTE_LEVELS = 16          # niveles por canal de la paleta: 16³ = 4096 colores, textura de 64x64
PALETTE_SIDE = 64


def splat_mesh(xyz: np.ndarray, opacity: np.ndarray, rgb: np.ndarray, lo, hi, voxel: float = 0.0015,
               sigma: float = 2.0, threshold: float = 0.015, min_opacity: float = 0.02,
               faces: int = 20000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vértices, caras, color por vértice en [0, 1]) de la superficie de las gaussianas dentro de [lo, hi].

    voxel en metros; sigma, el suavizado de la densidad en vóxeles; threshold, la densidad (suma de
    opacidades por vóxel, suavizada) del borde. Con menos threshold el objeto engorda y cierra huecos."""
    import open3d as o3d
    import trimesh
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage import measure

    lo, hi = np.asarray(lo, float) - 2 * voxel * sigma, np.asarray(hi, float) + 2 * voxel * sigma
    keep = (opacity > min_opacity) & np.all((xyz > lo) & (xyz < hi), axis=1)
    if not keep.any():
        raise ValueError("no hay gaussianas dentro de la caja")
    points, weights, colors = xyz[keep], opacity[keep], rgb[keep]

    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    index = np.clip(((points - lo) / voxel).astype(int), 0, shape - 1)
    density = np.zeros(shape)
    np.add.at(density, tuple(index.T), weights)
    density = ndimage.gaussian_filter(density, sigma)
    solid = ndimage.binary_fill_holes(ndimage.binary_closing(density > threshold, iterations=2))
    labels, n = ndimage.label(solid)
    if n == 0:
        raise ValueError(f"ningún vóxel supera la densidad {threshold}")
    solid = labels == 1 + np.argmax(ndimage.sum(solid, labels, range(1, n + 1)))

    vertices, triangles, _, _ = measure.marching_cubes(np.pad(solid, 1).astype(np.float32), 0.5)
    vertices = (vertices - 1) * voxel + lo
    mesh = trimesh.Trimesh(vertices, triangles)
    trimesh.smoothing.filter_taubin(mesh, iterations=10)
    if len(mesh.faces) > faces:
        o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                      o3d.utility.Vector3iVector(mesh.faces))
        o = o.simplify_quadric_decimation(faces)
        mesh = trimesh.Trimesh(np.asarray(o.vertices), np.asarray(o.triangles))

    distance, nearest = cKDTree(points).query(mesh.vertices, k=8)
    w = weights[nearest] / (distance + 1e-4)
    vertex_colors = (colors[nearest] * w[..., None]).sum(1) / w.sum(1)[:, None]
    return np.asarray(mesh.vertices), np.asarray(mesh.faces), np.clip(vertex_colors, 0, 1)


def palette_texture() -> np.ndarray:
    """La paleta como imagen RGB de PALETTE_SIDE x PALETTE_SIDE: el píxel k es el color k."""
    k = np.arange(PALETTE_LEVELS ** 3)
    rgb = np.stack([k // PALETTE_LEVELS ** 2, (k // PALETTE_LEVELS) % PALETTE_LEVELS, k % PALETTE_LEVELS], 1)
    return (rgb * 255 / (PALETTE_LEVELS - 1)).round().astype(np.uint8).reshape(PALETTE_SIDE, PALETTE_SIDE, 3)


def face_uv(face_colors: np.ndarray) -> np.ndarray:
    """Coordenada de textura (centro del píxel de la paleta) del color de cada cara."""
    q = np.clip(np.round(face_colors * (PALETTE_LEVELS - 1)), 0, PALETTE_LEVELS - 1).astype(int)
    k = (q[:, 0] * PALETTE_LEVELS + q[:, 1]) * PALETTE_LEVELS + q[:, 2]
    return np.stack([(k % PALETTE_SIDE + 0.5) / PALETTE_SIDE, 1 - (k // PALETTE_SIDE + 0.5) / PALETTE_SIDE], 1)


def write_textured_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray) -> None:
    """path (.obj), con su .mtl y la paleta (<nombre>.png) al lado; cada cara con el color medio de sus vértices."""
    from PIL import Image

    path = Path(path)
    texture, material = path.with_suffix(".png"), path.with_suffix(".mtl")
    Image.fromarray(palette_texture()).save(texture)
    material.write_text(f"newmtl splat\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd {texture.name}\n")
    uv = face_uv(vertex_colors[faces].mean(axis=1))
    lines = [f"mtllib {material.name}", "usemtl splat"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uv]
    lines += [f"f {a + 1}/{i + 1} {b + 1}/{i + 1} {c + 1}/{i + 1}" for i, (a, b, c) in enumerate(faces)]
    path.write_text("\n".join(lines) + "\n")


def write_object_urdf(folder: Path, template_urdf: Path, vertices: np.ndarray, faces: np.ndarray,
                      vertex_colors: np.ndarray) -> Path:
    """folder/model.urdf: el URDF de template_urdf (masa, roce, colisión) con la malla del splat como visual.
    La malla de colisión del template se copia al lado; también se guarda visual_colors.ply, la malla con
    color por vértice (la que muestra el panel)."""
    import shutil
    import xml.etree.ElementTree as ET

    import trimesh

    folder, template_urdf = Path(folder), Path(template_urdf)
    folder.mkdir(parents=True, exist_ok=True)
    write_textured_obj(folder / "visual.obj", vertices, faces, vertex_colors)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.visual.vertex_colors = np.c_[(vertex_colors * 255).round(), np.full(len(vertices), 255)].astype(np.uint8)
    mesh.export(folder / "visual_colors.ply")

    tree = ET.parse(template_urdf)
    links = tree.getroot().findall("link")
    if len(links) != 1:
        raise ValueError(f"{template_urdf}: se esperaba un solo link, tiene {len(links)}")
    link = links[0]
    for visual in link.findall("visual"):
        link.remove(visual)
    for mesh_tag in link.iter("mesh"):          # la colisión: se copia su malla al lado del URDF nuevo
        name = Path(mesh_tag.get("filename"))
        shutil.copy(template_urdf.parent / name, folder / f"collision{name.suffix}")
        mesh_tag.set("filename", f"collision{name.suffix}")
    visual = ET.SubElement(link, "visual")
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename="visual.obj", scale="1 1 1")
    ET.indent(tree)
    out = folder / "model.urdf"
    tree.write(out, encoding="unicode")
    return out
