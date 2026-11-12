"""Gripper hand_held_XL330 en el simulador: URDF del xArm5 con este gripper y splat sin escanear.

Solo numpy y la biblioteca estándar; los scripts son scripts/xarm5/build_hand_held_urdf.py y
scripts/xarm5/hand_held_splat.py. Las mallas están en mm, en el marco del ensamble CAD: y hacia la
punta del pruner, z hacia arriba de la tapa, cara de la brida del brazo en y = -51.63
(el disco mount_xarm5 va atornillado directo a bottom_plate).
"""
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np

MESH_DIR = "splatsim/robot_definitions/meshes/hand_held_xl330"
GRIPPER_BASE_LINK = "xarm_gripper_base_link"
KEEP_LINKS = ("wrist_camera_link",)
HORN_AXIS_MM = (12.7, 13.86524)
FLANGE_CENTER_MM = (0.0, -51.63, -15.8)
# el pruner real está trazado desde fotos sobre una cuadrícula (±1-2 mm); su mandíbula móvil gira alrededor del perno
PIVOT_MM = (9.66, 58.97)
JAW_JOINT = "hand_held_jaw_joint"
JAW_TRAVEL_RAD = 0.615          # 35.3°, medido entre las fotos abierta y cerrada
# mitad del borde dentado del yunque, a media altura de la mandíbula
GRIPPER_CENTER_MM = (-9.9, 97.0, -10.9)
# ensamble -> marco del gripper viejo: la punta sobre z (herramienta) y la z del ensamble sobre x, que apunta
# hacia arriba con la herramienta horizontal y joint5 = 0
R_FLANGE_ASSEMBLY = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
PLASTIC_KG_PER_MM3 = 1.24e-6                # PLA macizo: estimación, sin pesar la pieza
PART_MASS_KG = {"fixed_steel": 0.078, "moving_plate": 0.032, "moving_jaw": 0.035, "pivot_bolt": 0.008,
                "xl330": 0.018, "horn": 0.003, "tornillos_m2": 0.002}

# Soporte de la cámara de muñeca (Camera Stand_1300, placa de 3 mm en mm): el disco de r 37.5 centrado en el
# eje de la brida y la pestaña hacia +x de link_eef, con sus dos M3 a 45 mm (los de la espalda de la D435i).
CAMERA_MESH_DIR = "splatsim/robot_definitions/meshes/wrist_camera"
CAMERA_LINK = "wrist_camera_link"
STAND_MESH = "camera_stand_1300.stl"
# STL -> link_eef: x igual, el espesor (y del STL) sobre +z (hacia la herramienta), cara en la brida
T_EEF_STAND = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
# la placa va entre la brida y mount_xarm5: el gripper queda 3 mm más allá de la brida, sobre +z de link_eef
STAND_THICKNESS_M = 0.003
# mallas de scripts/xarm5/build_d435i_mesh.py, en el marco óptico de la RGB: (archivo, rgba)
D435I_MESHES = (("d435i_body.stl", "0.62 0.63 0.66 1"), ("d435i_front.stl", "0.03 0.03 0.04 1"))
D435I_DEPTH_M = 0.0251
# centro de la carcasa en el marco óptico de la RGB (la RGB está a 32.5 mm del centro, hacia un extremo)
D435I_CENTER_IN_RGB_M = (0.03245, 0.0, -0.00825)
# centro de la D435i en link_eef: entre los agujeros M3 (x = 67.5 mm) y con la espalda sobre la placa
D435I_CENTER_EEF_M = (0.0675, 0.0, STAND_THICKNESS_M + D435I_DEPTH_M / 2)
# ejes de la cámara en link_eef, los mismos de wrist_camera_link: x de la imagen sobre +y, mira hacia +z.
# Si la cámara está atornillada al revés (RGB del otro lado), es este giro más 180° en z
R_EEF_D435I = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])


def physical_wrist_camera_pose() -> np.ndarray:
    """Marco óptico de la RGB de la D435i montada en el soporte, en link_eef (4x4). wrist_camera_link
    (el marco que renderiza el sim) está en (0.08, 0, 0): esta es la pose donde queda la lente real."""
    T = np.eye(4)
    T[:3, :3] = R_EEF_D435I
    T[:3, 3] = np.array(D435I_CENTER_EEF_M) - R_EEF_D435I @ np.array(D435I_CENTER_IN_RGB_M)
    return T


def gripper_offset() -> np.ndarray:
    """La placa del soporte entre la brida y el gripper, en link_eef."""
    T = np.eye(4)
    T[2, 3] = STAND_THICKNESS_M
    return T


def _rot_rpy(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _rpy_of(R: np.ndarray) -> tuple[float, float, float]:
    p = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(p)) < 1e-9:
        return (math.atan2(-R[1, 2], R[1, 1]), p, 0.0)
    return (math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0]))


def origin_matrix(origin) -> np.ndarray:
    T = np.eye(4)
    if origin is not None:
        T[:3, :3] = _rot_rpy([float(v) for v in origin.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    return T


def _set_origin(element, T: np.ndarray) -> None:
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    origin.set("xyz", " ".join(f"{v:.8f}" for v in T[:3, 3]))
    origin.set("rpy", " ".join(f"{v:.8f}" for v in _rpy_of(T[:3, :3])))


def flange_from_assembly(roll_deg: float = 0.0) -> np.ndarray:
    """Marco del ensamble (en m) visto desde el marco donde iba el gripper viejo. roll_deg gira todo
    alrededor del eje de la herramienta, según cómo quede atornillado a la brida."""
    a = math.radians(roll_deg)
    roll = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = roll @ R_FLANGE_ASSEMBLY
    T[:3, 3] = -T[:3, :3] @ (np.array(FLANGE_CENTER_MM) / 1000)
    return T


def load_manifest(mesh_dir: str = MESH_DIR) -> list[dict]:
    with open(os.path.join(mesh_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)["parts"]


def _inertial(link, mass: float, com_m, inertia: float) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(f"{v:.6f}" for v in com_m), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.5f}")
    ET.SubElement(inertial, "inertia", ixx=f"{inertia:.3e}", iyy=f"{inertia:.3e}", izz=f"{inertia:.3e}",
                  ixy="0", ixz="0", iyz="0")


def _add_part(link, part: dict, shift_m, mesh_prefix: str) -> None:
    def geometry(parent):
        ET.SubElement(parent, "origin", xyz=" ".join(f"{v:.6f}" for v in shift_m), rpy="0 0 0")
        g = ET.SubElement(parent, "geometry")
        ET.SubElement(g, "mesh", filename=f"{mesh_prefix}/{part['file']}", scale="0.001 0.001 0.001")

    visual = ET.SubElement(link, "visual", name=part["name"])
    geometry(visual)
    material = ET.SubElement(visual, "material", name="hand_held_" + part["name"])
    ET.SubElement(material, "color", rgba=" ".join(str(v) for v in part["rgb"]) + " 1")
    if part.get("collision"):
        geometry(ET.SubElement(link, "collision", name=part["name"]))


def _mesh_visual(link, name: str, T: np.ndarray, filename: str, scale: str, rgba: str) -> None:
    visual = ET.SubElement(link, "visual", name=name)
    _set_origin(visual, T)
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=filename, scale=scale)
    ET.SubElement(ET.SubElement(visual, "material", name="wrist_camera_" + name), "color", rgba=rgba)


def add_wrist_camera_model(root, camera_parent: str, T_parent_camera: np.ndarray, mesh_prefix: str) -> None:
    """Cambia el prisma de wrist_camera_link por el soporte y una D435i, como visuales del mismo link: ni el
    marco de la cámara ni los índices de los links de PyBullet cambian. Las piezas se ubican en link_eef."""
    link = next(l for l in root.findall("link") if l.get("name") == CAMERA_LINK)
    if camera_parent != "link_eef":
        raise ValueError(f"{CAMERA_LINK} hangs from {camera_parent}; the stand is placed in link_eef")
    for element in link.findall("visual") + link.findall("collision"):
        link.remove(element)
    T_camera_eef = np.linalg.inv(T_parent_camera)

    _mesh_visual(link, "camera_stand", T_camera_eef @ T_EEF_STAND, f"{mesh_prefix}/{STAND_MESH}",
                 "0.001 0.001 0.001", "0.15 0.15 0.17 1")
    T_d435i = T_camera_eef @ physical_wrist_camera_pose()
    for filename, rgba in D435I_MESHES:
        _mesh_visual(link, filename.removesuffix(".stl"), T_d435i, f"{mesh_prefix}/{filename}", "1 1 1", rgba)


def _part_mass(part: dict) -> float:
    return PART_MASS_KG.get(part["name"], part["volume_mm3"] * PLASTIC_KG_PER_MM3)


def build_urdf(src: str, dst: str, travel_rad: float, mesh_dir: str = MESH_DIR, roll_deg: float = 0.0,
               extra_horn_mesh: str | None = None, keep_links=KEEP_LINKS) -> dict:
    """Escribe dst: src sin el gripper del xArm y con el hand_held_XL330, y copia las mallas al lado.
    Los links de keep_links (y lo que cuelga de ellos) siguen en la misma pose. extra_horn_mesh: STL
    (mm, marco del ensamble) de lo que gira con el horn, p.ej. la hoja móvil."""
    tree = ET.parse(src)
    root = tree.getroot()
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j.find("child").get("link"))
    links = {l.get("name"): l for l in root.findall("link")}
    if GRIPPER_BASE_LINK not in links:
        raise ValueError(f"{src} has no link {GRIPPER_BASE_LINK}; links: {sorted(links)}")

    def subtree(name: str) -> list[str]:
        out = [name]
        for child in children.get(name, []):
            out += subtree(child)
        return out

    kept = {n for k in keep_links if k in links for n in subtree(k)}
    removed = [n for n in subtree(GRIPPER_BASE_LINK) if n not in kept]
    base_joint = joints[GRIPPER_BASE_LINK]
    new_parent = base_joint.find("parent").get("link")
    T_parent_flange = origin_matrix(base_joint.find("origin"))

    # lo que se conserva y colgaba del gripper pasa al padre del gripper, en la misma pose
    reparented = []
    for name in kept:
        joint = joints[name]
        parent = joint.find("parent").get("link")
        if parent not in removed:
            continue
        T = origin_matrix(joint.find("origin"))
        while parent != new_parent:
            up = joints[parent]
            if up.get("type") != "fixed":
                raise ValueError(f"{name} hangs from the moving joint {up.get('name')}")
            T = origin_matrix(up.find("origin")) @ T
            parent = up.find("parent").get("link")
        joint.find("parent").set("link", new_parent)
        _set_origin(joint, T)
        reparented.append(name)

    removed_joints = set()
    for name in removed:
        root.remove(links[name])
        removed_joints.add(joints[name].get("name"))
        root.remove(joints[name])
    for tag in ("transmission", "gazebo"):      # solo si mencionan algo que ya no está
        for element in root.findall(tag):
            text = ET.tostring(element, encoding="unicode")
            if any(f'"{n}"' in text or f">{n}<" in text for n in [*removed, *removed_joints]):
                root.remove(element)

    # mallas junto al URDF, con ruta relativa
    prefix = "hand_held_xl330"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(dst)), prefix)
    os.makedirs(out_dir, exist_ok=True)
    parts = load_manifest(mesh_dir)
    for part in parts:
        shutil.copy2(os.path.join(mesh_dir, part["file"]), out_dir)
    base_parts = [p for p in parts if p["link"] == "base"]
    horn_parts = [p for p in parts if p["link"] == "horn"]
    jaw_parts = [p for p in parts if p["link"] == "jaw"]
    if extra_horn_mesh:
        shutil.copy2(extra_horn_mesh, out_dir)
        horn_parts.append({"name": "horn_extra", "file": os.path.basename(extra_horn_mesh), "link": "horn",
                           "rgb": [0.8, 0.8, 0.82], "collision": True, "volume_mm3": 0, "com_mm": None})

    base = ET.SubElement(root, "link", name="hand_held_base")
    for part in base_parts:
        _add_part(base, part, (0, 0, 0), prefix)
    masses = np.array([_part_mass(p) for p in base_parts])
    com = sum(m * np.array(p["com_mm"]) for m, p in zip(masses, base_parts)) / masses.sum() / 1000
    _inertial(base, float(masses.sum()), com, 1e-3)

    horn_origin_m = np.array([*HORN_AXIS_MM, 0.0]) / 1000
    horn = ET.SubElement(root, "link", name="hand_held_horn")
    for part in horn_parts:
        _add_part(horn, part, -horn_origin_m, prefix)
    _inertial(horn, PART_MASS_KG["horn"], (0, 0, -0.011), 1e-6)

    if jaw_parts:
        # las mallas están con la mandíbula cerrada; q = 0 es abierta, como el horn
        pivot_m = np.array([*PIVOT_MM, 0.0]) / 1000
        jaw = ET.SubElement(root, "link", name="hand_held_jaw")
        for part in jaw_parts:
            _add_part(jaw, part, -pivot_m, prefix)
        jaw_masses = np.array([_part_mass(p) for p in jaw_parts])
        jaw_com = sum(m * np.array(p["com_mm"]) for m, p in zip(jaw_masses, jaw_parts)) / jaw_masses.sum() / 1000
        _inertial(jaw, float(jaw_masses.sum()), jaw_com - pivot_m, 1e-4)
        jaw_joint = ET.SubElement(root, "joint", name=JAW_JOINT, type="revolute")
        ET.SubElement(jaw_joint, "origin", xyz=" ".join(f"{v:.6f}" for v in pivot_m), rpy=f"0 0 {-JAW_TRAVEL_RAD:.5f}")
        ET.SubElement(jaw_joint, "parent", link="hand_held_base")
        ET.SubElement(jaw_joint, "child", link="hand_held_jaw")
        ET.SubElement(jaw_joint, "axis", xyz="0 0 1")
        ET.SubElement(jaw_joint, "limit", lower="0", upper=f"{JAW_TRAVEL_RAD:.5f}", effort="5", velocity="5")

    center = ET.SubElement(root, "link", name="gripper_center_link")
    _inertial(center, 0.001, (0, 0, 0), 1e-8)

    fix = ET.SubElement(root, "joint", name="hand_held_fix", type="fixed")
    _set_origin(fix, gripper_offset() @ T_parent_flange @ flange_from_assembly(roll_deg))
    ET.SubElement(fix, "parent", link=new_parent)
    ET.SubElement(fix, "child", link="hand_held_base")

    drive = ET.SubElement(root, "joint", name="gripper_drive_joint", type="revolute")
    ET.SubElement(drive, "origin", xyz=" ".join(f"{v:.6f}" for v in horn_origin_m), rpy="0 0 0")
    ET.SubElement(drive, "parent", link="hand_held_base")
    ET.SubElement(drive, "child", link="hand_held_horn")
    ET.SubElement(drive, "axis", xyz="0 0 1")
    # 0.52 N·m y 10.8 rad/s (103 rpm): XL330-M288 a 5 V
    ET.SubElement(drive, "limit", lower="0", upper=f"{travel_rad:.5f}", effort="0.52", velocity="10.8")

    center_joint = ET.SubElement(root, "joint", name="gripper_center_fix", type="fixed")
    ET.SubElement(center_joint, "origin", xyz=" ".join(f"{v / 1000:.6f}" for v in GRIPPER_CENTER_MM), rpy="0 0 0")
    ET.SubElement(center_joint, "parent", link="hand_held_base")
    ET.SubElement(center_joint, "child", link="gripper_center_link")

    if CAMERA_LINK in kept:
        camera_prefix = "wrist_camera"
        camera_dir = os.path.join(os.path.dirname(os.path.abspath(dst)), camera_prefix)
        os.makedirs(camera_dir, exist_ok=True)
        for filename in (STAND_MESH, *(f for f, _ in D435I_MESHES)):
            shutil.copy2(os.path.join(CAMERA_MESH_DIR, filename), camera_dir)
        camera_joint = joints[CAMERA_LINK]
        add_wrist_camera_model(root, camera_joint.find("parent").get("link"),
                               origin_matrix(camera_joint.find("origin")), camera_prefix)

    ET.indent(tree)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return {"removed": removed, "reparented": reparented, "parent": new_parent,
            "mass_kg": float(masses.sum() + sum(_part_mass(p) for p in jaw_parts)),
            "parts": [*base_parts, *horn_parts, *jaw_parts]}


# ── splat ────────────────────────────────────────────────────────────────────
def read_stl(path: str) -> np.ndarray:
    """Triángulos (N, 3, 3) de un STL binario."""
    with open(path, "rb") as f:
        f.seek(80)
        n = int(np.fromfile(f, "<u4", 1)[0])
        data = np.fromfile(f, np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), n)
    return data["v"].reshape(-1, 3, 3).astype(float)


def sample_surface(triangles: np.ndarray, spacing: float, rng) -> np.ndarray:
    """Puntos al azar sobre la superficie, uno por spacing² de área."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    areas = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    n = max(int(areas.sum() / spacing**2), 1)
    index = rng.choice(len(triangles), n, p=areas / areas.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[index] + u[:, None] * (b - a)[index] + v[:, None] * (c - a)[index]


def read_ply(path: str):
    """Vértices de un .ply binario de 3DGS como arreglo estructurado."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fields, n = [], None
        while (line := f.readline().strip()) != b"end_header":
            words = line.decode().split()
            if words[:2] == ["format", "ascii"] or words[:2] == ["format", "binary_big_endian"]:
                raise ValueError(f"{path}: only binary little endian PLY is supported")
            if words[:2] == ["element", "vertex"]:
                n = int(words[2])
            elif words[0] == "property":
                if words[1] != "float":
                    raise ValueError(f"{path}: unexpected property {words}")
                fields.append((words[2], "<f4"))
        return np.fromfile(f, np.dtype(fields), n)


def write_ply(path: str, vertices) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}"]
    header += [f"property float {name}" for name in vertices.dtype.names] + ["end_header", ""]
    with open(path, "wb") as f:
        f.write("\n".join(header).encode())
        vertices.tofile(f)


def in_box(xyz: np.ndarray, aabb) -> np.ndarray:
    low, high = np.asarray(aabb[0]), np.asarray(aabb[1])
    return np.all((xyz > low) & (xyz < high), axis=1)


def merge_splat(vertices, labels, matrix, aabb, label_map: dict, new_xyz_world, new_rgb, new_labels,
                radius_m: float, opacity: float = 0.98):
    """El splat con el gripper viejo fuera y las gaussianas nuevas agregadas.

    labels es por gaussiana dentro de aabb (get_segmented_indices). label_map: índice de link viejo
    -> nuevo; los links viejos que no están se borran del splat. Devuelve (vértices, labels, aabb,
    gaussianas borradas); -1 marca lo que entró a la caja nueva sin ser del robot, y no se mueve."""
    M = np.asarray(matrix, dtype=float)
    R, t = M[:3, :3], M[:3, 3]
    scale = np.cbrt(np.linalg.det(R))
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(float)
    world = xyz @ R.T + t
    inside = in_box(world, aabb)
    if inside.sum() != len(labels):
        raise ValueError(f"{len(labels)} labels but {inside.sum()} gaussians inside the aabb: "
                         "the labels are not of this splat")

    full = np.full(len(vertices), -1, dtype=np.int64)
    full[inside] = [label_map.get(int(k), -2) for k in labels]
    keep = full != -2

    n = len(new_xyz_world)
    added = np.zeros(n, dtype=vertices.dtype)
    inv = np.linalg.inv(M)
    splat = np.asarray(new_xyz_world) @ inv[:3, :3].T + inv[:3, 3]
    added["x"], added["y"], added["z"] = splat.T
    dc = (np.asarray(new_rgb) - 0.5) / 0.28209479177387814
    added["f_dc_0"], added["f_dc_1"], added["f_dc_2"] = dc.T
    added["opacity"] = math.log(opacity / (1 - opacity))
    for i in range(3):
        added[f"scale_{i}"] = math.log(radius_m / scale)
    added["rot_0"] = 1.0       # cuaternión identidad: son esféricas

    merged = np.concatenate([vertices[keep], added])
    merged_labels = np.concatenate([full[keep], np.asarray(new_labels, dtype=np.int64)])

    margin = 0.002
    low = np.minimum(np.asarray(aabb[0]), np.min(new_xyz_world, axis=0) - margin)
    high = np.maximum(np.asarray(aabb[1]), np.max(new_xyz_world, axis=0) + margin)
    new_aabb = [[round(float(v), 4) for v in low], [round(float(v), 4) for v in high]]
    merged_xyz = np.stack([merged["x"], merged["y"], merged["z"]], axis=1).astype(float)
    selected = in_box(merged_xyz @ R.T + t, new_aabb)
    return merged, merged_labels[selected], new_aabb, int((~keep).sum())
