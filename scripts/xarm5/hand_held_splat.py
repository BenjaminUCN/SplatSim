"""Crea una escena con el gripper hand_held_XL330 sin volver a escanear el robot.

    python scripts/xarm5/hand_held_splat.py --scene xarm5_lite_plant_v3

Copia la escena: quita del splat las gaussianas del gripper del xArm, agrega gaussianas de colores
planos sobre las mallas del CAD y reordena las etiquetas por link. Agrega <scene>_hand_held a
configs/object_configs/objects.yaml. El original no se toca. Se ve como un CAD, no como una foto; un
escaneo real (README, "Adding a new robot") con el mismo URDF queda más parecido a la realidad.
"""
import argparse
import os
import shutil

import numpy as np
import pybullet as p
import yaml
from pybullet_utils.bullet_client import BulletClient

from splatsim.utils.hand_held import MESH_DIR, load_manifest, merge_splat, read_ply, read_stl, sample_surface, write_ply
from splatsim.utils.paths import load_yaml

OBJECTS = "configs/object_configs/objects.yaml"
DEFAULT_URDF = "~/data/xarm5_lite_urdf/ufactory_xarm5_urdf/xarm5_hand_held.urdf"


def link_indices(c, body) -> dict:
    return {c.getJointInfo(body, i)[12].decode(): i for i in range(c.getNumJoints(body))}


def latest_iteration(model_path: str) -> str:
    folders = [d for d in os.listdir(os.path.join(model_path, "point_cloud")) if d.startswith("iteration_")]
    return max(folders, key=lambda d: int(d.split("_")[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--scene", default="xarm5_lite_plant_v3")
    parser.add_argument("--new-scene", help="default: <scene>_hand_held")
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--spacing-mm", type=float, default=1.5)
    args = parser.parse_args()

    new_scene = args.new_scene or args.scene + "_hand_held"
    configs = load_yaml(OBJECTS)
    with open(OBJECTS, encoding="utf-8") as f:
        raw = yaml.safe_load(f)                 # sin expandir ~, para escribirlo igual
    if new_scene in raw:
        raise SystemExit(f"{new_scene} is already in {OBJECTS}: remove it first or use --new-scene")
    scene = configs[args.scene]
    urdf = os.path.expanduser(args.urdf)
    joint_states = scene["joint_states"][0]

    c = BulletClient(connection_mode=p.DIRECT)
    old = c.loadURDF(scene["urdf_path"][0], useFixedBase=True, basePosition=scene["base_position"][0])
    new = c.loadURDF(urdf, useFixedBase=True, basePosition=scene["base_position"][0])
    old_links, new_links = link_indices(c, old), link_indices(c, new)
    label_map = {i: new_links[name] for name, i in old_links.items() if name in new_links}
    print(f"Removed from the splat: {sorted(n for n in old_links if n not in new_links)}")

    # la pose en que se capturó el splat (igual que articulated_robot_pipeline.py); gripper en 0
    for i, value in enumerate(joint_states[:c.getNumJoints(new)]):
        c.resetJointState(new, i, value)

    def link_pose(name: str) -> np.ndarray:
        state = c.getLinkState(new, new_links[name], computeForwardKinematics=True)
        T = np.eye(4)
        T[:3, :3] = np.array(c.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        T[:3, 3] = state[4]
        return T

    # todas las mallas están en el marco del ensamble, que es el de hand_held_base
    base = link_pose("hand_held_base")
    rng = np.random.default_rng(0)
    xyz, rgb, labels = [], [], []
    for part in load_manifest():
        points = sample_surface(read_stl(os.path.join(MESH_DIR, part["file"])), args.spacing_mm, rng) / 1000
        world = points @ base[:3, :3].T + base[:3, 3]
        xyz.append(world)
        rgb.append(np.tile(part["rgb"], (len(world), 1)))
        labels.append(np.full(len(world), new_links["hand_held_" + part["link"]]))
        print(f"  {part['name']}: {len(world)} gaussians")
    c.disconnect()

    model_path = scene["model_path"]
    iteration = latest_iteration(model_path)
    vertices = read_ply(os.path.join(model_path, "point_cloud", iteration, "point_cloud.ply"))
    old_labels = np.load(f"data/labels_path/{args.scene}_labels.npy")
    merged, merged_labels, aabb, dropped = merge_splat(
        vertices, old_labels, scene["transformation"]["matrix"], scene["aabb"]["bounding_box"], label_map,
        np.concatenate(xyz), np.concatenate(rgb), np.concatenate(labels), args.spacing_mm / 1000 * 0.7)

    out = model_path.rstrip("/\\") + "_hand_held"
    shutil.copytree(model_path, out, ignore=lambda d, names: ["point_cloud"] if d == model_path else [],
                    dirs_exist_ok=True)
    write_ply(os.path.join(out, "point_cloud", iteration, "point_cloud.ply"), merged)
    np.save(f"data/labels_path/{new_scene}_labels.npy", merged_labels)

    entry = dict(raw[args.scene])
    entry["urdf_path"] = [args.urdf]
    entry["model_path"] = raw[args.scene]["model_path"].rstrip("/\\") + "_hand_held"
    entry["aabb"] = {**entry["aabb"], "bounding_box": aabb}
    entry["gripper"] = "hand_held_xl330"
    with open(OBJECTS, "a", encoding="utf-8") as f:
        f.write(yaml.safe_dump({new_scene: entry}, sort_keys=False, default_flow_style=None))
    print(f"{dropped} gaussians removed, {len(merged)} in total; {out}")
    print(f"Scene {new_scene} added to {OBJECTS}")


if __name__ == "__main__":
    main()
