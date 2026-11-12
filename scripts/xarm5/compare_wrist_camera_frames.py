"""Renders de la cámara de muñeca desde wrist_camera_link (el marco del sim) y desde donde queda la lente RGB
de la D435i atornillada al soporte (splatsim.utils.scene.hand_held.physical_wrist_camera_pose), lado a lado.

    python scripts/xarm5/compare_wrist_camera_frames.py
    python scripts/xarm5/compare_wrist_camera_frames.py --poses "0 -0.4886 -0.5235 -0.6108 0" "0 -0.1 -0.9 -0.3 0"

Cada pose da una imagen: marco del sim | lente física | mezcla 50/50. Los dos usan el mismo FoV
(WRIST_CAM_FOV), así que la diferencia es solo la pose. Necesita GPU.
"""
import argparse
import os
import socket

import numpy as np
import yaml
from PIL import Image, ImageDraw

from splatsim.utils.paths import OBJECTS_CONFIG
from splatsim.utils.scene.hand_held import physical_wrist_camera_pose

DEFAULT_SCENE = "xarm5_lite_plant_v3_hand_held_down"
OUT_DIR = "docs/experiments/wrist_camera_frame"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class PhysicalWristCamera:
    """La lente RGB de la D435i en el soporte: link_eef por physical_wrist_camera_pose()."""
    name = "wrist_physical"

    def __init__(self, server):
        from splatsim.robots.xarm.cameras import build_splat_camera
        from splatsim.robots.xarm.config import WRIST_CAM_FOV
        self.s, self._build, self._fov = server, build_splat_camera, WRIST_CAM_FOV
        c = server.pybullet_client
        names = [c.getJointInfo(server.dummy_robot, i)[12].decode() for i in range(c.getNumJoints(server.dummy_robot))]
        self.eef = names.index("link_eef")
        # con --camera-down (build_hand_held_urdf.py) el soporte está girado y wrist_camera_link en -x de link_eef
        camera = np.linalg.inv(link_pose(server, self.eef)) @ link_pose(server, names.index("wrist_camera_link"))
        self.camera_down = bool(camera[0, 3] < 0)

    def pose(self) -> np.ndarray:
        return link_pose(self.s, self.eef) @ physical_wrist_camera_pose(self.camera_down)

    def get(self):
        return self._build(self.s, self.pose(), self.name, 1, self._fov)


def link_pose(server, index: int) -> np.ndarray:
    c = server.pybullet_client
    st = c.getLinkState(server.dummy_robot, index, computeForwardKinematics=True)
    T = np.eye(4)
    T[:3, :3] = np.array(c.getMatrixFromQuaternion(st[5])).reshape(3, 3)
    T[:3, 3] = st[4]
    return T


def project(T_world_camera: np.ndarray, point, fov, size) -> tuple[float, float, float]:
    """Píxel (u, v) y profundidad de un punto del mundo, cámara pinhole con z hacia adelante e y hacia abajo."""
    p = T_world_camera[:3, :3].T @ (np.asarray(point) - T_world_camera[:3, 3])
    fx, fy = size[0] / 2 / np.tan(fov[0] / 2), size[1] / 2 / np.tan(fov[1] / 2)
    return fx * p[0] / p[2] + size[0] / 2, fy * p[1] / p[2] + size[1] / 2, p[2]


def to_image(tensor) -> Image.Image:
    array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255).astype(np.uint8))


def labelled(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 8 * len(text) + 12, 22), fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--poses", nargs="*", default=[],
                        help="5 arm joint angles (rad) per pose; default: the scene's initial pose")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    from splatsim.robots.xarm.config import IMAGE_H, IMAGE_W, WRIST_CAM_FOV
    from splatsim.robots.xarm.server import XArm5PybulletRobotServerCamera
    with open(OBJECTS_CONFIG) as f:
        use_gripper = yaml.safe_load(f)[args.scene].get("use_gripper", True)
    server = XArm5PybulletRobotServerCamera(
        port=free_port(), host="127.0.0.1", camera_names=["wrist_rgb"], robot_name=args.scene,
        use_gripper=use_gripper, gui="none", show_apple_window=False)
    server.setup_scene()
    server.cameras[PhysicalWristCamera.name] = PhysicalWristCamera(server)

    c = server.pybullet_client
    poses = [np.array(p.split(), float) for p in args.poses] or [np.array(server.initial_joint_state[:5])]
    os.makedirs(args.out_dir, exist_ok=True)
    for k, q in enumerate(poses):
        for i, value in enumerate(q):
            c.resetJointState(server.dummy_robot, i + 1, value * server.joint_signs[i])
        images = server.render_cameras(["wrist_rgb", PhysicalWristCamera.name])
        sim, physical = to_image(images["wrist_rgb"]), to_image(images[PhysicalWristCamera.name])
        blend = Image.blend(sim, physical, 0.5)
        row = Image.new("RGB", (sim.width * 3, sim.height))
        for j, (image, text) in enumerate(((sim, "sim frame: wrist_camera_link"),
                                           (physical, "physical D435i RGB lens"), (blend, "50/50 blend"))):
            row.paste(labelled(image, text), (j * sim.width, 0))
        path = os.path.join(args.out_dir, f"pose_{k}.jpg")
        row.save(path, quality=88)
        # dónde cae el centro del tomate (su caja en PyBullet) en cada imagen
        low, high = c.getAABB(server.apple.body_id)
        fruit = (np.array(low) + np.array(high)) / 2
        sim_px = project(link_pose(server, server.cameras["wrist_rgb"].link_index), fruit, WRIST_CAM_FOV, (IMAGE_W, IMAGE_H))
        phys_px = project(server.cameras[PhysicalWristCamera.name].pose(), fruit, WRIST_CAM_FOV, (IMAGE_W, IMAGE_H))
        shift = np.subtract(phys_px[:2], sim_px[:2])
        print(f"{path}: q = {np.round(q, 4).tolist()}")
        print(f"  tomato at {sim_px[2] * 100:.1f} cm: sim px ({sim_px[0]:.0f}, {sim_px[1]:.0f}), physical px "
              f"({phys_px[0]:.0f}, {phys_px[1]:.0f}), shift ({shift[0]:+.0f}, {shift[1]:+.0f}) = {np.hypot(*shift):.0f} px")


if __name__ == "__main__":
    main()
