"""El URDF del robot en la escena 3D del panel, moviéndose con las articulaciones del servidor.

Lo usa StationPanel si el backend tiene urdf_path. El backend entrega:
  urdf_joint_positions() -> {nombre de joint del URDF: ángulo}
  robot_base_pose()      -> (posición, cuaternión xyzw) de la base en el mundo, o None
Necesita yourdfpy (viser.extras.ViserUrdf).
"""
import numpy as np

JOINT_EPS = 1e-4   # rad: cambios menores no se envían al navegador
GRIPPER_MAX_ANGLE = 0.85 # rad del drive_joint con el gripper normalizado en 1 (igual que el sim)
GRIPPER_JOINTS = ("drive_joint", "left_finger_joint", "left_inner_knuckle_joint",
                  "right_outer_knuckle_joint", "right_finger_joint", "right_inner_knuckle_joint")


def xarm5_urdf_joints(q) -> dict:
    """joint_positions del xArm5 (5 articulaciones + gripper normalizado en [0, 1]) -> ángulo por
    nombre de joint del URDF."""
    q = np.asarray(q, dtype=float)
    joints = {f"joint{i + 1}": q[i] for i in range(5)}
    joints.update({name: q[5] * GRIPPER_MAX_ANGLE for name in GRIPPER_JOINTS})
    return joints


class RobotUrdfView:
    def __init__(self, viser_server, urdf_path: str, root: str = "/robot",
                 color: tuple[float, float, float] | None = None, opacity: float = 1.0):
        """color (RGB en [0, 1]): todas las mallas de ese color, con opacidad ajustable (viser solo
        la permite en mallas simples, no en las del URDF con sus materiales); None: los colores
        del URDF, opaco."""
        import yourdfpy
        from viser.extras import ViserUrdf

        # las mallas del URDF tienen rutas relativas: yourdfpy las busca junto al archivo
        urdf = yourdfpy.URDF.load(urdf_path)
        self.frame = viser_server.scene.add_frame(root, show_axes=False)
        self.urdf = ViserUrdf(viser_server, urdf, root_node_name=root,
                              mesh_color_override=None if color is None else (*color, opacity))
        self.names = list(self.urdf.get_actuated_joint_names())
        self._last = None

    @property
    def visible(self) -> bool:
        return self.frame.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self.frame.visible = value           # oculta todo el árbol del robot

    @property
    def opacity(self) -> float:
        meshes = self.urdf._meshes
        return meshes[0].opacity if meshes and meshes[0].opacity is not None else 1.0

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Solo con color (mallas simples)."""
        for mesh in self.urdf._meshes:      # ViserUrdf no expone sus mallas
            mesh.opacity = float(value)

    def update(self, joints: dict, base_pose=None) -> None:
        cfg = np.array([joints.get(name, 0.0) for name in self.names], dtype=float)
        if self._last is None or np.abs(cfg - self._last).max() > JOINT_EPS:
            self.urdf.update_cfg(cfg)
            self._last = cfg
        if base_pose is not None:
            pos, q = base_pose
            self.frame.position = tuple(float(v) for v in pos)
            self.frame.wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
