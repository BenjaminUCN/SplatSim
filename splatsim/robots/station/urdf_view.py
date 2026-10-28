"""El URDF del robot en la escena 3D del panel, moviéndose con las articulaciones del servidor.

Lo usa StationPanel si el backend tiene urdf_path. El backend entrega:
  urdf_joint_positions() -> {nombre de joint del URDF: ángulo}
  robot_base_pose()      -> (posición, cuaternión xyzw) de la base en el mundo, o None
Necesita yourdfpy (viser.extras.ViserUrdf).
"""
import numpy as np

JOINT_EPS = 1e-4   # rad: cambios menores no se envían al navegador


class RobotUrdfView:
    def __init__(self, viser_server, urdf_path: str, root: str = "/robot"):
        import yourdfpy
        from viser.extras import ViserUrdf

        # las mallas del URDF tienen rutas relativas: yourdfpy las busca junto al archivo
        urdf = yourdfpy.URDF.load(urdf_path)
        self.frame = viser_server.scene.add_frame(root, show_axes=False)
        self.urdf = ViserUrdf(viser_server, urdf, root_node_name=root)
        self.names = list(self.urdf.get_actuated_joint_names())
        self._last = None

    @property
    def visible(self) -> bool:
        return self.frame.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self.frame.visible = value           # oculta todo el árbol del robot

    def update(self, joints: dict, base_pose=None) -> None:
        cfg = np.array([joints.get(name, 0.0) for name in self.names], dtype=float)
        if self._last is None or np.abs(cfg - self._last).max() > JOINT_EPS:
            self.urdf.update_cfg(cfg)
            self._last = cfg
        if base_pose is not None:
            pos, q = base_pose
            self.frame.position = tuple(float(v) for v in pos)
            self.frame.wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
