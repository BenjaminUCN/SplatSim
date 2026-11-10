"""Recorta hojas de un splat con clics, para usarlas como oclusores en la simulación.

    python scripts/xarm5/segment_leaf.py [--robot_name xarm5_lite_plant_v3] [--port 8083]

Abre http://localhost:<port>: el splat de la escena, visto desde base_rgb. Con Pick activo, un clic
sobre una hoja hace crecer la selección (en rojo) por su superficie; los parámetros recalculan el
último clic. Save escribe ~/data/occluders/<escena>/<nombre>.ply (+ .yaml), que el panel del
simulador agrega a la escena con launch_nodes.py --occluders configs/occluders.yaml.
"""
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del repo
from splatsim.utils.render.camera_setup import base_camera_poses  # noqa: E402
from splatsim.utils.scene.leaf_segmentation import (  # noqa: E402
    GrowParams,
    LeafSelection,
    SplatCloud,
    crop_cloud,
    leaf_summary,
    next_leaf_name,
    occluder_path,
    pick_surface,
    save_leaf,
)
from splatsim.utils.paths import OBJECTS_CONFIG, load_yaml  # noqa: E402

SELECTION_RGB = (255, 40, 40)
TOO_FEW = 100  # en un brillo casi blanco no hay color de referencia y casi no crece
MODES = ("Grow", "Brush add", "Brush erase")


@dataclass
class Args:
    robot_name: str = "xarm5_lite_plant_v3"
    """escena de configs/object_configs/objects.yaml"""
    port: int = 8083
    box_min: tuple[float, float, float] = (-1.6, -1.0, -0.6)
    box_max: tuple[float, float, float] = (-0.2, 1.0, 1.0)
    """caja de mundo (m) que se muestra y se puede elegir; por defecto, las plantas"""


def latest_ply(model_path: str) -> Path:
    runs = sorted((Path(model_path).expanduser() / "point_cloud").glob("iteration_*"),
                  key=lambda p: int(p.name.split("_")[1]))
    if not runs:
        raise SystemExit(f"No hay point_cloud/iteration_* en {model_path}")
    return runs[-1] / "point_cloud.ply"


class LeafApp:
    def __init__(self, args: Args):
        config = load_yaml(OBJECTS_CONFIG)[args.robot_name]
        ply = latest_ply(config["model_path"])
        print(f"Cargando {ply}")
        whole = SplatCloud.from_ply(ply, config["transformation"]["matrix"])
        self.cloud = crop_cloud(whole, args.box_min, args.box_max)
        print(f"{len(self.cloud)} de {len(whole)} gaussianas en la caja")
        self.scene_name = args.robot_name
        self.selection = LeafSelection(self.cloud)
        self.lock = threading.Lock()
        self.preview = False

        self.v = viser.ViserServer(port=args.port, label="Segment leaf", verbose=False)
        self.v.scene.set_up_direction("+z")
        self._add_splat("/scene", np.arange(len(self.cloud)))
        self.overlay = None
        self.leaf_node = None
        self._build_gui()
        self.v.on_client_connect(self._on_connect)
        print(f"Abre http://localhost:{args.port}")

    def _add_splat(self, name: str, indices: np.ndarray):
        c = self.cloud
        return self.v.scene.add_gaussian_splats(
            name, centers=c.xyz[indices], covariances=c.covariances(indices),
            rgbs=c.rgb[indices], opacities=c.opacity[indices, None])

    def _on_connect(self, client) -> None:
        # arranca viendo lo que ve la cámara central; viser usa la misma convención (z adelante)
        xyz, rpy = base_camera_poses().get("base_rgb", ((-0.079, 0.0, 0.142), (-1.562, 0.0, -4.708)))
        xyzw = Rotation.from_euler("xyz", rpy).as_quat()
        client.camera.position = xyz
        client.camera.wxyz = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])

    def _build_gui(self) -> None:
        gui = self.v.gui
        self.pick = gui.add_checkbox(
            "Pick", True, hint="On: a click selects (dragging still moves the view). Off: only view")
        self.mode = gui.add_dropdown("Click does", list(MODES))
        with gui.add_folder("Grow"):
            d = GrowParams()
            self.normal = gui.add_slider("Max bend (deg)", 5.0, 60.0, 1.0, d.normal_deg,
                                         hint="Lower it if the selection spills onto other leaves")
            self.plane = gui.add_slider("Max step (mm)", 0.5, 8.0, 0.5, d.plane_mm,
                                        hint="Distance off the local surface: separates leaves that overlap")
            self.color = gui.add_slider("Color tolerance", 5.0, 50.0, 1.0, d.color_tol)
            self.link = gui.add_slider("Link (mm)", 2.0, 15.0, 0.5, d.link_mm,
                                       hint="Raise it if the selection stops before the edge of the leaf")
            self.radius = gui.add_slider("Max radius (cm)", 5.0, 40.0, 1.0, d.max_radius_cm)
        self.brush_cm = gui.add_slider("Brush radius (cm)", 0.3, 5.0, 0.1, 1.0)
        undo = gui.add_button("Undo")
        clear = gui.add_button("Clear")
        self.only_leaf = gui.add_checkbox("Show only selection", False,
                                          hint="The selection as it will render, without the scene")
        self.name = gui.add_text("Name", next_leaf_name(self.scene_name))
        save = gui.add_button("Save Leaf", color="green")
        self.info = gui.add_markdown("Click a leaf.")

        self.v.scene.on_click()(self._on_click)
        for slider in (self.normal, self.plane, self.color, self.link, self.radius):
            slider.on_update(lambda _: self._regrow())
        undo.on_click(lambda _: self._edit(self.selection.undo))
        clear.on_click(lambda _: self._edit(self.selection.clear))
        self.only_leaf.on_update(lambda _: self._refresh())
        save.on_click(lambda _: self._save())

    def _params(self) -> GrowParams:
        return GrowParams(link_mm=self.link.value, normal_deg=self.normal.value, plane_mm=self.plane.value,
                          color_tol=self.color.value, max_radius_cm=self.radius.value)

    def _on_click(self, event) -> None:
        if not self.pick.value:
            return
        with self.lock:
            seed = pick_surface(self.cloud, event.ray_origin, event.ray_direction)
            if seed is None:
                self.info.content = "Nothing there."
                return
            if self.mode.value == "Grow":
                self.selection.grow(seed, self._params())
            else:
                self.selection.paint(seed, self.brush_cm.value, erase=self.mode.value == "Brush erase")
            self._refresh()

    def _regrow(self) -> None:
        with self.lock:
            if self.selection.regrow(self._params()):
                self._refresh()

    def _edit(self, fn) -> None:
        with self.lock:
            fn()
            self._refresh()

    def _refresh(self) -> None:
        indices = self.selection.indices()
        only = self.only_leaf.value
        self.v.scene.add_point_cloud(
            "/selection", self.cloud.xyz[indices], colors=SELECTION_RGB, point_size=0.002,
            point_shape="circle", visible=not only)
        if only:
            self._add_splat("/leaf", indices)
        elif self.leaf_node is not None:
            self.leaf_node.remove()
        self.leaf_node = self.v.scene.get_handle_by_name("/leaf") if only else None
        scene = self.v.scene.get_handle_by_name("/scene")
        if scene is not None:
            scene.visible = not only
        if len(indices) == 0:
            self.info.content = "Nothing selected."
            return
        s = leaf_summary(self.cloud, indices)
        w, h, t = s["size_cm"]
        self.info.content = f"**{s['count']}** gaussians, {w:.1f} × {h:.1f} × {t:.1f} cm"
        if s["count"] < TOO_FEW:
            self.info.content += ". Very little: if you clicked a highlight, undo and click a greener spot."

    def _save(self) -> None:
        with self.lock:
            indices = self.selection.indices()
            name = self.name.value.strip()
            if len(indices) == 0 or not name or "/" in name:
                self.info.content = "Select something and give it a name first."
                return
            path = occluder_path(self.scene_name, name)
            if path.exists():
                self.info.content = f"`{name}` already exists: choose another name."
                return
            save_leaf(self.cloud, indices, path, self.scene_name)
            self.selection.clear()
            self._refresh()
            self.name.value = next_leaf_name(self.scene_name)
            self.info.content = f"Saved `{path}`"


def main(args: Args) -> None:
    app = LeafApp(args)
    threading.Event().wait()
    del app


if __name__ == "__main__":
    main(tyro.cli(Args))
