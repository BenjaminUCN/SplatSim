"""Constantes compartidas del servidor xArm5. Todo lo "ajustable" vive aquí."""
from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase

# ── Loop / grabación ─────────────────────────────────────────────────────────
LOOP_HZ = 30              # frecuencia real del serve_loop (sleep = 1/LOOP_HZ)
MAX_RECORD_SECONDS = 2    # duración máxima de una trayectoria
RECORD_EVERY = 1          # guarda un frame cada N ticks

# ── Cámaras (intrínsecos) ────────────────────────────────────────────────────
FOV_X = 1.2112585         # 69.4°
FOV_Y = 0.7417649         # 42.5°  (ojo: es de 16:9, el render es 4:3)
IMAGE_W = 640
IMAGE_H = 480

# Cámara base (xarm5_lite_plant_v2)
BASE_CAM_XYZ = (-0.079, 0.0, 0.142)
BASE_CAM_RPY = (-1.562, 0.0, -4.708)
BASE_CAM_RANGE_XYZ = 1.5  # ± metros alrededor del valor inicial
BASE_CAM_RANGE_RPY = 3.15 # ± radianes

# Cámara de la manzana (offset respecto al centro de la manzana)
APPLE_CAM_OFFSET_XYZ = (0.0, 0.0, 0.0)
APPLE_CAM_OFFSET_RPY = (-1.562, 0.0, -1.562)

# ── Escena ───────────────────────────────────────────────────────────────────
FLOOR_Z = -0.7
APPLE_NAME = "plastic_apple"
APPLE_RANGE = {"x": (-0.9, -0.3), "y": (-0.8, 0.8), "z": (-0.5, 0.6)}
# Valor inicial de los sliders. Antes x=0.3 (fuera de rango); ahora dentro.
APPLE_INIT = (-0.3, 0.2, 0.1)

ENV_CONFIG = {
    "objects": [
        {
            "object_name": APPLE_NAME,
            "splat_object_name": APPLE_NAME,
            "grasp_config": [PybulletRobotServerBase.GRASP_CONFIGS["apple"]],
        }
    ]
}

# ── Gripper ──────────────────────────────────────────────────────────────────
GRIPPER_RANGE = [0, 0.085]       # apertura en metros
GRIPPER_MAX_ANGLE = 0.85         # rango del drive_joint en rad
SPLAT_GRIPPER_POSE = 0.5         # ángulo con el que se capturó el splat
NUM_DOFS = 6                     # 5 joints + gripper
