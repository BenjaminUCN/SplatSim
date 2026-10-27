"""Constantes compartidas del servidor xArm5. Todo lo "ajustable" vive aquí."""
from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase

# ── Simulación / control ─────────────────────────────────────────────────────
SIM_DT = 1 / 240          # paso de PyBullet (lo fija la base con setTimeStep)
CONTROL_HZ = 30           # frecuencia de comandos por defecto; launch_nodes --hz la sobrescribe
LOOP_HZ = 240             # frecuencia del serve_loop (UI y física en reposo)
SIM_CLOCKS = ("fixed", "realtime")  # tiempo simulado por comando: 1/hz fijo o el tiempo real
MAX_REALTIME_STEP = 0.25  # s: tope del avance por comando con sim_clock=realtime
IDLE_STEP_AFTER = 0.2     # s reales sin comandos tras los que el serve_loop vuelve a avanzar la física
LOW_LEVEL_HZ = 30         # lazo de control articular (igual que el hilo del driver real del xArm5)

# ── Grabación ────────────────────────────────────────────────────────────────
MAX_RECORD_SECONDS = 2    # duración máxima de una trayectoria, en tiempo simulado
RECORD_EVERY = 1          # guarda un frame cada N comandos

# ── Cámaras (intrínsecos) ────────────────────────────────────────────────────
FOV_16_9 = (1.2112585, 0.7417649)            # 69.4° x 42.5° (ojo: 16:9, el render es 4:3)
FOV_4_3 = (1.375955594372348, 1.1025297299614814)
BASE_CAM_FOV = FOV_16_9
WRIST_CAM_FOV = FOV_4_3
APPLE_CAM_FOV = FOV_16_9
IMAGE_W = 640
IMAGE_H = 480

# Las cámaras base (poses) y los arreglos de cámaras viven en configs/camera_poses.yaml
# (ver splatsim/utils/camera_setup.py)
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
# grilla de posiciones para scripts/run_policy_trials.py (se llena con el botón de guardar)
APPLE_GRID_PATH = "configs/apple_grid.yaml"

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
