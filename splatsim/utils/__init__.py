"""Utilidades de splatsim.

En la raíz están las del SplatSim original (cameras, robot_splat_render_utils, sh_utils,
transform_utils, utils_fk), sin mover para no romper el código heredado del UR5, y las de uso
general del fork (paths, remote_call, viser_page). Lo demás del fork va por tema:

- data/      grabaciones y datasets: conversión a LeRobot, registro, selección de episodios, etiquetas
- render/    cámaras y render del xArm5: arreglos de cámaras, oclusión, render en paralelo
- scene/     objetos de la escena: tomate (grilla, pedúnculo), hojas, gripper hand_held
- geometry/  cinemática del robot: autocolisión
- trials/    evaluación de políticas: punto de agarre, duración de los ensayos
"""
