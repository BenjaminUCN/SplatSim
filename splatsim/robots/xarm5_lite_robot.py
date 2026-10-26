import dataclasses
import threading
import time
from typing import Dict, Optional

import numpy as np
from pyquaternion import Quaternion

from gello.robots.robot import Robot

"""
xarm_robot.py
Script que define una clase de robot para controlar un brazo xArm con gripper.
"""

"""
Función que convierte un cuaternión a una representación de ángulo de eje.
Args:
    quat (np.ndarray): El cuaternión a convertir.
Returns:
    np.ndarray: La representación de ángulo de eje del cuaternión.
"""
def _aa_from_quat(quat: np.ndarray) -> np.ndarray:
    # Asegura que el cuaternión tenga la forma correcta y no sea un vector cero
    assert quat.shape == (4,), "Input quaternion must be a 4D vector."
    norm = np.linalg.norm(quat)
    assert norm != 0, "Input quaternion must not be a zero vector."
    quat = quat / norm  # Normaliza el cuaternión

    Q = Quaternion(w=quat[3], x=quat[0], y=quat[1], z=quat[2])
    angle = Q.angle
    axis = Q.axis
    aa = axis * angle
    return aa

"""
Convierte una representación de ángulo de eje a un cuaternión.
Args:
    aa (np.ndarray): La representación de ángulo de eje a convertir.
Returns:
    np.ndarray: La representación de cuaternión del ángulo de eje.
"""
def _quat_from_aa(aa: np.ndarray) -> np.ndarray:
    # Asegura que la representación de ángulo de eje tenga la forma correcta y no sea un vector cero
    assert aa.shape == (3,), "Input axis-angle must be a 3D vector."
    norm = np.linalg.norm(aa)
    assert norm != 0, "Input axis-angle must not be a zero vector."
    axis = aa / norm  # Normaliza el vector de eje

    Q = Quaternion(axis=axis, angle=norm)
    quat = np.array([Q.x, Q.y, Q.z, Q.w])
    return quat


"""
Clase que representa un robot xArm con gripper, con control de posición en las articulaciones.
Implementa la interfaz de la clase base Robot, permitiendo obtener el estado del robot y enviar comandos de posición a las articulaciones.
"""
@dataclasses.dataclass(frozen=True)
class RobotState:
    x: float
    y: float
    z: float
    gripper: float
    j1: float
    j2: float
    j3: float
    j4: float
    j5: float
    aa: np.ndarray

    @staticmethod
    def from_robot(
        cartesian: np.ndarray,
        joints: np.ndarray,
        gripper: float,
        aa: np.ndarray,
    ) -> "RobotState":
        return RobotState(
            cartesian[0],
            cartesian[1],
            cartesian[2],
            gripper,
            joints[0],
            joints[1],
            joints[2],
            joints[3],
            joints[4],
            aa,
        )
    # Regresa la posición cartesiana del efector final como un vector numpy de 3 elementos (x, y, z).
    def cartesian_pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    # Regresa la orientación del efector final como un cuaternión (x, y, z, w) convertido desde la representación de ángulo de eje.
    def quat(self) -> np.ndarray:
        return _quat_from_aa(self.aa)
    # Regresa el estado de las articulaciones del robot como un vector numpy de 5 elementos (j1, j2, j3, j4, j5).
    def joints(self) -> np.ndarray:
        return np.array([self.j1, self.j2, self.j3, self.j4, self.j5])
    # Regresa la posición del gripper, que es un valor normalizado entre 0 (abierto) y 1 (cerrado).
    def gripper_pos(self) -> float:
        return self.gripper


"""
Clase de utilidad para controlar la frecuencia de actualización del robot, 
asegurando que los comandos se envíen a intervalos regulares.
"""
class Rate:
    # Inicializa el temporizador con la duración deseada entre comandos.
    def __init__(self, *, duration):
        self.duration = duration
        self.last = time.time()
        # Configuracion para gripper

    # Duerme el hilo actual hasta que haya pasado la duración especificada desde la última vez que se llamó a sleep().
    def sleep(self, duration=None) -> None:
        duration = self.duration if duration is None else duration
        assert duration >= 0
        now = time.time()
        passed = now - self.last
        remaining = duration - passed
        assert passed >= 0
        if remaining > 0.0001:
            time.sleep(remaining)
        self.last = time.time()


"""
Clase que representa el robot xArm con gripper, implementando la interfaz definida por la clase base Robot. 
Esta clase maneja la conexión con el hardware del robot, el envío de comandos de posición a las articulaciones y la lectura del estado actual del robot.
"""
class XArm5LiteRobot(Robot):
    # Define cuando el gripper está abierto y cerrado
    GRIPPER_OPEN = 800
    GRIPPER_CLOSE = 0
    #  MAX_DELTA = 0.2
    # Define la cantidad de grados de libertad (DoFs) del robot, que en este caso es 6 con gripper.

    DEFAULT_MAX_DELTA = 0.05

    def num_dofs(self) -> int:
        return 6
    # Regresa el estado de las articulaciones del robot como un vector numpy de 6 elementos (j1, j2, j3, j4, j5, gripper).
    def get_joint_state(self) -> np.ndarray:
        state = self.get_state()
        gripper = state.gripper_pos()
        all_dofs = np.concatenate([state.joints(), np.array([gripper])])
        return all_dofs
    # Recibe y valida los comandos de articulaciones para aplicarlos en el robot real.
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        if len(joint_state) == 5:
            self.set_command(joint_state, None)
        elif len(joint_state) == 6:
            self.set_command(joint_state[:5], joint_state[5])
        else:
            raise ValueError(
                f"Invalid joint state: {joint_state}, len={len(joint_state)}"
            )
    # Detiene el robot y cierra la conexión con el hardware, asegurando que el hilo de control se termine correctamente.
    def stop(self):
        self.running = False
        if self.robot is not None:
            self.robot.disconnect()

        if self.command_thread is not None:
            self.command_thread.join()
    """
    Inicializa la conexión con el robot xArm, configura los parámetros de control y lanza un hilo dedicado para enviar comandos al robot a una frecuencia constante.
    - ip: La dirección IP del robot en la red.
    - control_frequency: La frecuencia a la que se enviarán los comandos al robot.
    - max_delta: El cambio máximo permitido en las articulaciones por paso para evitar movimientos bruscos.
    """
    def __init__(
        self,
        ip: str = "192.168.1.239", # IP del robot en la red IoT_v3
        control_frequency: float = 50.0,
        max_delta: float = DEFAULT_MAX_DELTA,
    ):
        self._gripper_counter = 0
        self._cached_gripper_pos = 0.0
        self._last_sent_gripper = -1.0
        print(ip)
        self.max_delta = max_delta
        # Importa la librería de control del xArm de Ufactory
        from xarm.wrapper import XArmAPI
        self.robot = XArmAPI(ip, is_radian=True)

        self._control_frequency = control_frequency
        self._clear_error_states()
        self._set_gripper_position(self.GRIPPER_OPEN)

        self.last_state_lock = threading.Lock()
        self.target_command_lock = threading.Lock()
        self.last_state = self._update_last_state()
        self.target_command = {
            "joints": self.last_state.joints(),
            "gripper": 0,
        }
        self.running = True
        self.command_thread = None
        self.command_thread = threading.Thread(target=self._robot_thread)
        self.command_thread.start()


    # Regresa el estado completo del robot
    def get_state(self) -> RobotState:
        with self.last_state_lock:
            return self.last_state
        
    # Establece el comando de posición para las articulaciones del robot.
    def set_command(self, joints: np.ndarray, gripper: Optional[float] = None) -> None:
        with self.target_command_lock:
            self.target_command = {
                "joints": joints,
                "gripper": gripper,
            }

    # Método interno para limpiar los estados de error del robot.
    def _clear_error_states(self):
        if self.robot is None:
            return
        self.robot.clean_error()
        self.robot.clean_warn()
        self.robot.motion_enable(True)
        time.sleep(1)
        self.robot.set_mode(1)
        time.sleep(1)
        self.robot.set_collision_sensitivity(0)
        time.sleep(1)
        self.robot.set_state(state=0)
        time.sleep(1)
        self.robot.set_gripper_enable(True)
        time.sleep(1)
        self.robot.set_gripper_mode(0)
        time.sleep(1)
        self.robot.set_gripper_speed(3000)
        time.sleep(1)

    def _get_gripper_pos(self) -> float:
        if self.robot is None:
            return 0.0
        code, gripper_pos = self.robot.get_gripper_position()
        while code != 0 or gripper_pos is None:
            print(f"Error code {code} in get_gripper_position(). {gripper_pos}")
            time.sleep(0.001)
            code, gripper_pos = self.robot.get_gripper_position()
            if code == 22:
                self._clear_error_states()

        normalized_gripper_pos = (gripper_pos - self.GRIPPER_OPEN) / (
            self.GRIPPER_CLOSE - self.GRIPPER_OPEN
        )
        return normalized_gripper_pos

    def _set_gripper_position(self, pos: int) -> None:
        if self.robot is None:
            return
        self.robot.set_gripper_position(pos, wait=False)
        # while self.robot.get_is_moving():
        #     time.sleep(0.01)

    # Hilo dedicado para enviar comandos de posición al robot a una frecuencia constante.
    def _robot_thread(self):
        rate = Rate(
            duration=1 / self._control_frequency
        )
        step_times = []
        count = 0
        # Actualiza el estado del robot.
        while self.running:
            s_t = time.time()
            self.last_state = self._update_last_state()
            with self.target_command_lock:
                joint_delta = np.array(
                    self.target_command["joints"] - self.last_state.joints()
                )
                gripper_command = self.target_command["gripper"]

            norm = np.linalg.norm(joint_delta)
            # Umbral para limitar el cambio en las articulaciones por paso, evitando movimientos bruscos o peligrosos.
            if norm > self.max_delta: # Debe de ser 0.01
                delta = joint_delta / norm * self.max_delta
            else:
                delta = joint_delta

            # command position
            self._set_position(
                self.last_state.joints() + delta,
            )

            if gripper_command is not None:
                # Solo envía el comando si la diferencia es mayor al 2%
                if abs(gripper_command - self._last_sent_gripper) > 0.02:
                    set_point = gripper_command
                    target_pos = self.GRIPPER_OPEN + set_point * (self.GRIPPER_CLOSE - self.GRIPPER_OPEN)
                    self._set_gripper_position(int(target_pos))
                    self._last_sent_gripper = gripper_command
            self.last_state = self._update_last_state()

            rate.sleep()
            step_times.append(time.time() - s_t)
            count += 1
            if count % 1000 == 0:
                # Calcula y muestra estadísticas de la frecuencia de control cada 1000 pasos, incluyendo la media, desviación estándar, mínimo y máximo de los tiempos de paso.
                frequency = 1 / np.mean(step_times)
                print(
                    f"Low  Level Frequency - mean: {frequency:10.3f}, std: {np.std(frequency):10.3f}, min: {np.min(frequency):10.3f}, max: {np.max(frequency):10.3f}"
                )
                step_times = []
    # Actualiza el estado del robot leyendo los sensores y la posición actual, manejando cualquier error que pueda ocurrir durante la lectura.
    def _update_last_state(self) -> RobotState:
        with self.last_state_lock:
            if self.robot is None:
                return RobotState(0, 0, 0, 0, 0, 0, 0, 0, 0, np.zeros(3))

            self._gripper_counter += 1
            if self._gripper_counter >= 10:
                self._cached_gripper_pos = self._get_gripper_pos()
                self._gripper_counter = 0
        
            gripper_pos = self._cached_gripper_pos

            code, servo_angle = self.robot.get_servo_angle(is_radian=True)
            while code != 0:
                print(f"Error code {code} in get_servo_angle().")
                self._clear_error_states()
                code, servo_angle = self.robot.get_servo_angle(is_radian=True)

            code, cart_pos = self.robot.get_position_aa(is_radian=True)
            while code != 0:
                print(f"Error code {code} in get_position().")
                self._clear_error_states()
                code, cart_pos = self.robot.get_position_aa(is_radian=True)

            cart_pos = np.array(cart_pos)
            aa = cart_pos[3:]
            cart_pos[:3] /= 1000

            return RobotState.from_robot(
                cart_pos,
                servo_angle,
                gripper_pos,
                aa,
            )
    # Establece el comando de posición para las articulaciones del robot, enviando los comandos al hardware.
    def _set_position(
        self,
        joints: np.ndarray,
    ) -> None:
        if self.robot is None:
            return
        # Umbral para el eje xyz que está en un mínimo y máximio
        ret = self.robot.set_servo_angle_j(joints, wait=False, is_radian=True)
        if ret in [1, 9]:
            self._clear_error_states()

    def get_observations(self) -> Dict[str, np.ndarray]:
        state = self.get_state()
        pos_quat = np.concatenate([state.cartesian_pos(), state.quat()])
        joints = self.get_joint_state()
        observations = {
            "joint_positions": joints,  # rotational joint + gripper state
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": np.array(state.gripper_pos()),
            
        }
        # print(f"[XarmRobot] state: {ob}")
        return observations 

# Función de prueba para crear una instancia del robot, obtener su estado y luego detenerlo.
def main():
    ip = "192.168.1.239"
    robot = XArm5LiteRobot(ip)
    import time

    time.sleep(1)
    print(robot.get_state())

    time.sleep(1)
    print(robot.get_state())
    print("end")
    robot.stop()


if __name__ == "__main__":
    main()
