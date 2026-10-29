import dataclasses
import threading
import time
from typing import Dict, Optional

import numpy as np
from pyquaternion import Quaternion

from splatsim.robots.joint_controller import ControllerGains, JointController
from splatsim.robots.robot import Robot


def _aa_from_quat(quat: np.ndarray) -> np.ndarray:
    assert quat.shape == (4,), "Input quaternion must be a 4D vector."
    norm = np.linalg.norm(quat)
    assert norm != 0, "Input quaternion must not be a zero vector."
    quat = quat / norm

    Q = Quaternion(w=quat[3], x=quat[0], y=quat[1], z=quat[2])
    angle = Q.angle
    axis = Q.axis
    aa = axis * angle
    return aa


def _quat_from_aa(aa: np.ndarray) -> np.ndarray:
    assert aa.shape == (3,), "Input axis-angle must be a 3D vector."
    norm = np.linalg.norm(aa)
    assert norm != 0, "Input axis-angle must not be a zero vector."
    axis = aa / norm

    Q = Quaternion(axis=axis, angle=norm)
    quat = np.array([Q.x, Q.y, Q.z, Q.w])
    return quat


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

    def cartesian_pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def quat(self) -> np.ndarray:
        return _quat_from_aa(self.aa)

    def joints(self) -> np.ndarray:
        return np.array([self.j1, self.j2, self.j3, self.j4, self.j5])

    def gripper_pos(self) -> float:
        return self.gripper


class Rate:
    def __init__(self, *, duration):
        self.duration = duration
        self.last = time.time()

    def sleep(self, duration=None) -> None:
        duration = self.duration if duration is None else duration
        assert duration >= 0
        now = time.time()
        passed = now - self.last
        if passed < 0 :
            self.last = now
            return
        remaining = duration - passed
        assert passed >= 0
        if remaining > 0.0001:
            time.sleep(remaining)
        self.last = time.time()


class XArmRobot(Robot):
    GRIPPER_OPEN = 800
    GRIPPER_CLOSE = 0

    DEFAULT_MAX_DELTA = 0.05
    # modo velocidad: si el hilo de control se atrasa, el xArm detiene el brazo tras este tiempo
    VELOCITY_WATCHDOG_PERIODS = 3
    # estados del xArm en que acepta comandos: 1 en movimiento, 2 quieto (3 suspendido, 4 detenido,
    # p.ej. tras la parada de emergencia, una colisión o un stop desde la app)
    READY_STATES = (1, 2)

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def num_dofs(self) -> int:
        # 5 joints + 1 gripper
        return 6

    def set_object_pose(
        self,
        object_name: str,
        position,
        orientation,
        use_gravity: bool = True,
    ) -> None:
        """No-op on real hardware — physical objects can't be teleported."""
        pass

    def get_joint_state(self) -> np.ndarray:
        state = self.get_state()
        gripper = state.gripper_pos()
        all_dofs = np.concatenate([state.joints(), np.array([gripper])])
        return all_dofs

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        if len(joint_state) == 5:
            self.set_command(joint_state, None)
        elif len(joint_state) == 6:
            self.set_command(joint_state[:5], joint_state[5])
        else:
            raise ValueError(
                f"Invalid joint state: {joint_state}, len={len(joint_state)}"
            )

    def stop(self):
        self.running = False
        if self.command_thread is not None:
            self.command_thread.join()
            self.command_thread = None
        if self.robot is not None:
            if self.ctrl.mode == "velocity":
                self.robot.vc_set_joint_velocity([0.0] * 5, is_radian=True)
            self.robot.disconnect()
        if self.command_thread is not None:
            self.command_thread.join()

    # ------------------------------------------------------------------ #
    # Init                                                                 #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        ip: str = "192.168.123.52",  # IP del xArm5 en la red IoT_v3
        real: bool = True,
        control_frequency: float = 30.0,
        max_delta: float = DEFAULT_MAX_DELTA,
        control_mode: str = "position",
        command_hz: float = 30.0,
    ):
        """control_mode: position / interp / velocity (splatsim/robots/joint_controller.py),
        la misma ley que usa el simulador. command_hz: frecuencia con que llegan los objetivos
        (el --hz del cliente: run_env_sim con GELLO o con la política)."""
        print(f"[XArmRobot] Connecting to xArm5 at {ip} (control_mode={control_mode})")
        self.real = real
        self.max_delta = max_delta
        self.ctrl = JointController(control_mode, 1 / command_hz, ControllerGains(max_delta=max_delta))

        # Gripper cache: se actualiza cada GRIPPER_CACHE_STEPS pasos
        # para no saturar el bus de comunicación (igual que en Gello)
        self._gripper_counter = 0
        self._cached_gripper_pos = 0.0
        self._last_sent_gripper = -1.0   # evita envíos redundantes al gripper
        self._gripper_read_failed = False

        if real:
            from xarm.wrapper import XArmAPI
            self.robot = XArmAPI(ip, is_radian=True)
        else:
            self.robot = None

        self._control_frequency = control_frequency
        # por qué el brazo no acepta comandos (None: listo); mientras no sea None el hilo de control
        # no envía nada. Solo enable_arm() lo vuelve a habilitar: tras una parada de emergencia o
        # una colisión, el operador decide cuándo es seguro
        self.arm_problem = None
        if real and not self._report_connected():
            print(f"[XArmRobot] No report stream from {ip} (ports 30001-30003 unreachable): "
                  "the arm state is queried directly")
        self._clear_error_states()
        self._set_gripper_position(self.GRIPPER_OPEN)

        self.last_state_lock = threading.Lock()
        self.target_command_lock = threading.Lock()
        self.last_state = self._update_last_state()
        self.target_command = {
            "joints": self.last_state.joints(),
            "gripper": 0,
        }
        self.ctrl.reset(self.last_state.joints(), time.monotonic())

        self.running = True
        self.command_thread = None
        if real:
            self.command_thread = threading.Thread(
                target=self._robot_thread, daemon=True
            )
            self.command_thread.start()

    # ------------------------------------------------------------------ #
    # State / command                                                      #
    # ------------------------------------------------------------------ #

    def get_state(self) -> RobotState:
        with self.last_state_lock:
            return self.last_state

    def set_control_mode(self, mode: str) -> None:
        """Cambia la ley de control (y el modo del xArm: 1 posición, 4 velocidad); el brazo
        queda sostenido en la posición actual hasta el próximo objetivo."""
        with self.target_command_lock:
            old = self.ctrl.mode
            self.ctrl = JointController(mode, self.ctrl.nominal_period, self.ctrl.g)
            joints = self.get_state().joints()
            self.ctrl.reset(joints, time.monotonic())
            self.target_command = {"joints": joints, "gripper": self.target_command["gripper"]}
            if self.robot is not None and (old == "velocity") != (mode == "velocity"):
                if old == "velocity":
                    self.robot.vc_set_joint_velocity([0.0] * 5, is_radian=True)
                self.robot.set_mode(4 if mode == "velocity" else 1)
                self.robot.set_state(state=0)
                # con el lock tomado el hilo de control espera: que no vea el modo viejo del
                # reporte y lo tome por un brazo detenido
                deadline = time.monotonic() + 1.0
                while (self._report_connected() and time.monotonic() < deadline
                       and self.robot.mode != (4 if mode == "velocity" else 1)):
                    time.sleep(0.02)
        print(f"[XArmRobot] control_mode={mode}")

    def set_command(self, joints: np.ndarray, gripper: Optional[float] = None) -> None:
        with self.target_command_lock:
            self.target_command = {
                "joints": joints,
                "gripper": gripper,
            }
            self.ctrl.set_target(joints, time.monotonic())

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _clear_error_states(self):
        if self.robot is None:
            return
        codes = {"clean_error": self.robot.clean_error(),
                 "clean_warn": self.robot.clean_warn(),
                 "motion_enable": self.robot.motion_enable(True)}
        time.sleep(1)
        # modo 1: servo por posición (set_servo_angle_j); modo 4: velocidad articular
        codes["set_mode"] = self.robot.set_mode(4 if self.ctrl.mode == "velocity" else 1)
        time.sleep(1)
        codes["set_collision_sensitivity"] = self.robot.set_collision_sensitivity(0)
        time.sleep(1)
        codes["set_state"] = self.robot.set_state(state=0)
        time.sleep(1)
        if any(codes.values()):
            print(f"[XArmRobot] Return codes while enabling: {codes}")
        self.robot.set_gripper_enable(True)
        time.sleep(1)
        self.robot.set_gripper_mode(0)
        time.sleep(1)
        self.robot.set_gripper_speed(3000)
        time.sleep(1)

    def _report_connected(self) -> bool:
        """Si llega el reporte del xArm (puertos 30001-30003): de él salen robot.state, mode y
        error_code. Hay redes que solo dejan pasar el puerto de comandos (502); sin reporte esos
        valores quedan en los iniciales (state 4, mode 0) y no dicen nada."""
        stream = getattr(self.robot._arm, "_stream_report", None)
        return bool(stream is not None and stream.connected)

    def _check_arm(self, query: bool = False) -> Optional[str]:
        """Por qué el brazo no acepta comandos, o None si está listo. Con el reporte se usa el
        último; sin él, con query=True se le pregunta al brazo el estado y el error (el modo no se
        puede preguntar: vale el código de set_mode) y con query=False no se sabe (None)."""
        if self.robot is None:
            return None
        if self._report_connected():
            error, state, mode = self.robot.error_code, self.robot.state, self.robot.mode
        elif query:
            code, err_warn = self.robot.get_err_warn_code()
            error = err_warn[0] if code == 0 else 0
            code, state = self.robot.get_state()
            mode = None
        else:
            return None
        expected = 4 if self.ctrl.mode == "velocity" else 1
        if error:
            return f"controller error {error}"
        if state not in self.READY_STATES:
            return f"state {state} (3 = suspended, 4 = stopped, e.g. emergency stop)"
        if mode is not None and mode != expected:
            return f"mode {mode}, expected {expected}"
        return None

    def enable_arm(self) -> Optional[str]:
        """Limpia errores, habilita los motores y vuelve al modo de control (como al arrancar).
        El brazo queda sostenido donde está: el objetivo pasa a ser la posición actual. Devuelve
        el problema si sigue sin estar listo (p.ej. la parada de emergencia sigue apretada)."""
        self.arm_problem = "enabling"     # el hilo de control no envía nada mientras tanto
        print("[XArmRobot] Enabling the arm...")
        self._clear_error_states()
        with self.target_command_lock:
            joints = self._update_last_state().joints()
            self.ctrl.reset(joints, time.monotonic())
            self.target_command = {"joints": joints, "gripper": self.target_command["gripper"]}
        # tras set_state(0) el controlador reporta un rato el estado 0 (transición) antes de 1/2
        deadline = time.monotonic() + 3.0
        expected = 4 if self.ctrl.mode == "velocity" else 1
        while (problem := self._check_arm(query=True)) is not None and time.monotonic() < deadline:
            if problem.startswith("mode"):
                # el controlador puede ignorar el cambio de modo si llegó durante la transición
                print(f"[XArmRobot] {problem}: set_mode -> {self.robot.set_mode(expected)}, "
                      f"set_state -> {self.robot.set_state(state=0)}")
            time.sleep(0.5)
        self.arm_problem = problem
        print(f"[XArmRobot] Arm {'ready' if problem is None else f'still not ready: {problem}'}")
        return problem

    def _get_gripper_pos(self) -> float:
        if self.robot is None:
            return 0.0
        code, gripper_pos = self.robot.get_gripper_position()
        if code != 0 or gripper_pos is None:
            # con el brazo detenido (parada de emergencia) el gripper no responde por modbus
            # (code 22): se sigue con el último valor. Antes se reintentaba en un lazo rehabilitando
            # el brazo, que trababa el hilo de control; ahora lo rehabilita Enable arm
            if not self._gripper_read_failed:
                print(f"[XArmRobot] get_gripper_position() failed (code {code}): keeping the last "
                      "gripper position until it responds")
            self._gripper_read_failed = True
            return self._cached_gripper_pos
        self._gripper_read_failed = False

        normalized = (gripper_pos - self.GRIPPER_OPEN) / (
            self.GRIPPER_CLOSE - self.GRIPPER_OPEN
        )
        return float(normalized)

    def _set_gripper_position(self, pos: int) -> None:
        if self.robot is None:
            return
        self.robot.set_gripper_position(pos, wait=False)

    # ------------------------------------------------------------------ #
    # Control thread                                                       #
    # ------------------------------------------------------------------ #

    def _robot_thread(self):
        rate = Rate(duration=1 / self._control_frequency)
        step_times = []
        count = 0

        while self.running:
            s_t = time.time()

            self.last_state = self._update_last_state()

            with self.target_command_lock:
                gripper_command = self.target_command["gripper"]
                kind, values = self.ctrl.step(
                    self.last_state.joints(), time.monotonic(), 1 / self._control_frequency)

            if self.arm_problem is None and (problem := self._check_arm()) is not None:
                self._pause_arm(problem)
            if self.arm_problem is None:
                if kind == "position":
                    ret = self._set_position(values)
                else:
                    ret = self._set_velocity(values)
                # sin reporte, la parada se nota en el código del comando (1 error, 9 no listo)
                if ret in (1, 9) and not self._report_connected():
                    self._pause_arm(self._check_arm(query=True) or f"command returned code {ret}")

            if gripper_command is not None:
                # Solo envía si cambió más del 2 % (evita spam al gripper)
                if abs(gripper_command - self._last_sent_gripper) > 0.02:
                    target_pos = self.GRIPPER_OPEN + gripper_command * (
                        self.GRIPPER_CLOSE - self.GRIPPER_OPEN
                    )
                    self._set_gripper_position(int(target_pos))
                    self._last_sent_gripper = gripper_command

            self.last_state = self._update_last_state()

            rate.sleep()
            step_times.append(time.time() - s_t)
            count += 1
            if count % 1000 == 0:
                frequency = 1 / np.mean(step_times)
                print(
                    f"[XArmRobot] Low Level Frequency - "
                    f"mean: {frequency:10.3f}, "
                    f"std: {np.std(step_times):10.3f}, "
                    f"min: {np.min(step_times):10.3f}, "
                    f"max: {np.max(step_times):10.3f}"
                )
                step_times = []

    def _update_last_state(self) -> RobotState:
        with self.last_state_lock:
            if self.robot is None:
                return RobotState(0, 0, 0, 0, 0, 0, 0, 0, 0, np.zeros(3))

            # Actualiza el gripper cada 10 pasos (caché de Gello)
            self._gripper_counter += 1
            if self._gripper_counter >= 10:
                self._cached_gripper_pos = self._get_gripper_pos()
                self._gripper_counter = 0

            gripper_pos = self._cached_gripper_pos

            code, servo_angle = self.robot.get_servo_angle(is_radian=True)
            while code != 0:
                print(f"[XArmRobot] Error code {code} in get_servo_angle().")
                self._clear_error_states()
                code, servo_angle = self.robot.get_servo_angle(is_radian=True)

            code, cart_pos = self.robot.get_position_aa(is_radian=True)
            while code != 0:
                print(f"[XArmRobot] Error code {code} in get_position_aa().")
                self._clear_error_states()
                code, cart_pos = self.robot.get_position_aa(is_radian=True)

            cart_pos = np.array(cart_pos)
            aa = cart_pos[3:]
            cart_pos[:3] /= 1000  # mm -> m

            return RobotState.from_robot(
                cart_pos,
                servo_angle,
                gripper_pos,
                aa,
            )

    def _pause_arm(self, problem: str) -> None:
        self.arm_problem = problem
        print(f"[XArmRobot] Arm stopped: {problem}. Commands are paused until Enable arm "
              "(panel, Robot folder)")

    def _set_velocity(self, speeds: np.ndarray) -> int:
        if self.robot is None:
            return 0
        # is_sync=False: cada articulación sigue su propia velocidad (no se escalan juntas)
        ret = self.robot.vc_set_joint_velocity(
            [float(v) for v in speeds], is_radian=True, is_sync=False,
            duration=self.VELOCITY_WATCHDOG_PERIODS / self._control_frequency)
        # ret 1/9 (error, brazo no listo): pausa el brazo hasta Enable arm; antes se rehabilitaba
        # solo, también tras una parada de emergencia
        return ret

    def _set_position(self, joints: np.ndarray) -> int:
        if self.robot is None:
            return 0
        ret = self.robot.set_servo_angle_j(joints, wait=False, is_radian=True)
        # ret 1/9: ver _set_velocity
        return ret

    # ------------------------------------------------------------------ #
    # Observations (interfaz SplatSim)                                     #
    # ------------------------------------------------------------------ #

    def get_observations(self) -> Dict[str, np.ndarray]:
        state = self.get_state()
        pos_quat = np.concatenate([state.cartesian_pos(), state.quat()])
        joints = self.get_joint_state()

        # Estado cartesiano: [x, y, z, roll, pitch, yaw, gripper]
        # Igual al formato que usa la clase base del sim (sim_robot_pybullet_base.py)
        # El xArm devuelve axis-angle en cart_pos[3:], lo convertimos a euler
        aa = state.aa  # (3,) axis-angle
        aa_norm = np.linalg.norm(aa)
        if aa_norm > 1e-6:
            from pyquaternion import Quaternion
            axis = aa / aa_norm
            Q = Quaternion(axis=axis, angle=aa_norm)
            quat_xyzw = np.array([Q.x, Q.y, Q.z, Q.w])
        else:
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])

        # Convertir quaternion a euler usando pyquaternion
        import math
        x, y, z, w = quat_xyzw
        roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
        yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        euler = np.array([roll, pitch, yaw])

        cart_state = np.concatenate([
            state.cartesian_pos(),   # (3,) xyz en metros
            euler,                   # (3,) roll, pitch, yaw
            [state.gripper_pos()],   # (1,) gripper normalizado [0,1]
        ])  # → (7,)

        return {
            "joint_positions": joints,        # (6,) : 5 joints + gripper
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,           # (7,) : xyz + xyzw
            "gripper_position": np.array(state.gripper_pos()),
            "state": cart_state,              # (7,) : x, y, z, roll, pitch, yaw, gripper
            "control_mode": self.ctrl.mode,
        }


# ------------------------------------------------------------------ #
# Quick test                                                           #
# ------------------------------------------------------------------ #

def main():
    ip = "192.168.123.52"
    robot = XArmRobot(ip)
    time.sleep(1)
    print(robot.get_state())
    time.sleep(1)
    print(robot.get_state())
    print("end")
    robot.stop()


if __name__ == "__main__":
    main()