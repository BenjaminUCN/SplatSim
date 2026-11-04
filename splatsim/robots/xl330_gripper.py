"""Gripper hand_held_XL330 real: un Dynamixel XL330 detrás de una OpenRB-150 por USB.

La OpenRB-150 corre el ejemplo usb_to_dynamixel de ROBOTIS (paso directo USB <-> bus), así que aquí se
le habla al servo con el DynamixelSDK, igual que a un U2D2. Reemplaza al gripper del xArm, que iba
por el controlador del brazo (XArmRobot acepta este objeto como external_gripper).

  - Convención igual a la del xArm: 0 = abierto, 1 = cerrado.
  - Un hilo propio habla con el servo: command() y get_state() no tocan el USB, así un servo lento o
    desconectado no frena el lazo de 30 Hz del brazo.
  - Conectar no mueve el servo: sostiene donde esté hasta el primer command().
  - Modo 5 (posición con límite de corriente). Si queda empujando sin avanzar stall_seconds (p.ej.
    un tallo que no corta), deja de empujar hasta recibir un comando distinto.
  - Error del servo o USB perdido: ignora comandos hasta recover(). Es una decisión del operador,
    como Enable arm en el brazo, para que la hoja no se mueva sola al volver.
"""
import glob
import math
import threading
import time

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_GOAL_CURRENT = 102
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_BLOCK = 126    # corriente (2 bytes), velocidad (4), posición (4)
LEN_PRESENT_BLOCK = 10

MODE_CURRENT_BASED_POSITION = 5
XL330_MODELS = (1190, 1200)  # M077, M288
MAX_CURRENT_MA = 1750
TICKS_PER_TURN = 4096
ALERT_BIT = 0x80
HARDWARE_ERRORS = {0: "input voltage", 2: "overheating", 3: "encoder", 4: "electrical shock", 5: "overload"}

DEADBAND = 0.02             # cambio mínimo de comando que se envía (como en el gripper del xArm)
STALL_RESUME = 0.10         # cambio mínimo para volver a empujar tras un atasco
POSITION_TOLERANCE = 20     # pulsos: llegó al objetivo (pos_threshold de RoboHarvest)
STALL_VELOCITY = 2          # unidades de 0.229 rpm: bajo esto no avanza
MAX_COMM_FAILURES = 10      # lecturas seguidas sin respuesta antes de declararlo perdido
OPENRB_VID_PID = (0x2F5D, 0x2202)


class GripperError(RuntimeError):
    pass


def find_openrb_port() -> str:
    """El puerto de la OpenRB-150. Por identidad y no /dev/ttyACM0, que depende del orden de
    conexión (el GELLO también usa ese nombre)."""
    found = glob.glob("/dev/serial/by-id/*OpenRB*")
    if found:
        return sorted(found)[0]
    from serial.tools import list_ports
    for p in list_ports.comports():
        if (p.vid, p.pid) == OPENRB_VID_PID:
            return p.device
    raise GripperError("OpenRB-150 not found on USB (under WSL: usbipd attach); set gripper.port "
                       "in the config if it has another name")


def _signed(value: int, bytes_: int) -> int:
    limit = 1 << (8 * bytes_)
    return value - limit if value >= limit // 2 else value


class XL330Gripper:
    def __init__(self, port: str = "auto", dxl_id: int = 1, baudrate: int = 57600,
                 open_position: int | None = None, closed_position: int | None = None,
                 current_limit_ma: int = 1200, profile_velocity: int = 40,
                 profile_acceleration: int = 30, stall_seconds: float = 2.0, hz: float = 30.0,
                 handlers=None):
        """open/closed_position: pulsos del calibrado (scripts/xarm5/xl330_gripper.py calibrate).
        profile_velocity: unidades de 0.229 rpm (40 = 9 rpm). handlers: (puerto, paquetes) del SDK,
        para las pruebas sin hardware."""
        self.port = port
        self.dxl_id = dxl_id
        self.baudrate = baudrate
        self.open_position = open_position
        self.closed_position = closed_position
        self.current_limit_ma = min(int(current_limit_ma), MAX_CURRENT_MA)
        self.profile_velocity = profile_velocity
        self.profile_acceleration = profile_acceleration
        self.stall_seconds = stall_seconds
        self.period = 1 / hz
        self._handlers = handlers
        self._port_handler = None
        self._packet = None
        self._bus_lock = threading.RLock()  # el hilo y recover()/torque_off() comparten el bus
        self._thread = None
        self._running = False
        self._problem = None if self.calibrated else "not calibrated"
        self._ready = False                 # torque encendido y aceptando comandos
        self._target = None                 # último command() aún válido
        self._sent = None                   # último comando escrito al servo
        self._goal = None                   # pulsos, sin la vuelta de _turn
        self._turn = 0                      # múltiplo de 4096 entre el servo y el calibrado
        self._hold_request = False
        self._stall_since = None
        self._failures = 0
        self.stalled = False
        self.position = None                # pulsos, tal como los reporta el servo
        self.current_ma = 0
        self._state = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "XL330Gripper":
        keys = ("port", "dxl_id", "baudrate", "open_position", "closed_position", "current_limit_ma",
                "profile_velocity", "profile_acceleration", "stall_seconds")
        return cls(**{k: cfg[k] for k in keys if cfg.get(k) is not None})

    # ── interfaz para el robot y el panel (no tocan el USB) ──────────────────
    @property
    def calibrated(self) -> bool:
        if self.open_position is None or self.closed_position is None:
            return False
        return 50 <= abs(self.closed_position - self.open_position) <= TICKS_PER_TURN - 500

    @property
    def problem(self) -> str | None:
        return self._problem

    def command(self, value: float) -> None:
        """0 = abierto, 1 = cerrado; lo envía el hilo."""
        value = float(value)
        if math.isfinite(value):
            self._target = min(max(value, 0.0), 1.0)

    def hold(self) -> None:
        """Se queda donde está y olvida el comando pendiente (brazo detenido)."""
        self._target = None
        self._hold_request = True

    def get_state(self) -> float:
        return self._state

    def status_text(self) -> str:
        if self._problem:
            return f"**⚠ Gripper stopped** ({self._problem})"
        text = f"Gripper {self._state:.2f} (0 open, 1 closed), {abs(self.current_ma)} mA"
        return text + " — **stalled, not pushing**" if self.stalled else text

    # ── ciclo de vida ────────────────────────────────────────────────────────
    def start(self) -> None:
        """Conecta y lanza el hilo. Si falla, el brazo sigue funcionando y problem dice por qué."""
        self.recover()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._bus_lock:
            if self._port_handler is not None:
                try:
                    self._write(ADDR_TORQUE_ENABLE, 1, 0)
                except GripperError:
                    pass
                self._port_handler.closePort()
                self._port_handler = None
            self._ready = False

    def recover(self) -> str | None:
        """Reconecta, reinicia el servo si tiene un error y queda sostenido donde está. Devuelve el
        problema si no se pudo. La orden anterior se descarta."""
        with self._bus_lock:
            self._ready = False
            self._target = None
            try:
                self.connect()
                error = self._read(ADDR_HARDWARE_ERROR, 1)
                if error:
                    self._reboot()
                self._configure()
            except GripperError as e:
                self._problem = str(e)
            else:
                self._problem = None
                self._ready = True
            print(f"[XL330] {self._problem or f'ready, holding at {self._state:.2f}'}")
            return self._problem

    def torque_off(self) -> None:
        """Suelta el servo (para moverlo a mano); recover() lo vuelve a habilitar."""
        with self._bus_lock:
            self._ready = False
            self._target = None
            self._problem = "torque off"
            if self._port_handler is not None:
                try:
                    self._write(ADDR_TORQUE_ENABLE, 1, 0)
                except GripperError as e:
                    self._problem = str(e)

    # ── bus ──────────────────────────────────────────────────────────────────
    def connect(self) -> None:
        """Abre el puerto y comprueba que responde el servo; no cambia nada en él."""
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None
        if self._handlers is not None:
            port_handler, self._packet = self._handlers
        else:
            import dynamixel_sdk as sdk
            device = find_openrb_port() if self.port == "auto" else self.port
            port_handler, self._packet = sdk.PortHandler(device), sdk.PacketHandler(2.0)
        try:
            opened = port_handler.openPort() and port_handler.setBaudRate(self.baudrate)
        except Exception as e:  # noqa: BLE001  (pyserial lanza SerialException al no poder abrir)
            raise GripperError(f"cannot open the OpenRB-150 port: {e}") from e
        if not opened:
            raise GripperError("cannot open the OpenRB-150 port")
        self._port_handler = port_handler
        model, comm, _err = self._packet.ping(port_handler, self.dxl_id)
        if comm != 0:
            raise GripperError(f"no XL330 with ID {self.dxl_id} at {self.baudrate} baud: is "
                               "usb_to_dynamixel uploaded, and the servo powered?")
        if model not in XL330_MODELS:
            print(f"[XL330] warning: model number {model} is not an XL330")

    def _check(self, comm: int, err: int, what: str) -> None:
        if comm != 0:
            raise GripperError(f"no response ({what})")
        if err & ~ALERT_BIT:
            raise GripperError(f"servo rejected {what} (status {err:#x})")

    def _read(self, addr: int, size: int) -> int:
        reader = {1: self._packet.read1ByteTxRx, 2: self._packet.read2ByteTxRx,
                  4: self._packet.read4ByteTxRx}[size]
        value, comm, err = reader(self._port_handler, self.dxl_id, addr)
        self._check(comm, err, f"read {addr}")
        return _signed(value, size)

    def _write(self, addr: int, size: int, value: int) -> None:
        writer = {1: self._packet.write1ByteTxRx, 2: self._packet.write2ByteTxRx,
                  4: self._packet.write4ByteTxRx}[size]
        comm, err = writer(self._port_handler, self.dxl_id, addr, int(value) & ((1 << (8 * size)) - 1))
        self._check(comm, err, f"write {addr}")

    def _reboot(self) -> None:
        comm, err = self._packet.reboot(self._port_handler, self.dxl_id)
        self._check(comm, err, "reboot")
        time.sleep(0.5)

    def read_raw_position(self) -> int:
        """Posición actual; para el calibrado, con o sin torque."""
        with self._bus_lock:
            return self._read(132, 4)

    def _configure(self) -> None:
        if not self.calibrated:
            raise GripperError("not calibrated: run scripts/xarm5/xl330_gripper.py calibrate")
        self._write(ADDR_TORQUE_ENABLE, 1, 0)
        if self._read(ADDR_OPERATING_MODE, 1) != MODE_CURRENT_BASED_POSITION:  # EEPROM: solo si cambia
            self._write(ADDR_OPERATING_MODE, 1, MODE_CURRENT_BASED_POSITION)
        # cambiar de modo reinicia estos registros, así que se escriben siempre después
        self._write(ADDR_GOAL_CURRENT, 2, self.current_limit_ma)
        self._write(ADDR_PROFILE_ACCELERATION, 4, self.profile_acceleration)
        self._write(ADDR_PROFILE_VELOCITY, 4, self.profile_velocity)
        self._sync_to_present()
        self._write(ADDR_TORQUE_ENABLE, 1, 1)
        self._sync_to_present()              # encender el torque puede reiniciar la cuenta de vueltas
        self._sent = self._target = None
        self._hold_request = False
        self._stall_since = None
        self._failures = 0
        self.stalled = False

    def _sync_to_present(self) -> None:
        """Objetivo = posición actual, para no saltar. Tras reiniciarse el servo reporta la posición
        dentro de una vuelta, así que el calibrado se ubica en la vuelta más cercana."""
        present = self.read_raw_position()
        center = (self.open_position + self.closed_position) / 2
        self._turn = round((present - center) / TICKS_PER_TURN) * TICKS_PER_TURN
        self._write(ADDR_GOAL_POSITION, 4, present)
        self._goal = present - self._turn
        self.position = present
        self._state = self._fraction(present)

    def _fraction(self, present: int) -> float:
        span = self.closed_position - self.open_position
        return min(max((present - self._turn - self.open_position) / span, 0.0), 1.0)

    # ── hilo ─────────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while self._running:
            start = time.monotonic()
            with self._bus_lock:
                if self._ready:
                    self._tick(start)
            time.sleep(max(0.0, self.period - (time.monotonic() - start)))

    def _tick(self, now: float) -> None:
        """Una vuelta: lee, detecta problemas y atascos, y escribe el objetivo si cambió."""
        try:
            data, comm, err = self._packet.readTxRx(self._port_handler, self.dxl_id,
                                                    ADDR_PRESENT_BLOCK, LEN_PRESENT_BLOCK)
            if comm != 0:
                self._failures += 1
                if self._failures >= MAX_COMM_FAILURES:
                    self._stop_with("no response from the servo: check USB and power, then Recover")
                return
            self._failures = 0
            if err & ALERT_BIT:
                self._stop_with(self._describe_hardware_error())
                return
            self.current_ma = _signed(data[0] | data[1] << 8, 2)
            velocity = _signed(int.from_bytes(bytes(data[2:6]), "little"), 4)
            present = _signed(int.from_bytes(bytes(data[6:10]), "little"), 4)
            self.position = present
            self._state = self._fraction(present)
            self._send_target(present)
            self._watch_stall(now, present, velocity)
        except GripperError as e:
            self._stop_with(str(e))

    def _send_target(self, present: int) -> None:
        if self._hold_request:
            self._hold_request = False
            self._write_goal(present - self._turn)
            self._sent = None
            self.stalled = False
            return
        target = self._target
        if target is None:
            return
        change = math.inf if self._sent is None else abs(target - self._sent)
        needed = STALL_RESUME if self.stalled else DEADBAND
        # los extremos siempre se alcanzan, aunque el último paso sea menor que la banda muerta
        reaches_end = not self.stalled and target in (0.0, 1.0) and target != self._sent
        if change > needed or reaches_end:
            span = self.closed_position - self.open_position
            self._write_goal(round(self.open_position + target * span))
            self._sent = target
            self.stalled = False
            self._stall_since = None

    def _write_goal(self, ticks: int) -> None:
        self._write(ADDR_GOAL_POSITION, 4, ticks + self._turn)
        self._goal = ticks

    def _watch_stall(self, now: float, present: int, velocity: int) -> None:
        if self.stalled:
            return
        pushing = abs(self._goal + self._turn - present) > POSITION_TOLERANCE
        if not pushing or abs(velocity) > STALL_VELOCITY:
            self._stall_since = None
        elif self._stall_since is None:
            self._stall_since = now
        elif now - self._stall_since >= self.stall_seconds:
            self._write_goal(present - self._turn)
            self.stalled = True
            self._stall_since = None
            print(f"[XL330] stalled at {self._state:.2f}: not pushing until a new command")

    def _describe_hardware_error(self) -> str:
        try:
            bits = self._read(ADDR_HARDWARE_ERROR, 1)
        except GripperError:
            return "hardware error: then Recover"
        names = [name for bit, name in HARDWARE_ERRORS.items() if bits >> bit & 1]
        return f"hardware error: {', '.join(names) or hex(bits)}. Check the mechanism, then Recover"

    def _stop_with(self, problem: str) -> None:
        self._ready = False
        self._target = None
        self._problem = problem
        print(f"[XL330] {problem}")
