import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
import types
import yaml

from splatsim.robots.robot import BimanualRobot, PrintRobot
from gello.zmq_core.robot_node import ZMQServerRobot


@dataclass
class Args:
    robot: str = "xarm"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.1.10"
    gaussian_path : str = "/home/jennyw2/data/output/robot_iphone/point_cloud/iteration_30000/point_cloud.ply"
    robot_name: str = "robot_iphone"
    move_apple: bool = False
    hz: int = 30  # frecuencia de control del cliente; debe coincidir con run_env_sim --hz
    # cámaras renderizadas en cada observación (xArm5); se graban todas.
    # arrangement (p.ej. 2cam, ver configs/camera_poses.yaml) reemplaza a cameras
    cameras: tuple[str, ...] = ("base_rgb", "base_left_rgb", "base_right_rgb", "wrist_rgb")
    arrangement: str | None = None
    # controles del xArm5: "viser" (panel web) o "pybullet" (sliders de PyBullet, solo sim);
    # en el real: "viser" o "none"
    gui: str = "viser"
    # xArm5 real (--robot xarm5_lite): IP, home, cámaras y carpeta de grabaciones; --robot_ip
    # explícito reemplaza a la IP del archivo
    real_config: str = "configs/xarm5_real.yaml"
    panel_port: int = 8080
    # xArm5 (sim y real): cómo se siguen los objetivos articulares: position / interp / velocity
    control_mode: str = "position"
    # sim xArm5: tiempo simulado por comando. fixed = 1/hz (políticas, ensayos; dt fijo por frame),
    # realtime = el tiempo real entre comandos (teleop con GELLO, aunque el lazo no alcance hz)
    sim_clock: str = "fixed"


def launch_robot_server(args: Args):
    with open("configs/object_configs/objects.yaml", "r") as f:
        object_config = yaml.safe_load(f)

    has_wrist_camera = object_config[args.robot_name].get("wrist_camera_link_name", None) is not None
    if has_wrist_camera:
        camera_names = ["base_rgb", "wrist_rgb"]
    else:
        camera_names = ["base_rgb"]
    use_gripper = object_config[args.robot_name].get("use_gripper", True)
    
    port = args.robot_port
    if args.robot == "sim_ur":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        # gripper_xml = None
        from splatsim.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )

    elif args.robot == "sim_ur_pybullet_push":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        # gripper_xml = None
        # from splatsim.robots.sim_robot_pybullet import PybulletRobotServer
        from splatsim.robots.sim_robot_pybullet_push import PybulletRobotServer

        server = PybulletRobotServer(
           port=port, host=args.hostname,
        )

    elif args.robot == "sim_ur_pybullet_cup":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        # gripper_xml = None
        # from splatsim.robots.sim_robot_pybullet import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_cup import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_pick_planner import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_pick_place_planner import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_assembly import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_articulated import PybulletRobotServer
        from splatsim.robots.sim_robot_pybullet_deformable import PybulletRobotServer

        server = PybulletRobotServer(
           port=port, host=args.hostname,
        )

    elif args.robot == "sim_ur_pybullet_orange":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import OrangeOnPlatePybulletRobotServer

        server = OrangeOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=OrangeOnPlatePybulletRobotServer.SERVE_MODES.GENERATE_DEMOS,
           camera_names=[], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper
        )

    elif args.robot == "sim_ur_pybullet_orange_interactive":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import OrangeOnPlatePybulletRobotServer

        server = OrangeOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=OrangeOnPlatePybulletRobotServer.SERVE_MODES.INTERACTIVE,
            camera_names=camera_names, robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper
        )

    elif args.robot == "sim_ur_pybullet_apple":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import AppleOnPlatePybulletRobotServer

        server = AppleOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=AppleOnPlatePybulletRobotServer.SERVE_MODES.GENERATE_DEMOS,
           camera_names=[], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper
        )

    elif args.robot == "sim_ur_pybullet_apple_interactive":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import AppleOnPlatePybulletRobotServer

        server = AppleOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=AppleOnPlatePybulletRobotServer.SERVE_MODES.INTERACTIVE,
           camera_names=camera_names, robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper
        )

    elif args.robot == "sim_ur_pybullet_apple_interactive-nosplat":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import AppleOnPlatePybulletRobotServer

        server = AppleOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=AppleOnPlatePybulletRobotServer.SERVE_MODES.INTERACTIVE,
           camera_names=[], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper
        )

    elif args.robot == "sim_ur_pybullet_apple_search":
        from splatsim.robots.sim_robot_pybullet_apple_search import AppleSearchPybulletRobotServer

        server = AppleSearchPybulletRobotServer(
           port=port, host=args.hostname, serve_mode=AppleSearchPybulletRobotServer.SERVE_MODES.GENERATE_DEMOS,
           camera_names=[], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper
        )

    elif args.robot == "sim_ur_pybullet_apple_search_interactive":
        from splatsim.robots.sim_robot_pybullet_apple_search import AppleSearchPybulletRobotServer

        server = AppleSearchPybulletRobotServer(
           port=port, host=args.hostname, serve_mode=AppleSearchPybulletRobotServer.SERVE_MODES.INTERACTIVE,
           camera_names=camera_names, robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper
        )

    elif args.robot == "sim_ur_pybullet_banana":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import BananaOnPlatePybulletRobotServer

        server = BananaOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=BananaOnPlatePybulletRobotServer.SERVE_MODES.GENERATE_DEMOS,
           camera_names=[], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper
        )

    elif args.robot == "sim_ur_pybullet_banana_interactive":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import BananaOnPlatePybulletRobotServer

        server = BananaOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=BananaOnPlatePybulletRobotServer.SERVE_MODES.INTERACTIVE,
           camera_names=camera_names, robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper
        )

    elif args.robot == "sim_ur_splat":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        # gripper_xml = None
        # from splatsim.robots.sim_robot_pybullet import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_splat_6DOF import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_splat_servoing import PybulletRobotServer
        from splatsim.robots.sim_robot_pybullet_splat_servoing_improved import GaussianRenderServer
        server = GaussianRenderServer(
           port=port, host=args.hostname, gaussian_path=args.gaussian_path
        )

    elif args.robot == "sim_panda":
        from splatsim.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "franka_emika_panda" / "panda.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )

    elif args.robot == "sim_xarm":
        from splatsim.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "ufactory_xarm7" / "xarm7.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
    
    elif args.robot in ("sim_xarm5_splat_camera", "sim_xarm5_splat_camera_2"):
        from splatsim.robots.xarm.server import XArm5PybulletRobotServerCamera
        from splatsim.utils.camera_setup import arrangement_cameras

        server = XArm5PybulletRobotServerCamera(
            port=port,
            host=args.hostname,
            camera_names=(arrangement_cameras(args.arrangement) if args.arrangement
                          else list(args.cameras)),
            robot_name=args.robot_name,
            cam_i=347,
            use_gripper=use_gripper,
            move_apple=args.move_apple,
            control_hz=args.hz,
            gui=args.gui,
            panel_port=args.panel_port,
            control_mode=args.control_mode,
            sim_clock=args.sim_clock,
        )

    elif args.robot == "xarm5_lite":
        # xArm5 real: driver + cámaras RealSense detrás del mismo protocolo, recorder y panel
        # que el sim (configs/xarm5_real.yaml)
        from splatsim.robots.xarm5_lite_robot_plant import XArmRobot
        from splatsim.robots.xarm_real.cameras import ZMQCamera
        from splatsim.robots.xarm_real.config import load_real_config
        from splatsim.robots.xarm_real.server import XArm5RealServer

        cfg = load_real_config(args.real_config)
        explicit_ip = any(a.startswith(("--robot_ip", "--robot-ip")) for a in sys.argv[1:])
        ip = args.robot_ip if explicit_ip else cfg["ip"]
        cameras = {name: ZMQCamera(cam["port"], host=args.hostname)
                   for name, cam in (cfg.get("cameras") or {}).items()}
        robot = XArmRobot(ip=ip, control_mode=args.control_mode, command_hz=args.hz)
        server = XArm5RealServer(
            robot, cameras, port=port, host=args.hostname, control_hz=args.hz,
            panel_port=None if args.gui == "none" else args.panel_port,
            home_joints=cfg.get("home_joints"), reset_speed=cfg.get("reset_speed", 0.3),
            path=cfg.get("recordings", "~/data/bc_data/real"),
            max_record_seconds=cfg.get("max_record_seconds", 120),
            # el URDF del archivo, o el de la escena del sim en objects.yaml
            urdf_path=cfg.get("urdf") or object_config.get(args.robot_name, {}).get("urdf_path", [None])[0])

    else:
        if args.robot == "xarm":
            from splatsim.robots.xarm_robot import XArmRobot

        elif args.robot == "ur":
            from splatsim.robots.ur import URRobot

            robot = URRobot(robot_ip=args.robot_ip, no_gripper=False)
        elif args.robot == "panda":
            from splatsim.robots.panda import PandaRobot

            robot = PandaRobot(robot_ip=args.robot_ip, no_gripper=False)
        elif args.robot == "bimanual_ur":
            from splatsim.robots.ur import URRobot

            # IP for the bimanual robot setup is hardcoded
            _robot_l = URRobot(robot_ip="192.168.2.11")
            _robot_r = URRobot(robot_ip="192.168.1.11")
            robot = BimanualRobot(_robot_l, _robot_r)
        
        elif args.robot == "ur_pybullet":
            from splatsim.robots.ur_pybullet import URRobotPybullet
            robot = URRobotPybullet(robot_ip=args.robot_ip, no_gripper=True)

        elif args.robot == "none" or args.robot == "print":
            robot = PrintRobot(8)

        else:
            raise NotImplementedError(
                f"Robot {args.robot} not implemented, choose one of: sim_ur, xarm, ur, bimanual_ur, none"
            )
        server = ZMQServerRobot(robot, port=port, host=args.hostname)
        print(f"Starting robot server on port {port}")
    
    
    try:
        server.serve()
    except KeyboardInterrupt:
        if hasattr(server, "shutdown") and type(getattr(server, "shutdown")) == types.MethodType:
            server.shutdown()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
