"""Pruebas del orquestador sin nube: un proveedor simulado corre el runner real en un hilo, con el
bucket en una carpeta local.

    PYTHONPATH=. python tests/test_cloud.py
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

os.environ["SPLATSIM_HEARTBEAT_S"] = "0.05"
os.environ["SPLATSIM_RESULT_RETRY_S"] = "0.01"
os.environ["SPLATSIM_WARMUP_S"] = "0"

import yaml  # noqa: E402

from splatsim.cloud.estimate import (PREFER_CACHED_USD, Estimate, convert_workload, estimate, lookup_rate,  # noqa: E402
                                     prefer_machines, rank, train_profile, train_workload)
from splatsim.cloud import orchestrator  # noqa: E402
from splatsim.cloud.orchestrator import Orchestrator, Settings, train_config  # noqa: E402
from splatsim.cloud.providers import Instance, Offer, Requirements, VastProvider  # noqa: E402
from splatsim.cloud import runner  # noqa: E402
from splatsim.cloud.runner import Run, run_stage  # noqa: E402
from scripts.xarm5.data_jobs import plan_pipeline  # noqa: E402
from splatsim.cloud import stages  # noqa: E402
from splatsim.cloud.stages import describe_pipeline, scan_recordings, scan_states  # noqa: E402
from splatsim.cloud.storage import Storage  # noqa: E402

CPU = "AMD Ryzen 9 7950X 16-Core Processor"
FAKE_TRAIN = """
import sys, time, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[sys.argv.index("--config") + 1]))
assert Path(cfg["dataset_root"], "data.bin").read_text() == "dataset"
if cfg.get("description") == "crash":
    print("boom"); sys.exit(3)
if cfg.get("description") == "hang":
    time.sleep(60)
out = Path(cfg["output_root"], cfg["model_name"]); out.mkdir(parents=True)
for step in range(0, 5):
    print(f"step: {step} loss: 0.1", flush=True); time.sleep(0.02)
(out / "model.safetensors").write_text("weights")
(out / "checkpoints").mkdir(); (out / "checkpoints" / "a.bin").write_text("ckpt")
"""
FAKE_CONVERT = """
import argparse, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--input_dir", nargs="+"); p.add_argument("--output_dir"); p.add_argument("--fps")
p.add_argument("--repo_suffix"); p.add_argument("--cameras", nargs="+"); p.add_argument("--no_push", action="store_true")
p.add_argument("--ignore_selection", action="store_true")
a = p.parse_args()
assert a.no_push
assert Path(a.input_dir[0], "000", "00001.pkl").exists()
if "crash" in a.repo_suffix:
    raise SystemExit(4)
out = Path(a.output_dir, f"splatsim_xarm5_lite_{a.fps}fps_{a.repo_suffix}")
out.mkdir(parents=True)     # como LeRobot: falla si ya existe
time.sleep(0.2)
print("episodes:  50%|#####     | 1/2 [00:01<00:01,  1.00it/s]", flush=True)
(out / "data.bin").write_text("dataset")
"""
FAKE_RENDER = """
import argparse, pickle
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("input"); p.add_argument("--out"); p.add_argument("--fps"); p.add_argument("--arrangement")
p.add_argument("--workers"); p.add_argument("--no-with-original", action="store_true")
a = p.parse_args()
source = sorted(q for q in Path(a.input).iterdir() if q.is_dir())
assert source
print("[1/1] 000: 3 frames", flush=True)
for episode in source:
    images = Path(a.out, episode.name, "image_1"); images.mkdir(parents=True)
    for frame in ("00001", "00002"):
        (images.parent / f"{frame}.pkl").write_bytes(pickle.dumps({"action": 1}))
        (images / f"base_rgb_{frame}.png").write_bytes(b"png")
"""


class FolderStorage:
    """Misma interfaz que Storage, sobre una carpeta."""

    def __init__(self, root: Path):
        self.root, self.remote = Path(root), str(root)
        self.corrupt = False
        self.excluded = {}

    def read(self, key):
        path = self.root / key
        return path.read_text() if path.exists() else None

    def write(self, key, text):
        (self.root / key).parent.mkdir(parents=True, exist_ok=True)
        tmp = (self.root / key).with_name(f".{threading.get_ident()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, self.root / key)

    def upload(self, local, key, exclude=None):
        self.excluded[key] = exclude
        shutil.copytree(local, self.root / key, dirs_exist_ok=True)

    def upload_file(self, local, key):
        (self.root / key).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, self.root / key)

    def download(self, key, local):
        shutil.copytree(self.root / key, local, dirs_exist_ok=True)
        if self.corrupt:
            (Path(local) / "model.safetensors").write_text("dañado")

    def copy(self, source, target):
        shutil.copytree(self.root / source, self.root / target, dirs_exist_ok=True)

    def delete(self, key):
        shutil.rmtree(self.root / key)


def offer(id=1, machine=10, cpu=CPU, dph=1.0, gpu="RTX 5090"):
    return Offer(id=id, machine_id=machine, gpu_name=gpu, cpu_name=cpu, cores=32, ram_gb=64, pcie_gbps=50, dph=dph,
                 down_cost_gb=0.01, up_cost_gb=0.02, inet_down_mbps=8000, inet_up_mbps=8000, reliability=0.99)


class FakeProvider:
    """Cada create() arranca el runner real, como lo haría el contenedor."""

    def __init__(self, storage, offers=None, autostart=True):
        self.storage, self.offers, self.autostart = storage, offers or [offer()], autostart
        self.live: dict[int, Instance] = {}
        self.created: list[tuple[int, dict]] = []
        self.destroyed: list[int] = []
        self.credit = 100.0
        self.next_id = 100
        self.bad_machines: set[int] = set()

    def balance(self):
        return self.credit

    def search(self, req, disk_gb):
        return [o for o in self.offers if not req.gpu_name or o.gpu_name == req.gpu_name]

    def create(self, chosen, image, args, env, disk_gb, label):
        self.next_id += 1
        instance = Instance(self.next_id, "loading", label)
        self.live[instance.id] = instance
        spec = json.loads(self.storage.read(f"{args[0]}/spec.json"))
        self.created.append((chosen.machine_id, spec))
        if chosen.machine_id in self.bad_machines:
            spec["health"] = [sys.executable, "-c", "raise SystemExit('CUDA out of memory')"]
        if self.autostart:
            threading.Thread(target=self._boot, args=(instance, args[0], spec), daemon=True).start()
        return instance.id

    def _boot(self, instance, key, spec):
        instance.status = "running"
        run_stage(Run(self.storage, key), spec)
        instance.status = "exited"

    def instances(self):
        return list(self.live.values())

    def destroy(self, instance_id):
        self.destroyed.append(instance_id)
        self.live.pop(instance_id, None)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.storage = FolderStorage(self.tmp / "bucket")
        self.dataset = self.tmp / "ds"
        self.dataset.mkdir()
        (self.dataset / "data.bin").write_text("dataset")
        self.script = self.tmp / "fake_train.py"
        self.script.write_text(FAKE_TRAIN)
        self.convert_script = self.tmp / "fake_convert.py"
        self.convert_script.write_text(FAKE_CONVERT)
        self.provider = FakeProvider(self.storage)
        self.orch = self.make()

    def make(self):
        return Orchestrator(self.tmp / "queue", self.provider, self.storage, health=None, python=sys.executable,
                            scripts={"train": str(self.script), "convert": str(self.convert_script)},
                            pod_root=str(self.tmp / "pod"))

    def cfg(self, name="m", **extra):
        return {"model_name": name, "output_root": str(self.tmp / "models"), "dataset_root": str(self.tmp / "pod_ds"),
                "training_steps": 100, "batch_size": 32, "save_freq": 100, "cameras": ["a", "b"],
                "image_augmentation": False, "torch_compile": True, **extra}

    def submit(self, name="m", max_cost=1.0, **extra):
        return self.orch.submit(name, self.cfg(name, **extra), str(self.dataset), Requirements(), max_cost)

    def run_until(self, condition, timeout=20.0):
        end = time.time() + timeout
        while time.time() < end:
            self.orch.tick()
            if condition():
                return
            time.sleep(0.03)
        self.fail(f"tiempo agotado; trabajos: {[(j['id'], j['status'], j.get('error')) for j in self.orch.jobs]}")


class TestPipeline(Base):
    def test_job_runs_and_model_arrives_verified(self):
        job = self.submit()
        self.run_until(lambda: job["status"] == "done")
        target = self.tmp / "models" / "m"
        self.assertEqual((target / "model.safetensors").read_text(), "weights")
        self.assertEqual((target / "checkpoints" / "a.bin").read_text(), "ckpt")
        self.assertIn(job["status"], ("done",))
        self.assertEqual(self.provider.live, {}, "el pod debe quedar eliminado")
        self.assertGreater(job["cost"], 0)
        self.assertEqual(self.orch.history[-1]["profile"], "bs32-cam2-noaug-compile")

    def test_failed_command_fails_without_retry_and_frees_pod(self):
        job = self.submit(description="crash")
        self.run_until(lambda: job["status"] == "failed")
        self.assertIn("command_failed", job["error"])
        self.assertEqual(len(self.provider.created), 1)
        self.assertEqual(self.provider.live, {})
        self.assertIn("boom", self.storage.read(f"{job['run_key']}/run.log"))

    def test_bad_host_moves_to_another_machine(self):
        self.provider.offers = [offer(1, machine=10), offer(2, machine=20, dph=1.5)]
        self.provider.bad_machines = {10}
        job = self.submit()
        self.run_until(lambda: job["status"] == "done")
        self.assertEqual([m for m, _ in self.provider.created], [10, 20])
        self.assertEqual(job["excluded_machines"], [10])
        self.assertEqual(job["attempts"], 2)

    def test_gives_up_after_max_attempts(self):
        self.provider.offers = [offer(i, machine=i) for i in range(1, 6)]
        self.provider.bad_machines = {1, 2, 3, 4, 5}
        job = self.submit()
        self.run_until(lambda: job["status"] == "failed")
        self.assertEqual(len(self.provider.created), 3)

    def test_deadline_on_the_pod_stops_the_job(self):
        run = Run(self.storage, "runs/x")
        spec = {"command": [sys.executable, "-c", "import time; time.sleep(60)"], "deadline_s": 0.3}
        start = time.time()
        run_stage(run, spec)
        self.assertLess(time.time() - start, 20)
        self.assertEqual(json.loads(self.storage.read("runs/x/result.json"))["reason"], "deadline")

    def test_result_is_written_after_outputs(self):
        out = self.tmp / "out"
        out.mkdir()
        (out / "m.bin").write_text("x")
        seen = {}
        write = self.storage.write

        def spy(key, text):
            if key.endswith("result.json"):
                seen["outputs_present"] = (self.tmp / "bucket/runs/y/outputs/m.bin").exists()
            write(key, text)

        self.storage.write = spy
        run_stage(Run(self.storage, "runs/y"), {"command": [sys.executable, "-c", ""], "deadline_s": 30,
                                                "outputs": [{"key": "runs/y/outputs", "path": str(out)}]})
        self.assertTrue(seen["outputs_present"])

    def test_corrupt_download_fails_and_retry_only_downloads_again(self):
        self.storage.corrupt = True
        job = self.submit()
        self.run_until(lambda: job["status"] == "failed")
        self.assertIn("sha256", job["error"])
        self.storage.corrupt = False
        self.orch.retry(job["id"])
        self.run_until(lambda: job["status"] == "done")
        self.assertEqual(len(self.provider.created), 1, "no debe entrenar de nuevo")

    def test_stop_destroys_pod(self):
        self.provider.autostart = False
        job = self.submit()
        self.run_until(lambda: job["status"] == "starting")
        instance = job["instance_id"]
        self.orch.stop(job["id"])
        self.assertEqual(job["status"], "stopped")
        self.assertIn(instance, self.provider.destroyed)

    def test_cost_limit_destroys_pod(self):
        self.provider.autostart = False
        job = self.submit(max_cost=1.0)
        self.run_until(lambda: job["status"] == "starting")
        job["billing_ts"] = time.time() - 2 * 3600    # dos horas a $1/h
        self.orch.tick()
        self.assertEqual(job["status"], "failed")
        self.assertIn("máximo", job["error"])
        self.assertEqual(self.provider.live, {})


def make_recordings(root: Path, episodes=("000", "001"), cameras=("base_rgb", "wrist_rgb"), discarded=()):
    for episode in episodes:
        (root / episode / "image_1").mkdir(parents=True)
        for frame in ("00001", "00002"):
            (root / episode / f"{frame}.pkl").write_bytes(b"x" * 100)
            for camera in cameras:
                (root / episode / "image_1" / f"{camera}_{frame}.png").write_bytes(b"y" * 1000)
    if discarded:
        (root / "selection.yaml").write_text(yaml.safe_dump({"discarded": list(discarded)}))


class TestChain(Base):
    def setUp(self):
        super().setUp()
        self.recordings = self.tmp / "gello_x"
        make_recordings(self.recordings)
        self.datasets = self.tmp / "datasets"
        self.orch.settings = Settings(max_pods=2, budget_usd=100)

    def convert(self, suffix="a"):
        return self.orch.submit_convert(f"convert {suffix}", {"input": str(self.recordings), "fps": 10, "suffix": suffix},
                                        Requirements(gpu_name=""), 1.0, output_dir=str(self.datasets))

    def chained_train(self, parent):
        cfg = self.cfg("m", dataset_root=str(self.tmp / "pod_ds"))
        return self.orch.submit("m", cfg, parent["output"], Requirements(), 1.0, depends_on=parent["id"])

    def test_convert_then_train(self):
        parent = self.convert()
        real_frames, parent["frames"] = parent["frames"], 100_000
        child = self.chained_train(parent)
        parent["frames"] = real_frames
        self.assertEqual(child["status"], "pending")
        self.assertGreater(child["dataset_gb"], 0, "se usa el tamaño previsto")
        self.run_until(lambda: parent["status"] == "done" or parent["status"] == "failed")
        self.assertEqual(parent["status"], "done", parent.get("error"))
        name = "splatsim_xarm5_lite_10fps_a"
        self.assertEqual((self.datasets / name / "data.bin").read_text(), "dataset")
        self.run_until(lambda: child["status"] in ("done", "failed"))
        self.assertEqual(child["status"], "done", child.get("error"))
        self.assertEqual((self.tmp / "models/m/model.safetensors").read_text(), "weights")
        self.assertLess(child["dataset_gb"], 0.01, "ya con el tamaño real")

    def test_child_waits_for_parent_without_a_pod(self):
        self.provider.autostart = False
        parent = self.convert()
        child = self.chained_train(parent)
        self.run_until(lambda: parent["status"] == "starting")
        self.orch.tick()
        self.assertEqual(child["status"], "pending")
        self.assertIn(f"#{parent['id']}", child["wait"])
        self.assertEqual(len(self.provider.created), 1)

    def test_failed_parent_keeps_child_waiting_and_retry_resumes(self):
        parent = self.orch.submit_convert("c", {"input": str(self.recordings), "fps": 10, "suffix": "crash"},
                                          Requirements(gpu_name=""), 1.0, output_dir=str(self.datasets))
        child = self.chained_train(parent)
        self.run_until(lambda: parent["status"] == "failed")
        self.orch.tick()
        self.assertEqual(child["status"], "pending")
        self.orch.remove(parent["id"])
        self.assertIn(parent, self.orch.jobs, "no se quita un padre con hijos esperando")
        self.assertTrue(self.orch.warnings)

    def test_bucket_keeps_only_the_dataset(self):
        job = self.convert()
        self.run_until(lambda: job["status"] in ("done", "failed"))
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(self.storage.read("datasets/splatsim_xarm5_lite_10fps_a/data.bin"), "dataset")
        self.assertIsNone(self.storage.read(f"{job['run_key']}/outputs/dataset/data.bin"))
        self.assertFalse((self.tmp / "bucket/recordings/gello_x").exists())

    def test_recordings_stay_while_another_job_needs_them(self):
        first, second = self.convert("a"), self.convert("b")
        self.orch.settings.max_pods = 1
        self.run_until(lambda: first["status"] == "done")
        self.assertTrue((self.tmp / "bucket/recordings/gello_x").exists())
        self.run_until(lambda: second["status"] == "done")
        self.assertFalse((self.tmp / "bucket/recordings/gello_x").exists())

    def test_convert_runs_without_gpu_check(self):
        self.orch.health = ["false"]
        job = self.convert()
        self.run_until(lambda: job["status"] in ("done", "failed"))
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertIsNone(self.provider.created[0][1]["health"])

    def test_restarted_conversion_starts_clean(self):
        (self.tmp / "pod/datasets/splatsim_xarm5_lite_10fps_a").mkdir(parents=True)
        job = self.convert()
        self.run_until(lambda: job["status"] in ("done", "failed"))
        self.assertEqual(job["status"], "done", job.get("error"))

    def test_rate_is_recorded_in_frames(self):
        job = self.convert()
        self.run_until(lambda: job["status"] == "done")
        record = self.orch.history[-1]
        self.assertEqual((record["kind"], record["profile"]), ("convert", "convert-cam2"))
        self.assertAlmostEqual(record["rate"], 4 / job["result"]["seconds"]["run"], places=1)

    def test_existing_or_queued_dataset_is_rejected(self):
        self.convert("a")
        with self.assertRaises(ValueError):
            self.convert("a")
        (self.datasets / "splatsim_xarm5_lite_10fps_b").mkdir(parents=True)
        with self.assertRaises(ValueError):
            self.convert("b")

    def test_child_must_use_the_parent_dataset(self):
        parent = self.convert()
        with self.assertRaises(ValueError):
            self.orch.submit("m", self.cfg(), str(self.tmp / "otro"), Requirements(), 1.0, depends_on=parent["id"])


def make_states(root: Path, episodes=("000", "001"), seconds=1.0):
    for episode in episodes:
        (root / episode).mkdir(parents=True)
        for i, t in enumerate((0.0, seconds / 2, seconds), 1):
            frame = {"sim_time": t, "sim_joint_positions": [0.0] * 19, "state": [0.0]}
            (root / episode / f"{i:05d}.pkl").write_bytes(pickle.dumps(frame))


class TestRenderChain(Base):
    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, stages, "scene_assets", stages.scene_assets)
        self.addCleanup(setattr, stages, "pod_files", stages.pod_files)
        scene = self.tmp / "scene"
        scene.mkdir()
        (scene / "point_cloud.ply").write_text("splat")
        stages.scene_assets = lambda: ([{"local": str(scene), "key": "assets/scene"}],
                                       [{"key": "assets/scene", "path": str(self.tmp / "pod/scene")}])
        stages.pod_files = lambda: {}
        self.addCleanup(self.drain)     # corre antes que lo anterior: ningún hilo debe ver la escena real
        self.render_script = self.tmp / "fake_render.py"
        self.render_script.write_text(FAKE_RENDER)
        self.orch.scripts.update(render=str(self.render_script), mirror=str(self.render_script))
        self.orch.settings = Settings(max_pods=2, budget_usd=100)
        self.source = self.tmp / "gello_x"
        make_states(self.source)
        self.datasets = self.tmp / "datasets"

    def drain(self):
        self.orch.settings.paused = True
        end = time.time() + 10
        while self.orch.busy and time.time() < end:
            time.sleep(0.02)

    def plan(self, mirror=True, convert=True):
        return plan_pipeline(str(self.source), {"fps": 10, "arrangement": None, "workers": 1},
                             {"with_original": True, "arrangement": None, "workers": 1} if mirror else None,
                             {"output_dir": str(self.datasets), "fps": 10, "suffix": "x", "cameras": None,
                              "ignore_selection": False, "push": False} if convert else None)

    def submit_plan(self, **kwargs):
        return self.orch.submit_pipeline(self.plan(**kwargs), Requirements(), Requirements(gpu_name=""), 1.0)

    def test_stages_chain_through_the_bucket_and_only_the_last_comes_down(self):
        render, mirror, convert = self.submit_plan()
        self.run_until(lambda: convert["status"] in ("done", "failed"), timeout=40)
        self.assertEqual(convert["status"], "done", convert.get("error"))
        self.assertEqual((self.datasets / "splatsim_xarm5_lite_10fps_x/data.bin").read_text(), "dataset")
        self.assertFalse(Path(render["output"]).exists(), "lo intermedio no baja al lab")
        self.assertFalse(Path(mirror["output"]).exists())
        self.assertEqual(sorted(p.name for p in (self.tmp / "bucket/recordings").iterdir()), [],
                         "lo intermedio tampoco se queda en el bucket")
        self.assertEqual(self.storage.read("assets/scene/point_cloud.ply"), "splat", "la escena sí queda")
        self.assertEqual(self.provider.live, {})

    def test_render_alone_comes_down_verified(self):
        render, = self.submit_plan(mirror=False, convert=False)
        self.run_until(lambda: render["status"] in ("done", "failed"))
        self.assertEqual(render["status"], "done", render.get("error"))
        self.assertTrue((Path(render["output"]) / "000/image_1/base_rgb_00001.png").exists())
        self.assertEqual(render["output_dir"], render["output"])
        self.assertEqual(list((self.tmp / "bucket/recordings").iterdir()), [])

    def test_stages_use_their_own_image(self):
        images = []
        create = self.provider.create

        def spy(chosen, image, *args):
            images.append(image)
            return create(chosen, image, *args)

        self.provider.create = spy
        _, _, convert = self.submit_plan()
        self.run_until(lambda: convert["status"] == "done", timeout=40)
        self.assertEqual([i.split("/")[-1].split(":")[0] for i in images],
                         ["splatsim-render", "splatsim-render", "splatsim-train"])

    def test_output_stays_until_the_child_finishes(self):
        render, mirror = self.submit_plan(convert=False)
        self.run_until(lambda: render["status"] == "done")
        self.assertTrue((self.tmp / "bucket" / f"recordings/{Path(render['output']).name}").exists())
        self.run_until(lambda: mirror["status"] in ("done", "failed"))
        self.assertEqual(mirror["status"], "done", mirror.get("error"))
        self.assertFalse((self.tmp / "bucket" / f"recordings/{Path(render['output']).name}").exists())

    def test_existing_output_is_rejected(self):
        Path(self.plan()[0]["output"]).mkdir()
        with self.assertRaisesRegex(ValueError, "Ya existe"):
            self.submit_plan()

    def test_chained_training_waits_for_the_whole_chain(self):
        *_, convert = self.submit_plan()
        cfg = self.cfg("m")
        child = self.orch.submit("m", cfg, convert["output"], Requirements(), 1.0, depends_on=convert["id"])
        self.assertEqual(child["status"], "pending")
        self.run_until(lambda: child["status"] in ("done", "failed"), timeout=60)
        self.assertEqual(child["status"], "done", child.get("error"))


class TestPipelinePlan(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.source = self.tmp / "gello"
        make_states(self.source, seconds=1.0)

    def test_frames_match_the_resampling_of_render_recordings(self):
        self.assertEqual(scan_states(self.source, 10)["frames"], 2 * 11)

    def test_recording_without_joint_state_is_rejected(self):
        (self.tmp / "old" / "000").mkdir(parents=True)
        (self.tmp / "old" / "000" / "00001.pkl").write_bytes(pickle.dumps({"state": 1}))
        with self.assertRaisesRegex(ValueError, "realtime"):
            scan_states(self.tmp / "old", 10)

    def test_mirror_doubles_and_sizes_carry_over(self):
        plan = plan_pipeline(str(self.source), {"fps": 10, "workers": 3},
                             {"with_original": True, "workers": 3},
                             {"output_dir": "/d", "fps": 10, "suffix": "s"})
        render, mirror, convert = describe_pipeline(plan)
        self.assertEqual((render["frames"], mirror["frames"], convert["frames"]), (22, 44, 44))
        self.assertEqual(len(render["cameras"]), 4)
        self.assertAlmostEqual(mirror["input_gb"], 22 * 4 * 0.44e-3)
        self.assertAlmostEqual(convert["recordings_gb"], 44 * 4 * 0.44e-3)
        self.assertEqual(convert["exclude"], [])
        self.assertEqual(convert["args"]["input"], mirror["output"])

    def test_real_scene_assets_are_minimal(self):
        if not Path(stages.OBJECTS_FILE).exists() or not (Path.home() / "data/Colmaps/260729_2003").exists():
            self.skipTest("sin la escena del lab")
        uploads, inputs = stages.scene_assets()
        keys = [u["key"] for u in uploads]
        self.assertIn("assets/Colmaps/260729_2003/images/0348.jpg", keys)
        self.assertEqual(sum("/images/" in k for k in keys), 2)
        self.assertTrue(all(Path(u["local"]).expanduser().exists() for u in uploads), uploads)
        self.assertIn({"key": "assets/xarm5_lite_urdf", "path": "~/data/xarm5_lite_urdf"}, inputs)
        self.assertIn({"key": "assets/output/plastic_apple", "path": "~/data/output/plastic_apple"}, inputs)


class TestRecordings(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_scan_skips_discarded_and_unused_cameras(self):
        make_recordings(self.tmp, ("000", "001"), ("base_rgb", "wrist_rgb", "base_left_rgb"), discarded=["001"])
        scan = scan_recordings(self.tmp, ["base_rgb", "wrist_rgb"])
        self.assertEqual((scan["episodes"], scan["frames"], scan["cameras"]), (1, 2, ["base_rgb", "wrist_rgb"]))
        self.assertEqual(scan["exclude"], ["/001/**", "**/base_left_rgb_*.png"])
        self.assertAlmostEqual(scan["gb"], round((2 * 100 + 4 * 1000) / 1e9, 2))

    def test_ignore_selection_includes_discarded(self):
        make_recordings(self.tmp, discarded=["001"])
        self.assertEqual(scan_recordings(self.tmp, ignore_selection=True)["episodes"], 2)

    def test_state_only_recordings_are_rejected(self):
        (self.tmp / "000").mkdir()
        (self.tmp / "000" / "00001.pkl").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "renderizar"):
            scan_recordings(self.tmp)

    def test_missing_camera_is_rejected(self):
        make_recordings(self.tmp)
        with self.assertRaisesRegex(ValueError, "base_left_rgb"):
            scan_recordings(self.tmp, ["base_left_rgb"])

    @unittest.skipUnless(shutil.which("rclone"), "sin rclone")
    def test_rclone_applies_the_scan_filters_and_copies_inside_the_bucket(self):
        source = self.tmp / "gello"
        make_recordings(source, ("000", "001"), ("base_rgb", "wrist_rgb"), discarded=["001"])
        scan = scan_recordings(source, ["base_rgb"])
        bucket = self.tmp / "bucket"
        storage = Storage(str(bucket))
        storage.upload(source, "recordings/g", scan["exclude"])
        uploaded = {str(p.relative_to(bucket / "recordings/g")) for p in (bucket / "recordings/g").rglob("*") if p.is_file()}
        self.assertEqual(uploaded, {"selection.yaml", "000/00001.pkl", "000/00002.pkl",
                                    "000/image_1/base_rgb_00001.png", "000/image_1/base_rgb_00002.png"})
        storage.copy("recordings/g", "copia/g")
        self.assertTrue((bucket / "copia/g/000/00001.pkl").exists())
        storage.delete("recordings/g")
        self.assertFalse((bucket / "recordings/g").exists())


class TestPodLifecycle(Base):
    """Lo que pasó en Vast: el contenedor se reinicia al terminar y el almacenamiento puede fallar."""

    def start(self, command="pass", deadline=30):
        self.storage.write("runs/z/spec.json", json.dumps(
            {"command": [sys.executable, "-c", command], "deadline_s": deadline}))
        os.environ["SPLATSIM_REMOTE"] = self.storage.remote
        self.addCleanup(os.environ.pop, "SPLATSIM_REMOTE", None)
        return runner.run_once(Run(self.storage, "runs/z"), self.storage, "runs/z")

    def test_finished_job_does_not_run_again_after_restart(self):
        self.start("print('primera')")
        first = self.storage.read("runs/z/result.json")
        self.start("raise SystemExit(9)")
        self.assertEqual(self.storage.read("runs/z/result.json"), first)

    def test_deadline_survives_restarts(self):
        self.storage.write("runs/z/boot.json", json.dumps({"first": time.time() - 1000, "boots": 1}))
        self.start("import time; time.sleep(60)", deadline=100)
        self.assertEqual(json.loads(self.storage.read("runs/z/result.json"))["reason"], "deadline")

    def test_restart_loop_is_cut(self):
        self.storage.write("runs/z/boot.json", json.dumps({"first": time.time(), "boots": runner.MAX_BOOTS}))
        self.assertEqual(self.start(), 1)
        self.assertEqual(json.loads(self.storage.read("runs/z/result.json"))["reason"], "restart_loop")

    def test_status_failure_does_not_lose_the_result(self):
        write = self.storage.write

        def flaky(key, text):
            if key.endswith("status.json"):
                raise RuntimeError("501 Not Implemented")
            write(key, text)

        self.storage.write = flaky
        self.assertEqual(self.start(), 0)
        self.assertTrue(json.loads(self.storage.read("runs/z/result.json"))["ok"])

    def test_result_write_is_retried(self):
        write, fails = self.storage.write, [1, 2]

        def flaky(key, text):
            if key.endswith("result.json") and fails:
                fails.pop()
                raise RuntimeError("corte de red")
            write(key, text)

        self.storage.write = flaky
        self.start()
        self.assertTrue(json.loads(self.storage.read("runs/z/result.json"))["ok"])

    def test_pod_stops_its_own_instance(self):
        seen = {}

        class Response:
            status = 200
            __enter__ = lambda s: s
            __exit__ = lambda s, *a: None

        def urlopen(request, timeout):
            seen.update(url=request.full_url, method=request.method, body=request.data,
                        auth=request.get_header("Authorization"))
            return Response()

        original = runner.urllib.request.urlopen
        runner.urllib.request.urlopen = urlopen
        self.addCleanup(setattr, runner.urllib.request, "urlopen", original)
        os.environ.update(CONTAINER_API_KEY="k", CONTAINER_ID="55")
        self.addCleanup(lambda: [os.environ.pop(k) for k in ("CONTAINER_API_KEY", "CONTAINER_ID")])
        self.assertEqual(runner.stop_own_instance(), "http 200")
        self.assertEqual((seen["method"], seen["auth"]), ("PUT", "Bearer k"))
        self.assertTrue(seen["url"].endswith("/instances/55/"))
        self.assertEqual(json.loads(seen["body"]), {"state": "stopped"})

    def test_rate_ignores_the_warmup(self):
        run = Run(self.storage, "runs/w")
        times = iter([100.0, 100.0])
        original, runner.WARMUP_S = runner.WARMUP_S, 50
        self.addCleanup(setattr, runner, "WARMUP_S", original)
        clock = runner.time.monotonic
        self.addCleanup(setattr, runner.time, "monotonic", clock)
        for now, step in [(0.0, 0), (40.0, 10), (60.0, 100), (160.0, 600)]:
            runner.time.monotonic = lambda now=now: now
            run._track(f"step: {step} loss: 0.1")
        self.assertEqual((run.first_step[0], run.last_step[0]), (10, 600))
        self.assertAlmostEqual(run.rate(), 590 / 120, places=2)

    def test_progress_only_follows_the_episode_bar(self):
        run = Run(self.storage, "runs/w")
        run._track("Casting the dataset: 100%|####| 300/300 [00:01<00:00]")
        run._track("/data/12/34 svt 1/2")
        run._track("[w0] Loading 3/4")
        self.assertNotIn("done", run.status)
        run._track("episodes:  40%|####      | 2/5 [00:01<00:02,  1.00it/s]")
        self.assertEqual((run.status["done"], run.status["total"]), (2, 5))

    def test_no_credentials_is_reported(self):
        self.assertIn("sin CONTAINER", runner.stop_own_instance())


class TestLimits(Base):
    def test_parallel_jobs_respect_max_pods(self):
        self.provider.autostart = False
        self.orch.settings = Settings(max_pods=2, budget_usd=100)
        jobs = [self.submit(f"m{i}") for i in range(3)]
        self.run_until(lambda: sum(j["status"] == "starting" for j in jobs) == 2)
        for _ in range(5):
            self.orch.tick()
        self.assertEqual(sum(j["status"] == "starting" for j in jobs), 2)
        self.assertEqual(jobs[2]["status"], "pending")

    def test_two_jobs_really_run_at_once(self):
        self.orch.settings = Settings(max_pods=2, budget_usd=100)
        jobs = [self.submit(f"m{i}") for i in range(2)]
        self.run_until(lambda: all(j["status"] == "done" for j in jobs))
        self.assertTrue((self.tmp / "models/m0/model.safetensors").exists())
        self.assertTrue((self.tmp / "models/m1/model.safetensors").exists())

    def test_budget_counts_what_active_jobs_may_spend(self):
        self.provider.autostart = False
        self.orch.settings = Settings(max_pods=5, budget_usd=3.0)
        first, second = self.submit("a", max_cost=2.0), self.submit("b", max_cost=2.0)
        self.run_until(lambda: first["status"] == "starting")
        for _ in range(3):
            self.orch.tick()
        self.assertEqual(second["status"], "pending")
        self.assertIn("presupuesto", second["wait"])

    def test_low_balance_keeps_job_pending(self):
        self.provider.credit = 0.5
        job = self.submit(max_cost=1.0)
        self.run_until(lambda: "saldo" in job.get("wait", ""))
        self.assertEqual(job["status"], "pending")
        self.assertEqual(self.provider.created, [])

    def test_offer_over_max_cost_is_not_used(self):
        self.provider.offers = [offer(dph=5.0)]
        cfg = self.cfg(training_steps=63000)    # 2.8 h a 6.30 pasos/s × $5/h
        job = self.orch.submit("x", cfg, str(self.dataset), Requirements(), 1.0)
        self.run_until(lambda: "sin ofertas" in job.get("wait", ""))
        self.assertEqual(self.provider.created, [])

    def test_pause_launches_nothing(self):
        self.orch.settings.paused = True
        job = self.submit()
        for _ in range(3):
            self.orch.tick()
        self.assertEqual(job["status"], "pending")


class TestRestart(Base):
    def test_new_panel_follows_running_pod(self):
        self.provider.autostart = False
        job = self.submit()
        self.run_until(lambda: job["status"] == "starting")
        instance = job["instance_id"]
        self.orch = self.make()
        again = self.orch.get(job["id"])
        self.orch.tick()
        self.assertEqual(again["status"], "starting")
        self.assertEqual(self.provider.destroyed, [], "no debe eliminar un pod que sigue en uso")
        self.assertIn(instance, self.provider.live)

    def test_orphan_is_destroyed_but_other_pods_are_not(self):
        self.provider.live[7] = Instance(7, "running", f"{self.orch.label_prefix}0009_1_x")
        self.provider.live[8] = Instance(8, "running", "experimento-a-mano")
        self.provider.live[9] = Instance(9, "running", "splatsim-abc123-de-otra-cola")
        self.orch.tick()
        self.assertEqual(self.provider.destroyed, [7])
        self.assertIn(8, self.provider.live)
        self.assertIn(9, self.provider.live)
        self.assertTrue(self.orch.warnings)

    def test_state_survives_on_disk(self):
        self.submit()
        data = yaml.safe_load((self.tmp / "queue/jobs.yaml").read_text())
        self.assertEqual(data["jobs"][0]["name"], "m")


class TestStubbornPod(Base):
    def test_a_pod_that_resists_does_not_fail_the_job_and_gets_destroyed_later(self):
        self.orch.tick()
        real, calls = self.provider.destroy, []

        def destroy(instance_id):
            calls.append(instance_id)
            if len(calls) == 1:
                raise RuntimeError("vastai destroy instance: sin respuesta en 120 s")
            real(instance_id)
        self.provider.destroy = destroy
        job = self.submit()
        self.run_until(lambda: job["status"] == "done" and not self.provider.live)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertTrue(any("No se pudo eliminar" in w for w in self.orch.warnings))

    def test_a_new_pod_is_not_taken_for_an_orphan_while_its_job_is_starting(self):
        self.orch.tick()
        self.orch._orphans_checked = False
        self.orch.busy.add(999)
        self.provider.live[7] = Instance(7, "loading", self.orch.label_prefix + "recien_creado")
        self.orch.tick()
        self.assertIn(7, self.provider.live)
        self.orch.busy.discard(999)
        self.orch.tick()
        self.assertNotIn(7, self.provider.live)


class TestMachineMemory(Base):
    """Un pod arranca en segundos donde la imagen ya está, y una máquina que falló no merece otra espera."""

    def setUp(self):
        super().setUp()
        self.provider.offers = [offer(1, machine=10, dph=1.0), offer(2, machine=20, dph=1.2)]
        self.orch = self.make()

    def machines_used(self):
        return [machine for machine, _ in self.provider.created]

    def run_job(self, name):
        job = self.submit(name)
        self.run_until(lambda: job["status"] in ("done", "failed"))
        self.assertEqual(job["status"], "done", job.get("error"))
        return job

    def test_first_job_takes_the_cheapest_and_the_boot_is_recorded(self):
        job = self.run_job("a")
        self.assertEqual(self.machines_used(), [10])
        self.assertIsNotNone(job["boot_s"])
        self.assertTrue(self.orch.machines[10]["ok"])
        self.assertEqual(self.orch.history[-1]["boot_s"], job["boot_s"])

    def test_a_machine_that_has_the_image_wins_if_it_costs_a_little_more(self):
        self.orch._note_machine(20, True, 300)
        self.run_job("a")
        self.assertEqual(self.machines_used(), [20])

    def test_but_not_if_it_costs_much_more(self):
        self.provider.offers = [offer(1, machine=10, dph=1.0), offer(2, machine=20, dph=9.0)]
        self.orch._note_machine(20, True, 300)
        self.run_job("a")
        self.assertEqual(self.machines_used(), [10])

    def test_a_chosen_offer_still_comes_first(self):
        self.orch._note_machine(20, True, 300)
        job = self.orch.submit("a", self.cfg("a"), str(self.dataset), Requirements(), 1.0, offer_id=1)
        self.run_until(lambda: job["status"] == "done")
        self.assertEqual(self.machines_used(), [10])

    def test_a_machine_that_failed_goes_last_for_the_next_jobs(self):
        self.provider.bad_machines.add(10)
        self.run_job("a")
        self.assertEqual(self.machines_used(), [10, 20])
        self.assertFalse(self.orch.machines[10]["ok"])
        self.provider.created.clear()
        self.provider.bad_machines.clear()
        self.provider.offers = [offer(1, machine=10, dph=0.5), offer(2, machine=20, dph=1.2)]
        self.run_job("b")
        self.assertEqual(self.machines_used(), [20])

    def test_memory_survives_a_restart(self):
        self.orch._note_machine(20, True, 300)
        self.assertTrue(self.make().machines[20]["ok"])

    def test_memory_expires(self):
        self.orch._note_machine(20, True, 300)
        self.orch.machines[20]["ts"] -= orchestrator.MACHINE_MEMORY_S + 1
        self.run_job("a")
        self.assertEqual(self.machines_used(), [10])

    def test_the_offers_say_which_machine_has_the_image(self):
        self.orch._note_machine(20, True, 300)
        ranked = self.orch.estimate(self.cfg(), str(self.dataset), Requirements())
        self.assertEqual([(e.offer.machine_id, e.cached) for e in ranked], [(20, True), (10, False)])


class TestPreferMachines(unittest.TestCase):
    @staticmethod
    def est(machine, total, dph=1.0):
        return Estimate(offer(machine, machine=machine, dph=dph), None, "baja", "", None, None, 0.0, total)

    def order(self, estimates, cached=(), failed=()):
        return [e.offer.machine_id for e in prefer_machines(estimates, set(cached), set(failed))]

    def test_within_the_margin(self):
        self.assertEqual(self.order([self.est(1, 1.0), self.est(2, 1.0 + PREFER_CACHED_USD)], cached=[2]), [2, 1])

    def test_past_the_margin(self):
        self.assertEqual(self.order([self.est(1, 1.0), self.est(2, 1.01 + PREFER_CACHED_USD)], cached=[2]), [1, 2])

    def test_without_estimates_compares_the_hourly_price(self):
        self.assertEqual(self.order([self.est(1, None, 1.0), self.est(2, None, 1.2)], cached=[2]), [2, 1])
        self.assertEqual(self.order([self.est(1, None, 1.0), self.est(2, None, 1.3)], cached=[2]), [1, 2])

    def test_no_estimate_does_not_beat_a_known_price(self):
        self.assertEqual(self.order([self.est(1, 1.0), self.est(2, None, 0.5)], cached=[2]), [1, 2])

    def test_the_margin_is_counted_from_the_best_machine_that_did_not_fail(self):
        estimates = [self.est(1, 0.1), self.est(2, 1.0), self.est(3, 1.1)]
        self.assertEqual(self.order(estimates, cached=[3], failed=[1]), [3, 2, 1])

    def test_all_failed_keeps_the_order(self):
        self.assertEqual(self.order([self.est(1, 1.0), self.est(2, 2.0)], failed=[1, 2]), [1, 2])


class TestVastReplies(unittest.TestCase):
    """La API a veces responde vacío: una consulta se repite; crear un pod, no."""

    def provider(self, replies):
        p, calls = VastProvider(), []

        def cli(*args):
            calls.append(args)
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        p._cli, p.calls = cli, calls
        return p

    def setUp(self):
        patcher = mock.patch("splatsim.cloud.providers.READ_RETRY_S", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_query_survives_an_empty_reply_and_an_error(self):
        p = self.provider(["", RuntimeError("429"), '{"credit": 3.5}'])
        self.assertEqual(p.balance(), 3.5)
        self.assertEqual(len(p.calls), 3)

    def test_a_query_gives_up_with_a_clear_error(self):
        p = self.provider(["", "", ""])
        with self.assertRaisesRegex(RuntimeError, "vastai show user"):
            p.balance()

    def test_create_is_not_repeated(self):
        p = self.provider(["", '{"success": true, "new_contract": 7}'])
        offer = Offer(1, 1, "g", "c", 16, 64, 50, 0.5, 0, 0, 500, 500, 0.99)
        with self.assertRaisesRegex(RuntimeError, "respuesta inválida"):
            p.create(offer, "img", ["k"], {}, 40, "l")
        self.assertEqual(len(p.calls), 1)

    def test_destroy_is_repeated(self):
        p = self.provider([RuntimeError("sin respuesta"), ""])
        p.destroy(5)
        self.assertEqual(len(p.calls), 2)

    def test_a_cli_timeout_is_a_normal_error(self):
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("vastai", 120)):
            with self.assertRaisesRegex(RuntimeError, "sin respuesta"):
                VastProvider()._cli("show", "user")


class TestEstimate(unittest.TestCase):
    CFG = {"training_steps": 50000, "batch_size": 32, "save_freq": 10000, "cameras": ["a", "b"],
           "image_augmentation": False, "torch_compile": True}

    def test_matches_the_measured_run(self):
        work = train_workload(self.CFG, dataset_gb=0.77)
        e = estimate([{"kind": "train", "profile": work.profile, "gpu": "RTX 5090", "cpu": CPU, "rate": 6.30}],
                     work, offer(dph=0.793))
        self.assertEqual(e.confidence, "alta")
        self.assertAlmostEqual(50000 / 6.30 * 0.793 / 3600, 1.75, places=2)
        self.assertGreater(e.total, 1.75)
        self.assertAlmostEqual(e.transfer_cost, (4.82 + 0.77) * 0.01 + 6 * 1.1 * 0.02, places=6)

    def test_other_cpu_uses_slowest_known_and_says_so(self):
        work = train_workload(self.CFG, 1)
        history = [{"kind": "train", "profile": work.profile, "gpu": "RTX 5090", "cpu": CPU, "rate": 6.3},
                   {"kind": "train", "profile": work.profile, "gpu": "RTX 5090", "cpu": "slow", "rate": 4.4}]
        self.assertEqual(lookup_rate(history, work, offer(cpu="otra"))[:2], (4.4, "baja"))

    def test_conversion_speed_ignores_the_gpu_and_startup_is_shorter(self):
        work = convert_workload(1000, 2, 24.0)
        history = [{"kind": "convert", "profile": work.profile, "gpu": "RTX 3060", "cpu": "otra", "rate": 10.0}]
        e = estimate(history, work, offer())
        self.assertEqual((e.rate, e.confidence, e.source), (10.0, "baja", "otra CPU"))
        self.assertLess(work.startup_s, train_workload(self.CFG, 1).startup_s)
        self.assertAlmostEqual(work.up_gb, 1000 * 2 * 14.6e-6)

    def test_no_data_gives_no_number(self):
        work = train_workload({**self.CFG, "batch_size": 8}, 1)
        e = estimate([], work, offer())
        self.assertIsNone(e.total)
        self.assertEqual(e.confidence, "sin datos")

    def test_rank_prefers_reliable_over_cheap_guess(self):
        work = train_workload(self.CFG, 1)
        history = [{"kind": "train", "profile": work.profile, "gpu": "RTX 5090", "cpu": CPU, "rate": 6.3}]
        ranked = rank(history, work, [offer(1, cpu="otra", dph=0.3), offer(2, dph=0.8)])
        self.assertEqual([e.offer.id for e in ranked], [2, 1])

    def test_profile_with_arrangement(self):
        self.assertEqual(train_profile({"batch_size": 16, "arrangement": "3cam"}), "bs16-cam3-noaug-nocompile")


class TestConfig(unittest.TestCase):
    BASE = {"output_root": "~/o", "push_to_hub": True, "wandb_mode": "online", "batch_size": 32, "dataset_id": "x",
            "dataset_root": "y", "name": "", "model_name": None}

    def test_paths_and_safe_defaults(self):
        cfg = train_config(self.BASE, "mi modelo", "~/lerobot/splatsim_test_data/ds1", {"batch_size": 8}, False)
        self.assertEqual(cfg["dataset_root"], "~/lerobot/splatsim_test_data/ds1")
        self.assertEqual(cfg["dataset_id"], "local/ds1")
        self.assertFalse(cfg["push_to_hub"])
        self.assertEqual(cfg["wandb_mode"], "offline")
        self.assertEqual(cfg["batch_size"], 8)
        self.assertTrue(cfg["model_name"].startswith("mi_modelo_"))

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValueError):
            train_config(self.BASE, "m", "ds", {"batchsize": 8}, False)


class TestVastCli(unittest.TestCase):
    def test_raw_goes_before_args(self):
        provider = VastProvider()
        calls = []

        class Done:
            returncode, stderr = 0, ""
            stdout = json.dumps({"success": True, "new_contract": 5})

        import splatsim.cloud.providers as module
        original = module.subprocess.run
        module.subprocess.run = lambda command, **_: (calls.append(command), Done)[1]
        self.addCleanup(setattr, module.subprocess, "run", original)
        self.assertEqual(provider.create(offer(), "img", ["runs/a"], {"A": "1"}, 40, "splatsim-a"), 5)
        command = calls[0]
        self.assertEqual(command[-2:], ["--args", "runs/a"])
        self.assertIn("--raw", command)
        self.assertLess(command.index("--raw"), command.index("--args"))


if __name__ == "__main__":
    unittest.main()
