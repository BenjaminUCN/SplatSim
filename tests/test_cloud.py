"""Pruebas del orquestador sin nube: un proveedor simulado corre el runner real en un hilo, con el
bucket en una carpeta local.

    PYTHONPATH=. python tests/test_cloud.py
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import asdict
from pathlib import Path

os.environ["SPLATSIM_HEARTBEAT_S"] = "0.05"
os.environ["SPLATSIM_RESULT_RETRY_S"] = "0.01"
os.environ["SPLATSIM_WARMUP_S"] = "0"

import yaml  # noqa: E402

from splatsim.cloud.estimate import estimate, lookup_rate, rank, train_profile, train_workload  # noqa: E402
from splatsim.cloud.orchestrator import Orchestrator, Settings, train_config  # noqa: E402
from splatsim.cloud.providers import Instance, Offer, Requirements, VastProvider  # noqa: E402
from splatsim.cloud import runner  # noqa: E402
from splatsim.cloud.runner import Run, run_stage  # noqa: E402

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


class FolderStorage:
    """Misma interfaz que Storage, sobre una carpeta."""

    def __init__(self, root: Path):
        self.root, self.remote = Path(root), str(root)
        self.corrupt = False

    def read(self, key):
        path = self.root / key
        return path.read_text() if path.exists() else None

    def write(self, key, text):
        (self.root / key).parent.mkdir(parents=True, exist_ok=True)
        tmp = (self.root / key).with_name(f".{threading.get_ident()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, self.root / key)

    def upload(self, local, key):
        shutil.copytree(local, self.root / key, dirs_exist_ok=True)

    def upload_file(self, local, key):
        (self.root / key).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, self.root / key)

    def download(self, key, local):
        shutil.copytree(self.root / key, local, dirs_exist_ok=True)
        if self.corrupt:
            (Path(local) / "model.safetensors").write_text("dañado")


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
        return [o for o in self.offers if o.gpu_name == req.gpu_name]

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
        self.provider = FakeProvider(self.storage)
        self.orch = self.make()

    def make(self):
        return Orchestrator(self.tmp / "queue", self.provider, self.storage, health=None, python=sys.executable,
                            train_script=str(self.script))

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
