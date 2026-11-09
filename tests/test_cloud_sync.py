"""Cola compartida por R2 entre paneles (splatsim/cloud/sync.py), con el bucket en una carpeta local.

    PYTHONPATH=. python tests/test_cloud_sync.py
"""
import os
import shutil
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from test_cloud import Base, FakeProvider  # noqa: E402

from splatsim.cloud.orchestrator import Orchestrator  # noqa: E402
from splatsim.cloud.providers import Requirements  # noqa: E402
from splatsim.cloud.sync import CONTROL, LEASE_KEY, STATE_KEY, VIEW, QueueSync  # noqa: E402


class SyncBase(Base):
    def setUp(self):
        super().setUp()
        self.provider_b = FakeProvider(self.storage)
        self.orch_b = self.make_at("queue_b", self.provider_b)

    def make_at(self, folder, provider):
        return Orchestrator(self.tmp / folder, provider, self.storage, health=None, python=sys.executable,
                            scripts={"train": str(self.script), "convert": str(self.convert_script)},
                            pod_root=str(self.tmp / "pod"))

    def panel(self, orch, name, **kwargs):
        sync = QueueSync(orch, name, settle_s=0, **kwargs)
        sync.start()
        return sync

    def cycle_until(self, sync, condition, timeout=20.0):
        end = time.time() + timeout
        while time.time() < end:
            sync.cycle()
            if condition():
                return
            time.sleep(0.03)
        self.fail(f"tiempo agotado; trabajos: {[(j['id'], j['status'], j.get('wait')) for j in sync.orch.jobs]}")


class TestControl(SyncBase):
    def test_first_panel_controls_and_second_only_watches(self):
        a = self.panel(self.orch, "lab")
        self.submit()
        a.cycle()
        b = self.panel(self.orch_b, "laptop")
        self.assertEqual((a.role, b.role), (CONTROL, VIEW))
        self.assertEqual([j["name"] for j in self.orch_b.jobs], ["m"], "el que mira ve la cola del otro")
        self.assertTrue(self.orch_b.read_only)
        b.cycle()
        self.assertEqual(self.provider_b.created, [], "el que mira no lanza pods")

    def test_viewer_follows_changes(self):
        a = self.panel(self.orch, "lab")
        b = self.panel(self.orch_b, "laptop")
        job = self.submit()
        self.cycle_until(a, lambda: job["status"] == "done")
        b.cycle()
        self.assertEqual(self.orch_b.get(job["id"])["status"], "done")

    def test_handover_waits_for_the_holder_and_moves_the_queue(self):
        a = self.panel(self.orch, "lab")
        job = self.submit()
        self.cycle_until(a, lambda: job["status"] == "done")
        b = self.panel(self.orch_b, "laptop")
        b.request_control()
        self.assertEqual(b.role, VIEW, "con el permiso vigente, solo lo pide")
        a.cycle()                                  # ve el pedido, sube la cola y suelta
        self.assertEqual(a.role, VIEW)
        b.cycle()
        self.assertEqual((a.role, b.role), (VIEW, CONTROL))
        self.assertFalse(self.orch_b.read_only)
        self.assertTrue(self.orch.read_only)
        job_b = self.orch_b.submit("n", self.cfg("n"), str(self.dataset), Requirements(), 1.0)
        self.cycle_until(b, lambda: job_b["status"] == "done")
        self.assertEqual(len(self.provider_b.created), 1, "ahora lanza el nuevo panel")
        self.assertEqual(job_b["id"], job["id"] + 1, "sigue la numeración de la cola compartida")

    def test_holder_finishes_its_transfers_before_handing_over(self):
        a = self.panel(self.orch, "lab")
        b = self.panel(self.orch_b, "laptop")
        b.request_control()
        self.orch.busy.add(99)
        a.cycle()
        a.cycle()
        self.assertEqual(self.storage_lease()["holder"], "lab", "con una subida en curso no suelta")
        self.orch.busy.discard(99)
        a.cycle()
        b.cycle()
        self.assertEqual(b.role, CONTROL)

    def test_dead_panel_lease_expires(self):
        self.panel(self.orch, "lab", ttl_s=0.2)      # se cierra sin soltar
        b = self.panel(self.orch_b, "laptop")
        self.assertEqual(b.role, VIEW)
        time.sleep(0.3)
        b.request_control()
        self.assertEqual(b.role, CONTROL)

    def test_same_pc_reopened_takes_back_its_own_lease(self):
        self.panel(self.orch, "lab")
        again = self.panel(self.make_at("queue", self.provider), "lab")
        self.assertEqual(again.role, CONTROL)

    def test_shutdown_frees_the_lease(self):
        a = self.panel(self.orch, "lab")
        b = self.panel(self.orch_b, "laptop")
        a.shutdown()
        b.request_control()
        self.assertEqual(b.role, CONTROL)

    def test_viewer_lost_control_stops_ticking(self):
        a = self.panel(self.orch, "lab")
        self.storage.write(LEASE_KEY, yaml.safe_dump({"holder": "laptop", "expires": time.time() + 60}))
        a.cycle()
        self.assertEqual(a.role, VIEW)

    def storage_lease(self):
        return yaml.safe_load(self.storage.read(LEASE_KEY))


class TestState(SyncBase):
    def test_paths_travel_with_tilde(self):
        home = self.tmp / "home"
        (home / "ds").mkdir(parents=True)
        (home / "ds" / "data.bin").write_text("dataset")
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            a = self.panel(self.orch, "lab")
            self.orch.submit("m", self.cfg("m"), str(home / "ds"), Requirements(), 1.0)
            a.cycle()
            self.assertEqual(yaml.safe_load(self.storage.read(STATE_KEY))["jobs"][0]["dataset"], "~/ds")
        with mock.patch.dict(os.environ, {"HOME": "/home/otro"}):
            self.panel(self.orch_b, "laptop")
            self.assertEqual(self.orch_b.jobs[0]["dataset"], "/home/otro/ds")

    def test_done_jobs_travel_without_their_hashes(self):
        a = self.panel(self.orch, "lab")
        job = self.submit()
        self.cycle_until(a, lambda: job["status"] == "done")
        self.assertIn("outputs", job["result"], "en este PC se conservan")
        remote = yaml.safe_load(self.storage.read(STATE_KEY))["jobs"][0]
        self.assertNotIn("outputs", remote["result"])
        self.assertTrue(remote["result"]["ok"])

    def test_unchanged_queue_is_not_uploaded_again(self):
        a = self.panel(self.orch, "lab")
        rev = a.rev
        a.cycle()
        a.cycle()
        self.assertEqual(a.rev, rev)

    def test_local_changes_that_never_reached_r2_are_kept_aside(self):
        a = self.panel(self.orch, "lab")
        self.submit()
        a.cycle()
        a.shutdown()
        b = self.panel(self.orch_b, "laptop")
        self.orch_b.submit("n", self.cfg("n"), str(self.dataset), Requirements(), 1.0)
        b.cycle()
        b.shutdown()
        self.orch.read_only = False
        self.orch.submit("offline", self.cfg("offline"), str(self.dataset), Requirements(), 1.0)  # sin subir
        again = self.panel(self.make_at("queue", self.provider), "lab")
        self.assertEqual([j["name"] for j in again.orch.jobs], ["m", "n"], "manda la cola de R2")
        self.assertTrue(list((self.tmp / "queue").glob("jobs.yaml.bak*")))

    def test_first_sync_keeps_the_queue_with_jobs_over_an_empty_one(self):
        self.submit()                                   # cola del lab, de antes de compartirla
        b = self.panel(self.orch_b, "laptop")           # el laptop abre primero: sube una cola vacía
        self.assertEqual(b.role, CONTROL)
        a = self.panel(self.orch, "lab")
        self.assertEqual(a.role, CONTROL)
        self.assertEqual([j["name"] for j in self.orch.jobs], ["m"])
        b.cycle()
        self.assertEqual(b.role, VIEW, "el laptop ve que otro tomó el control")
        self.assertEqual([j["name"] for j in self.orch_b.jobs], ["m"])

    def test_viewer_keeps_aside_a_queue_it_never_uploaded(self):
        a = self.panel(self.orch, "lab")
        self.submit()
        a.cycle()
        other = self.make_at("queue_c", FakeProvider(self.storage))
        other.submit("solo_aqui", self.cfg("solo_aqui"), str(self.dataset), Requirements(), 1.0)
        self.panel(other, "otro")
        self.assertEqual([j["name"] for j in other.jobs], ["m"])
        self.assertTrue(list((self.tmp / "queue_c").glob("jobs.yaml.bak*")))


class TestKeepOutput(Base):
    def test_output_stays_in_the_bucket_until_downloaded(self):
        job = self.orch.submit("m", self.cfg("m"), str(self.dataset), Requirements(), 1.0, keep_output=True)
        self.run_until(lambda: job["status"] == "done")
        target = self.tmp / "models" / "m"
        self.assertNotIn("output_dir", job, "no baja")
        self.assertTrue((self.storage.root / job["remote_output"]).is_dir(), "no se borra del bucket")
        shutil.rmtree(target, ignore_errors=True)     # el pod simulado corre en este disco y escribió ahí
        history = len(self.orch.history)
        self.assertTrue(self.orch.download(job["id"]))
        self.run_until(lambda: job["status"] == "done" and target.exists())
        self.assertEqual((target / "model.safetensors").read_text(), "weights")
        self.assertNotIn("remote_output", job)
        self.assertEqual(len(self.orch.history), history, "el pod corrió una sola vez")
        self.assertEqual(len(self.provider.created), 1)

    def test_missing_input_waits_instead_of_failing(self):
        job = self.submit()
        (self.dataset / "data.bin").unlink()
        self.dataset.rmdir()
        self.orch.tick()
        self.run_until(lambda: "falta en este PC" in (job.get("wait") or ""))
        self.assertEqual(job["status"], "pending")
        self.assertEqual(self.provider.created, [])


if __name__ == "__main__":
    unittest.main()
