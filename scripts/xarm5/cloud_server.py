"""Panel del orquestador en la nube: cada trabajo levanta su propio pod, en paralelo. Convierte
grabaciones, las renderiza y espeja sin pasar por este PC, y entrena; cada etapa espera a la anterior.

    python scripts/xarm5/cloud_server.py            # http://localhost:8083

Muestra el costo estimado antes de encolar y permite fijar el máximo de cada trabajo. La cola vive
en ~/data/cloud_queue/, así que el panel se puede cerrar: los pods siguen y al volver se retoman.
No reemplaza a train_server.py, que entrena en este PC.

Necesita ~/.config/splatsim/cloud.env (configs/cloud.env.example), rclone y la CLI de vastai.
"""
import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro
import viser
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.xarm5.data_jobs import plan_pipeline  # noqa: E402
from splatsim.cloud.orchestrator import (ACTIVE, DATASETS_ROOT, DEFAULT_IMAGE, DEFAULT_RENDER_IMAGE,  # noqa: E402
                                         Orchestrator, train_config)
from splatsim.cloud.providers import Requirements, VastProvider  # noqa: E402
from splatsim.cloud.storage import SECRETS_FILE, Storage, load_secrets, storage_env  # noqa: E402
from splatsim.utils.viser_page import set_page_title  # noqa: E402

TICK_PERIOD = 20.0
REFRESH_PERIOD = 3.0
OFFERS_SHOWN = 8
AUTO = "Auto (best estimate)"
RECORDINGS_ROOT = "~/data/bc_data"
LEAF_DIR = "~/data/occluders/xarm5_lite_plant_v3"
CONVERT_REQUIREMENTS = {"min_cores": 8, "min_ram_gb": 16}   # sin GPU; medido: usa ~1.5 núcleos, más no acelera
PENDING_DATASET = "⏳ #"


@dataclass
class Args:
    queue_dir: str = "~/data/cloud_queue"
    port: int = 8083
    image: str = DEFAULT_IMAGE
    render_image: str = DEFAULT_RENDER_IMAGE
    secrets: str = SECRETS_FILE


def money(value) -> str:
    return "—" if value is None else f"${value:.2f}"


class Panel:
    def __init__(self, orch: Orchestrator, port: int, has_wandb_key: bool):
        self.orch, self.has_wandb_key = orch, has_wandb_key
        self.v = viser.ViserServer(port=port, label="Cloud queue", verbose=False)
        set_page_title(self.v, "Cloud queue")
        tabs = self.v.gui.add_tab_group()
        with tabs.add_tab("Queue", viser.Icon.LIST):
            self._build_queue()
        with tabs.add_tab("New job", viser.Icon.PLUS):
            self._build_new_job()
        with tabs.add_tab("Data", viser.Icon.DATABASE):
            self._build_data()
        with tabs.add_tab("Limits", viser.Icon.COIN):
            self._build_limits()

    # ── cola ─────────────────────────────────────────────────────────────────
    def _build_queue(self):
        gui = self.v.gui
        self.summary = gui.add_markdown("")
        self.table = gui.add_markdown("")
        self.selected = gui.add_dropdown("Job", ("—",))
        gui.add_button("Stop (deletes the pod)", color="red").on_click(lambda _: self._act(self.orch.stop))
        gui.add_button("Retry").on_click(lambda _: self._act(self.orch.retry))
        gui.add_button("Remove").on_click(lambda _: self._act(self.orch.remove))
        self.new_max = gui.add_number("Max cost of the selected job ($)", 0.0, min=0.0, step=0.5,
                                      hint="Only without a pod running. A failed job then needs Retry.")
        gui.add_button("Set max cost").on_click(lambda _: self._set_max_cost())
        self.max_note = gui.add_markdown("")
        self.selected.on_update(lambda _: self._show_max_cost())
        self.detail = gui.add_markdown("")

    def _selected_id(self) -> int | None:
        value = self.selected.value
        return int(value.split(" ", 1)[0]) if value != "—" else None

    def _show_max_cost(self):
        job = self.orch.get(self._selected_id()) if self._selected_id() is not None else None
        self.new_max.value = job["max_cost"] if job else 0.0
        self.max_note.content = ""

    def _set_max_cost(self):
        job_id = self._selected_id()
        if job_id is None:
            return
        changed = self.orch.set_max_cost(job_id, self.new_max.value)
        self.max_note.content = (f"Max cost of job {job_id}: {money(self.new_max.value)}." if changed else
                                 "Not changed: it must be above 0, and the job can't have a pod or be done.")
        self.refresh()

    def _act(self, action):
        if (job_id := self._selected_id()) is not None:
            action(job_id)
            self.refresh()

    @staticmethod
    def _progress(job: dict) -> str:
        p = job.get("progress") or {}
        if "done" not in p:
            return job.get("phase") or job.get("wait") or ""
        total = p.get("total") or (job.get("config") or {}).get("training_steps") or "?"
        rate = f" · {p['rate']:.2f}/s" if p.get("rate") else ""
        return f"{p['done']:,}/{total:,}{rate}" if isinstance(total, int) else f"{p['done']:,}/?{rate}"

    def refresh(self):
        orch = self.orch
        with orch.lock:
            rows = ["| # | Job | Status | Progress | Machine | Est. | Cost | Max |", "|---|---|---|---|---|---|---|---|"]
            for j in orch.jobs:
                offer = j.get("offer") or {}
                rows.append(f"| {j['id']} | {j['name']} | {j['status']} | {self._progress(j)} | "
                            f"{offer.get('cpu_name', '')[:24]} | {money((j.get('estimate') or {}).get('total'))} | "
                            f"{money(j.get('cost'))} | {money(j['max_cost'])} |")
            self.table.content = "\n".join(rows) if orch.jobs else "No jobs yet."
            active = sum(j["status"] in ACTIVE for j in orch.jobs)
            warnings = "".join(f"\n\n⚠️ {w}" for w in orch.warnings[-3:])
            self.summary.content = (f"**{active}/{orch.settings.max_pods} pods** · spent {money(orch.spent())} · "
                                    f"committed {money(orch.committed())} · budget {money(orch.settings.budget_usd)}"
                                    f"{' · **paused**' if orch.settings.paused else ''}{warnings}")
            options = tuple(f"{j['id']} {j['name']}" for j in orch.jobs) or ("—",)
            if tuple(self.selected.options) != options:
                self.selected.options = options
            datasets = self._datasets()
            if tuple(self.f_dataset.options) != datasets:
                self.f_dataset.options = datasets
            job = orch.get(self._selected_id()) if self._selected_id() is not None else None
            self.detail.content = self._detail(job) if job else ""

    def _detail(self, job: dict) -> str:
        lines = []
        if job.get("error"):
            lines.append(f"**Error:** {job['error']}")
        if job.get("wait"):
            lines.append(f"**Waiting:** {job['wait']}")
        if job.get("estimate"):
            e = job["estimate"]
            lines.append(f"Estimate: {money(e['total'])} ({e['confidence']}: {e['source']})")
        if job.get("boot_s") is not None:
            lines.append(f"Pod started in {job['boot_s'] / 60:.1f} min")
        if job.get("output_dir"):
            lines.append(f"Output: `{job['output_dir']}`")
        if job.get("run_key"):
            log = self.orch.storage.read(f"{job['run_key']}/run.log") or ""
            tail = "\n".join(line[:150] for line in log.splitlines()[-12:])
            if tail:
                lines.append(f"```\n{tail}\n```")
        return "\n\n".join(lines)

    # ── nuevo trabajo ────────────────────────────────────────────────────────
    def _datasets(self) -> tuple[str, ...]:
        root = Path(DATASETS_ROOT).expanduser()
        names = sorted((p.name for p in root.iterdir() if (p / "meta").is_dir()), reverse=True) if root.exists() else []
        pending = [f"{PENDING_DATASET}{j['id']} {Path(j['output']).name}" for j in self.orch.jobs
                   if j["kind"] == "convert" and j["status"] != "done"]
        return (*pending, *names) or ("—",)

    def _chosen_dataset(self) -> tuple[str, int | None]:
        """Ruta del dataset y, si es el de una conversión en cola, el trabajo al que debe esperar."""
        value = self.f_dataset.value
        if not value.startswith(PENDING_DATASET):
            return f"{DATASETS_ROOT}/{value}", None
        job_id, name = value.removeprefix(PENDING_DATASET).split(" ", 1)
        return f"{DATASETS_ROOT}/{name}", int(job_id)

    def _build_new_job(self):
        gui = self.v.gui
        configs = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs/train").glob("*.yaml")
                         if p.name != "queue_example.yaml")
        self.f_name = gui.add_text("Name", "")
        self.f_base = gui.add_dropdown("Base config", tuple(configs))
        self.f_dataset = gui.add_dropdown("Dataset", self._datasets())
        self.f_arrangement = gui.add_text("Arrangement", "2cam")
        self.f_steps = gui.add_number("Training steps", 50000, min=1, step=1000)
        self.f_batch = gui.add_number("Batch size", 32, min=1, step=1)
        self.f_aug = gui.add_checkbox("Image augmentation", False)
        self.f_compile = gui.add_checkbox("torch.compile", True)
        self.f_extra = gui.add_text("Extra (YAML)", "", hint="Other config fields, e.g. {horizon: 32}")
        with gui.add_folder("Machine"):
            self.f_gpu = gui.add_text("GPU", "RTX 5090")
            self.f_cores = gui.add_number("Min CPU cores", 16, min=1, step=1)
            self.f_ram = gui.add_number("Min RAM (GB)", 48, min=8, step=8)
            self.f_pcie = gui.add_number("Min PCIe (GB/s)", 0.0, min=0.0, step=1.0)
        gui.add_button("Estimate cost").on_click(lambda _: self._estimate())
        self.f_offers = gui.add_markdown("Press *Estimate cost* to see the available machines.")
        self.f_offer = gui.add_dropdown("Machine offer", (AUTO,))
        self.f_max = gui.add_number("Max cost ($)", 0.0, min=0.0, step=0.5,
                                    hint="The pod stops at this cost, even if this PC is off")
        self.f_queue = gui.add_button("Add to queue", disabled=True)
        self.f_queue.on_click(lambda _: self._submit())

    def _form(self) -> tuple[dict, str, Requirements, int | None]:
        base = yaml.safe_load((ROOT / self.f_base.value).read_text()) or {}
        extra = yaml.safe_load(self.f_extra.value) if self.f_extra.value.strip() else {}
        name = self.f_name.value.strip()
        if not name:
            raise ValueError("Falta el nombre")
        overrides = {"arrangement": self.f_arrangement.value.strip() or None, "training_steps": int(self.f_steps.value),
                     "batch_size": int(self.f_batch.value), "image_augmentation": self.f_aug.value,
                     "torch_compile": self.f_compile.value, **(extra or {})}
        dataset, parent = self._chosen_dataset()
        req = Requirements(gpu_name=self.f_gpu.value.strip(), min_cores=int(self.f_cores.value),
                           min_ram_gb=int(self.f_ram.value), min_pcie_gbps=float(self.f_pcie.value))
        return train_config(base, name, dataset, overrides, self.has_wandb_key), dataset, req, parent

    def _estimate(self):
        try:
            cfg, dataset, req, parent = self._form()
            self.f_offers.content = "Searching…"
            ranked = self.orch.estimate(cfg, dataset, req, parent)
        except Exception as e:  # noqa: BLE001  se muestra en el panel
            self.f_offers.content = f"**Error:** {e}"
            return
        self._show_offers(ranked, self.f_offers, self.f_offer, self.f_max, self.f_queue)

    def _submit(self):
        try:
            cfg, dataset, req, parent = self._form()
            offer_id = None if self.f_offer.value == AUTO else int(self.f_offer.value)
            self.orch.submit(self.f_name.value.strip(), cfg, dataset, req, float(self.f_max.value), offer_id,
                             depends_on=parent)
        except Exception as e:  # noqa: BLE001  se muestra en el panel
            self.f_offers.content = f"**Error:** {e}"
            return
        self.f_queue.disabled = True
        self.f_offers.content = f"Queued **{cfg['model_name']}**."
        self.refresh()

    @staticmethod
    def _show_offers(ranked, table, offer, max_cost, queue, intro: str = "") -> None:
        if not ranked:
            table.content = intro + "No machines match these requirements right now."
            queue.disabled = True
            return
        rows = ["| Offer | CPU | $/h | PCIe | Speed | Hours | Total | Confidence | Image |", "|---|---|---|---|---|---|---|---|---|"]
        for e in ranked[:OFFERS_SHOWN]:
            o = e.offer
            rows.append(f"| {o.id} | {o.cpu_name[:22]} ({o.cores:.0f}) | {o.dph:.3f} | {o.pcie_gbps:.0f} | "
                        f"{f'{e.rate:.2f}/s' if e.rate else '—'} | {f'{e.hours:.2f}' if e.hours else '—'} | "
                        f"{money(e.total)} | {e.confidence} | {'cached' if e.cached else ''} |")
        table.content = intro + "\n".join(rows)
        offer.options = (AUTO, *(str(e.offer.id) for e in ranked[:OFFERS_SHOWN]))
        offer.value = AUTO
        known = ranked[0].total
        max_cost.value = math.ceil(known * 1.5 * 10) / 10 if known else 0.0
        queue.disabled = False

    # ── datos: render → mirror → convert ─────────────────────────────────────
    @staticmethod
    def _recordings() -> tuple[str, ...]:
        root = Path(RECORDINGS_ROOT).expanduser()
        names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
        return tuple(names) or ("—",)

    def _build_data(self):
        gui = self.v.gui
        self.d_source = gui.add_dropdown("Recordings", self._recordings())
        self.d_render = gui.add_checkbox("Render", True, hint="For state-only recordings")
        self.d_leaves = gui.add_checkbox("Random leaf occlusions", False, hint="Render: a random leaf per episode, fitted to a target")
        self.d_leaf_dir = gui.add_text("Leaf folder", LEAF_DIR, hint="From segment_leaf.py")
        self.d_targets = gui.add_text("Occlusion targets (%)", "0 25 50 75", hint="One is drawn per episode; 0 = no leaf")
        self.d_seed = gui.add_number("Leaf seed", 0, min=0, step=1, hint="Same seed, same draw per episode")
        self.d_mirror = gui.add_checkbox("Mirror", False)
        self.d_with_original = gui.add_checkbox("Keep originals", True, hint="Mirror: originals + mirrors, twice the frames")
        self.d_convert = gui.add_checkbox("Convert to LeRobot", True)
        self.d_fps = gui.add_number("FPS", 10, min=1, step=1)
        self.d_arrangement = gui.add_text("Arrangement", "2cam", hint="From configs/camera_poses.yaml; empty = all cameras")
        self.d_workers = gui.add_number("Workers", 3, min=1, step=1, hint="~8 GB of GPU memory each")
        self.d_suffix = gui.add_text("Dataset suffix", datetime.now().strftime("%y%m%d_%H%M"))
        self.d_discarded = gui.add_checkbox("Include discarded", False)
        self.d_gpu = gui.add_text("GPU", "RTX 5090", hint="Render and mirror; conversion uses any machine")
        gui.add_button("Estimate cost").on_click(lambda _: self._estimate_data())
        self.d_plan = gui.add_markdown("Press *Estimate cost* to see each stage and what it costs.")
        self.d_max = gui.add_number("Max cost per stage ($)", 0.0, min=0.0, step=0.1,
                                    hint="Each pod stops at this cost, even if this PC is off")
        self.d_queue = gui.add_button("Add to queue", disabled=True)
        self.d_queue.on_click(lambda _: self._submit_data())

    def _data_plan(self) -> tuple[list[dict], Requirements, Requirements]:
        suffix, arrangement = self.d_suffix.value.strip(), self.d_arrangement.value.strip() or None
        if self.d_source.value == "—" or (self.d_convert.value and not suffix):
            raise ValueError("Faltan las grabaciones o el sufijo")
        fps, workers = int(self.d_fps.value), int(self.d_workers.value)
        render = {"fps": fps, "arrangement": arrangement, "workers": workers} if self.d_render.value else None
        if render and self.d_leaves.value:
            leaf_dir = self.d_leaf_dir.value.strip()
            if not next(Path(leaf_dir).expanduser().glob("*.ply"), None):
                raise ValueError(f"No hay hojas en {leaf_dir}: recorta alguna con segment_leaf.py")
            render.update(leaf_dir=leaf_dir, occlusion_targets=[float(t) for t in self.d_targets.value.split()],
                          seed=int(self.d_seed.value))
        mirror =({"with_original": self.d_with_original.value, "arrangement": arrangement, "workers": workers}
                  if self.d_mirror.value else None)
        convert = ({"output_dir": DATASETS_ROOT, "fps": fps, "suffix": suffix, "cameras": None,
                    "ignore_selection": self.d_discarded.value, "push": False} if self.d_convert.value else None)
        if not (render or mirror or convert):
            raise ValueError("Elige al menos una etapa")
        plan = plan_pipeline(f"{RECORDINGS_ROOT}/{self.d_source.value}", render, mirror, convert)
        return plan, Requirements(gpu_name=self.d_gpu.value.strip()), Requirements(gpu_name="", **CONVERT_REQUIREMENTS)

    def _estimate_data(self):
        try:
            plan, gpu, cpu = self._data_plan()
            self.d_plan.content = "Reading the recordings and searching…"
            stages = self.orch.estimate_pipeline(plan, gpu, cpu)
        except Exception as e:  # noqa: BLE001  se muestra en el panel
            self.d_plan.content = f"**Error:** {e}"
            self.d_queue.disabled = True
            return
        rows = ["| Stage | Frames | Cameras | Download GB | Upload GB | Best offer | Speed | Hours | Total | Confidence |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        totals = []
        for stage, ranked in stages:
            work = self.orch.workload(stage)
            best = ranked[0] if ranked else None
            rows.append(f"| {stage['kind']} | {stage['frames']:,} | {len(stage['cameras'])} | {work.down_gb - work.image_gb:.1f} | "
                        f"{work.up_gb:.1f} | {f'{best.offer.gpu_name} ${best.offer.dph:.2f}/h{" (image cached)" if best.cached else ""}' if best else 'none'} | "
                        f"{f'{best.rate:.1f}/s' if best and best.rate else '—'} | {f'{best.hours:.2f}' if best and best.hours else '—'} | "
                        f"{money(best.total) if best else '—'} | {best.confidence if best else '—'} |")
            totals.append(best.total if best else None)
        known = None not in totals
        self.d_plan.content = "\n".join(rows) + "\n\nOnly the last stage comes down to this PC; the rest stays in the bucket." + (
            "" if known else "\n\n**No speed measured for some stage: set the max cost yourself.**")
        # un máximo sacado solo de las otras etapas cortaría a medias la que no tiene estimación
        self.d_max.value = math.ceil(max(totals) * 1.5 * 10) / 10 if known else 0.0
        self.d_queue.disabled = False

    def _submit_data(self):
        try:
            plan, gpu, cpu = self._data_plan()
            self.orch.submit_pipeline(plan, gpu, cpu, float(self.d_max.value))
        except Exception as e:  # noqa: BLE001  se muestra en el panel
            self.d_plan.content = f"**Error:** {e}"
            return
        self.d_queue.disabled = True
        self.d_plan.content = f"Queued {len(plan)} stage(s). A queued dataset shows in *New job → Dataset* as {PENDING_DATASET}…"
        self.refresh()

    # ── límites ──────────────────────────────────────────────────────────────
    def _build_limits(self):
        gui, s = self.v.gui, self.orch.settings
        self.l_pods = gui.add_number("Max pods at once", s.max_pods, min=1, step=1)
        self.l_budget = gui.add_number("Queue budget ($)", s.budget_usd, min=0.0, step=1.0)
        self.l_paused = gui.add_checkbox("Paused (no new pods)", s.paused)

        def apply(_):
            with self.orch.lock:
                s.max_pods, s.budget_usd, s.paused = int(self.l_pods.value), float(self.l_budget.value), self.l_paused.value
                self.orch.save()

        for control in (self.l_pods, self.l_budget, self.l_paused):
            control.on_update(apply)
        self.l_balance = gui.add_markdown("")
        gui.add_button("Check balance").on_click(lambda _: self._show_balance())

    def _show_balance(self):
        try:
            self.l_balance.content = f"Vast balance: **{money(self.orch.provider.balance())}**"
        except Exception as e:  # noqa: BLE001  se muestra en el panel
            self.l_balance.content = f"**Error:** {e}"


def update_registry(_job: dict) -> None:
    """Modelos y datasets nuevos aparecen en el panel del robot tras *Reload Registry*."""
    subprocess.run([sys.executable, str(ROOT / "scripts/xarm5/build_registry.py")], cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(args: Args):
    secrets = load_secrets(args.secrets)
    env = storage_env(secrets)
    pod_env = dict(env)
    if "WANDB_API_KEY" in secrets:
        pod_env["WANDB_API_KEY"] = secrets["WANDB_API_KEY"]
    orch = Orchestrator(args.queue_dir, VastProvider(), Storage(env["SPLATSIM_REMOTE"], env), pod_env,
                        image=args.image, render_image=args.render_image, on_ready=update_registry)
    panel = Panel(orch, args.port, "WANDB_API_KEY" in secrets)

    def worker():
        while True:
            try:
                orch.tick()
            except Exception as e:  # noqa: BLE001  un error de red no debe parar el seguimiento
                print(f"[Cloud] error: {type(e).__name__}: {e}")
            time.sleep(TICK_PERIOD)

    threading.Thread(target=worker, daemon=True).start()
    print(f"[Cloud] panel en http://localhost:{args.port}")
    while True:
        panel.refresh()
        time.sleep(REFRESH_PERIOD)


if __name__ == "__main__":
    main(tyro.cli(Args))
