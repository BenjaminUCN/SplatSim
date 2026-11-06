"""Panel del orquestador en la nube: cada trabajo levanta su propio pod, en paralelo. Convierte
grabaciones a LeRobot y entrena; un entrenamiento puede esperar a una conversión en cola.

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
from splatsim.cloud.orchestrator import ACTIVE, DATASETS_ROOT, DEFAULT_IMAGE, Orchestrator, train_config  # noqa: E402
from splatsim.cloud.providers import Requirements, VastProvider  # noqa: E402
from splatsim.cloud.storage import SECRETS_FILE, Storage, load_secrets, storage_env  # noqa: E402
from splatsim.utils.viser_page import set_page_title  # noqa: E402

TICK_PERIOD = 20.0
REFRESH_PERIOD = 3.0
OFFERS_SHOWN = 8
AUTO = "Auto (best estimate)"
RECORDINGS_ROOT = "~/data/bc_data"
CONVERT_REQUIREMENTS = {"min_cores": 8, "min_ram_gb": 16}   # sin GPU; medido: usa ~1.5 núcleos, más no acelera
PENDING_DATASET = "⏳ #"


@dataclass
class Args:
    queue_dir: str = "~/data/cloud_queue"
    port: int = 8083
    image: str = DEFAULT_IMAGE
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
        with tabs.add_tab("Convert", viser.Icon.TRANSFORM):
            self._build_convert()
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
        self.detail = gui.add_markdown("")

    def _selected_id(self) -> int | None:
        value = self.selected.value
        return int(value.split(" ", 1)[0]) if value != "—" else None

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
        rows = ["| Offer | CPU | $/h | PCIe | Speed | Hours | Total | Confidence |", "|---|---|---|---|---|---|---|---|"]
        for e in ranked[:OFFERS_SHOWN]:
            o = e.offer
            rows.append(f"| {o.id} | {o.cpu_name[:22]} ({o.cores:.0f}) | {o.dph:.3f} | {o.pcie_gbps:.0f} | "
                        f"{f'{e.rate:.2f}/s' if e.rate else '—'} | {f'{e.hours:.2f}' if e.hours else '—'} | "
                        f"{money(e.total)} | {e.confidence} |")
        table.content = intro + "\n".join(rows)
        offer.options = (AUTO, *(str(e.offer.id) for e in ranked[:OFFERS_SHOWN]))
        offer.value = AUTO
        known = ranked[0].total
        max_cost.value = math.ceil(known * 1.5 * 10) / 10 if known else 0.0
        queue.disabled = False

    # ── convertir ────────────────────────────────────────────────────────────
    @staticmethod
    def _recordings() -> tuple[str, ...]:
        """Carpetas con imágenes: las que solo tienen el estado aún hay que renderizarlas."""
        root = Path(RECORDINGS_ROOT).expanduser()
        names = []
        for folder in sorted(root.iterdir()) if root.exists() else []:
            episode = next((p for p in folder.iterdir() if p.is_dir()), None) if folder.is_dir() else None
            if episode and any((episode / d).is_dir() for d in ("image_1", "images")):
                names.append(folder.name)
        return tuple(names) or ("—",)

    def _build_convert(self):
        gui = self.v.gui
        self.c_source = gui.add_dropdown("Recordings", self._recordings())
        self.c_fps = gui.add_number("FPS", 10, min=1, step=1)
        self.c_cameras = gui.add_text("Cameras", "", hint="Space separated; empty = all that the recordings have")
        self.c_suffix = gui.add_text("Name suffix", datetime.now().strftime("%y%m%d_%H%M"))
        self.c_discarded = gui.add_checkbox("Include discarded", False)
        gui.add_button("Estimate cost").on_click(lambda _: self._estimate_convert())
        self.c_offers = gui.add_markdown("Press *Estimate cost* to see what has to be uploaded and the machines.")
        self.c_offer = gui.add_dropdown("Machine offer", (AUTO,))
        self.c_max = gui.add_number("Max cost ($)", 0.0, min=0.0, step=0.1,
                                    hint="The pod stops at this cost, even if this PC is off")
        self.c_queue = gui.add_button("Add to queue", disabled=True)
        self.c_queue.on_click(lambda _: self._submit_convert())

    def _convert_args(self) -> tuple[dict, Requirements]:
        suffix = self.c_suffix.value.strip()
        if not suffix or self.c_source.value == "—":
            raise ValueError("Faltan las grabaciones o el sufijo")
        cameras = self.c_cameras.value.split() or None
        args = {"input": f"{RECORDINGS_ROOT}/{self.c_source.value}", "fps": int(self.c_fps.value), "suffix": suffix,
                "cameras": cameras, "ignore_selection": self.c_discarded.value}
        return args, Requirements(gpu_name="", **CONVERT_REQUIREMENTS)

    def _estimate_convert(self):
        try:
            args, req = self._convert_args()
            self.c_offers.content = "Reading the recordings and searching…"
            scan, ranked = self.orch.estimate_convert(args, req)
        except Exception as e:  # noqa: BLE001  se muestra en el panel
            self.c_offers.content = f"**Error:** {e}"
            self.c_queue.disabled = True
            return
        intro = (f"{scan['episodes']} episodes · {scan['frames']:,} frames · {' '.join(scan['cameras'])} · "
                 f"**{scan['gb']:.1f} GB to upload**\n\n")
        self._show_offers(ranked, self.c_offers, self.c_offer, self.c_max, self.c_queue, intro)

    def _submit_convert(self):
        try:
            args, req = self._convert_args()
            offer_id = None if self.c_offer.value == AUTO else int(self.c_offer.value)
            name = f"convert {self.c_source.value} {args['suffix']}"
            self.orch.submit_convert(name, args, req, float(self.c_max.value), offer_id)
        except Exception as e:  # noqa: BLE001  se muestra en el panel
            self.c_offers.content = f"**Error:** {e}"
            return
        self.c_queue.disabled = True
        self.c_offers.content = f"Queued. Its dataset now shows in *New job → Dataset* as {PENDING_DATASET}…"
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
                        image=args.image, on_ready=update_registry)
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
