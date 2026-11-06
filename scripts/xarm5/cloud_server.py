"""Panel del orquestador en la nube: cada trabajo levanta su propio pod, en paralelo.

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
        total = p.get("total") or job["config"].get("training_steps")
        rate = f" · {p['rate']:.2f}/s" if p.get("rate") else ""
        return f"{p['done']:,}/{total:,}{rate}"

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
        if job.get("model_dir"):
            lines.append(f"Model: `{job['model_dir']}`")
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
        return tuple(names) or ("—",)

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

    def _form(self) -> tuple[dict, str, Requirements]:
        base = yaml.safe_load((ROOT / self.f_base.value).read_text()) or {}
        extra = yaml.safe_load(self.f_extra.value) if self.f_extra.value.strip() else {}
        name = self.f_name.value.strip()
        if not name:
            raise ValueError("Falta el nombre")
        overrides = {"arrangement": self.f_arrangement.value.strip() or None, "training_steps": int(self.f_steps.value),
                     "batch_size": int(self.f_batch.value), "image_augmentation": self.f_aug.value,
                     "torch_compile": self.f_compile.value, **(extra or {})}
        dataset = f"{DATASETS_ROOT}/{self.f_dataset.value}"
        req = Requirements(gpu_name=self.f_gpu.value.strip(), min_cores=int(self.f_cores.value),
                           min_ram_gb=int(self.f_ram.value), min_pcie_gbps=float(self.f_pcie.value))
        return train_config(base, name, dataset, overrides, self.has_wandb_key), dataset, req

    def _estimate(self):
        try:
            cfg, dataset, req = self._form()
            self.f_offers.content = "Searching…"
            ranked = self.orch.estimate(cfg, dataset, req)
        except Exception as e:  # noqa: BLE001  se muestra en el panel
            self.f_offers.content = f"**Error:** {e}"
            return
        if not ranked:
            self.f_offers.content = "No machines match these requirements right now."
            self.f_queue.disabled = True
            return
        rows = ["| Offer | CPU | $/h | PCIe | Speed | Hours | Total | Confidence |", "|---|---|---|---|---|---|---|---|"]
        for e in ranked[:OFFERS_SHOWN]:
            o = e.offer
            rows.append(f"| {o.id} | {o.cpu_name[:22]} ({o.cores:.0f}) | {o.dph:.3f} | {o.pcie_gbps:.0f} | "
                        f"{f'{e.rate:.2f}/s' if e.rate else '—'} | {f'{e.hours:.1f}' if e.hours else '—'} | "
                        f"{money(e.total)} | {e.confidence} |")
        self.f_offers.content = "\n".join(rows)
        self.f_offer.options = (AUTO, *(str(e.offer.id) for e in ranked[:OFFERS_SHOWN]))
        self.f_offer.value = AUTO
        known = ranked[0].total
        self.f_max.value = math.ceil(known * 1.5 * 10) / 10 if known else 0.0
        self.f_queue.disabled = False

    def _submit(self):
        try:
            cfg, dataset, req = self._form()
            offer_id = None if self.f_offer.value == AUTO else int(self.f_offer.value)
            self.orch.submit(self.f_name.value.strip(), cfg, dataset, req, float(self.f_max.value), offer_id)
        except Exception as e:  # noqa: BLE001  se muestra en el panel
            self.f_offers.content = f"**Error:** {e}"
            return
        self.f_queue.disabled = True
        self.f_offers.content = f"Queued **{cfg['model_name']}**."
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
    subprocess.run([sys.executable, str(ROOT / "scripts/xarm5/build_registry.py")], cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(args: Args):
    secrets = load_secrets(args.secrets)
    env = storage_env(secrets)
    pod_env = dict(env)
    if "WANDB_API_KEY" in secrets:
        pod_env["WANDB_API_KEY"] = secrets["WANDB_API_KEY"]
    orch = Orchestrator(args.queue_dir, VastProvider(), Storage(env["SPLATSIM_REMOTE"], env), pod_env,
                        image=args.image, on_model_ready=update_registry)
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
