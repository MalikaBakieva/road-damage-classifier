"""Experiment tracking wrapper.

MLflow is used when installed and enabled; otherwise every call degrades to a
no-op that still writes a local JSON run record, so the pipeline never fails
because of a missing tracking server.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .utils import get_logger, save_json

LOG = get_logger(__name__)


class RunTracker:
    def __init__(
        self,
        enabled: bool,
        experiment_name: str,
        tracking_uri: str,
        run_name: str | None = None,
        fallback_dir: str = "reports/runs",
    ) -> None:
        self.enabled = enabled
        self.run_name = run_name
        self.fallback_dir = Path(fallback_dir)
        self._mlflow = None
        self._active = False
        self._record: dict[str, Any] = {"params": {}, "metrics": [], "artifacts": []}

        if not enabled:
            return
        try:
            # MLflow 3.x refuses the filesystem store unless this opt-out is
            # set, and our configured tracking_uri is `file:./mlruns` — runs
            # stay inside the repo with no server to stand up, which is what
            # a prototype wants. `setdefault`, so an operator who exports the
            # variable (or moves to a database backend) still wins.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
        except Exception as exc:
            LOG.warning("MLflow unavailable (%s); falling back to local JSON logs.", exc)

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        if self._mlflow is not None:
            self._mlflow.start_run(run_name=self.run_name)
            self._active = True

    def end(self) -> None:
        if self._mlflow is not None and self._active:
            self._mlflow.end_run()
            self._active = False
        if self._record["params"] or self._record["metrics"]:
            name = (self.run_name or "run").replace("/", "_")
            save_json(self._record, self.fallback_dir / f"{name}.json")

    # ---------------- logging ----------------

    def log_params(self, params: dict[str, Any]) -> None:
        clean = {k: ("" if v is None else v) for k, v in params.items()}
        self._record["params"].update({k: str(v) for k, v in clean.items()})
        if self._mlflow is not None and self._active:
            try:
                self._mlflow.log_params(clean)
            except Exception as exc:
                LOG.debug("log_params failed: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        self._record["metrics"].append({"step": step, **numeric})
        if self._mlflow is not None and self._active:
            try:
                self._mlflow.log_metrics(numeric, step=step)
            except Exception as exc:
                LOG.debug("log_metrics failed: %s", exc)

    def log_artifact(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        self._record["artifacts"].append(str(path))
        if self._mlflow is not None and self._active:
            try:
                self._mlflow.log_artifact(str(path))
            except Exception as exc:
                LOG.debug("log_artifact failed: %s", exc)

    def set_tags(self, tags: dict[str, str]) -> None:
        if self._mlflow is not None and self._active:
            try:
                self._mlflow.set_tags(tags)
            except Exception as exc:
                LOG.debug("set_tags failed: %s", exc)


@contextmanager
def track(cfg) -> Any:
    tracker = RunTracker(
        enabled=cfg.tracking.enabled,
        experiment_name=cfg.tracking.experiment_name,
        tracking_uri=cfg.tracking.tracking_uri,
        run_name=cfg.tracking.run_name or f"{cfg.train.task}-{cfg.model.backbone}",
        fallback_dir=Path(cfg.reports_dir) / "runs",
    )
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.end()
