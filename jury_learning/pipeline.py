from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    import pandas as pd

from jury_learning.config import RunConfig
from jury_learning.data import DataBundle, build_data_bundle
from jury_learning.evaluation import run_evaluation as eval_run
from jury_learning.model import MoralJuryDCN
from jury_learning.training import TrainingHistory, build_model, resolve_device, train_moral_model


@dataclass
class PipelineResult:
    """Outputs from ``run_full_pipeline``; inspect metrics via the helper methods below."""

    config: RunConfig
    bundle: DataBundle
    device: torch.device
    model: Optional[MoralJuryDCN]
    training_history: Optional[TrainingHistory] = None
    eval_accuracy_by_split: Optional[dict[str, float]] = None

    def training_metrics_df(self) -> Optional["pd.DataFrame"]:
        if self.training_history is None:
            return None
        return self.training_history.to_dataframe()

    def final_training_metrics(self) -> dict[str, float]:
        """Training + validation metrics from the last epoch (if training ran)."""
        if self.training_history is None:
            return {}
        return self.training_history.last()

    def eval_metrics_df(self) -> Optional["pd.DataFrame"]:
        if self.eval_accuracy_by_split is None:
            return None
        from jury_learning.evaluation import split_metrics_table

        return split_metrics_table(self.eval_accuracy_by_split)


def prepare_data(cfg: RunConfig) -> DataBundle:
    """Load SQL, split, and build train/val dataloaders."""
    return build_data_bundle(cfg)


def run_training(
    cfg: RunConfig,
    bundle: DataBundle,
) -> tuple[MoralJuryDCN, torch.device, TrainingHistory]:
    device = resolve_device(cfg)
    model = build_model(cfg, bundle)
    model, history = train_moral_model(cfg, model, bundle, device)
    return model, device, history


def run_evaluation(
    cfg: RunConfig,
    bundle: DataBundle,
    model: Optional[MoralJuryDCN] = None,
) -> tuple[MoralJuryDCN, dict[str, float]]:
    """Evaluate on held-out splits; loads weights from ``cfg.model_path`` if ``model`` is None."""
    return eval_run(cfg, bundle, model=model)


def run_full_pipeline(cfg: RunConfig) -> PipelineResult:
    """
    End-to-end: data prep → optional training → optional evaluation.

    Access metrics via ``result.training_metrics_df()``, ``result.final_training_metrics()``,
    and ``result.eval_metrics_df()`` / ``result.eval_accuracy_by_split``.
    """
    bundle = prepare_data(cfg)
    device = resolve_device(cfg)
    model: Optional[MoralJuryDCN] = None
    history: Optional[TrainingHistory] = None
    eval_metrics: Optional[dict[str, float]] = None

    if cfg.run_training_stage:
        model, device, history = run_training(cfg, bundle)

    if cfg.run_evaluation_stage:
        model, eval_metrics = eval_run(cfg, bundle, model=model)

    return PipelineResult(
        config=cfg,
        bundle=bundle,
        device=device,
        model=model,
        training_history=history,
        eval_accuracy_by_split=eval_metrics,
    )
