from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from jury_learning.config import RunConfig
from jury_learning.data import DataBundle, build_data_bundle
from jury_learning.evaluation import run_evaluation_report
from jury_learning.model import MoralJuryDCN
from jury_learning.training import build_model, resolve_device, train_moral_model


@dataclass
class PipelineResult:
    config: RunConfig
    bundle: DataBundle
    device: torch.device
    model: Optional[MoralJuryDCN]


def prepare_data(cfg: RunConfig) -> DataBundle:
    """Load SQL, split, and build train/val dataloaders."""
    return build_data_bundle(cfg)


def run_training(cfg: RunConfig, bundle: DataBundle) -> tuple[MoralJuryDCN, torch.device]:
    device = resolve_device(cfg)
    model = build_model(cfg, bundle)
    model = train_moral_model(cfg, model, bundle, device)
    return model, device


def run_evaluation(cfg: RunConfig, bundle: DataBundle, model: Optional[MoralJuryDCN] = None) -> MoralJuryDCN:
    """Evaluate on held-out splits; loads weights from ``cfg.model_path`` if ``model`` is None."""
    return run_evaluation_report(cfg, bundle, model=model)


def run_full_pipeline(cfg: RunConfig) -> PipelineResult:
    """
    End-to-end: data prep → optional training → optional evaluation.

    Toggle stages with ``cfg.run_training_stage`` and ``cfg.run_evaluation_stage``.
    """
    bundle = prepare_data(cfg)
    device = resolve_device(cfg)
    model: Optional[MoralJuryDCN] = None

    if cfg.run_training_stage:
        model, device = run_training(cfg, bundle)

    if cfg.run_evaluation_stage:
        model = run_evaluation(cfg, bundle, model=model)

    return PipelineResult(config=cfg, bundle=bundle, device=device, model=model)
