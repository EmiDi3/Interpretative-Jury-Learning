"""Interpretative Jury Learning — modular pipeline and configuration."""

from jury_learning.config import RunConfig
from jury_learning.data import DataBundle
from jury_learning.pipeline import (
    PipelineResult,
    prepare_data,
    run_evaluation,
    run_full_pipeline,
    run_training,
)

__all__ = [
    "RunConfig",
    "DataBundle",
    "PipelineResult",
    "prepare_data",
    "run_training",
    "run_evaluation",
    "run_full_pipeline",
]
