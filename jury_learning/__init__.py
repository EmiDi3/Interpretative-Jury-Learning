"""Interpretative Jury Learning — modular pipeline and configuration."""

from jury_learning.config import RunConfig
from jury_learning.data import DataBundle
from jury_learning.evaluation import accuracy_by_country, accuracy_by_split, split_metrics_table
from jury_learning.pipeline import (
    PipelineResult,
    prepare_data,
    run_evaluation,
    run_full_pipeline,
    run_training,
)
from jury_learning.training import TrainingHistory

# Optional figures (matplotlib / pycountry_convert): import via ``jury_learning.plots`` as needed.
from jury_learning.plots import (
    aggregate_mean_accuracy_by_continent,
    continent_metrics_table,
    country_metrics_table,
    plot_calibration,
    plot_continent_accuracy,
    plot_country_accuracy,
    plot_split_generalization,
    plot_training_history,
    run_evaluation_plots,
)

__all__ = [
    "RunConfig",
    "DataBundle",
    "TrainingHistory",
    "PipelineResult",
    "accuracy_by_country",
    "accuracy_by_split",
    "split_metrics_table",
    "prepare_data",
    "run_training",
    "run_evaluation",
    "run_full_pipeline",
    "aggregate_mean_accuracy_by_continent",
    "continent_metrics_table",
    "country_metrics_table",
    "plot_calibration",
    "plot_continent_accuracy",
    "plot_country_accuracy",
    "plot_split_generalization",
    "plot_training_history",
    "run_evaluation_plots",
]
