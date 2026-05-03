"""Evaluate a saved checkpoint (reloads data — prefer ``run_full_pipeline`` from the notebook)."""

from jury_learning.config import RunConfig
from jury_learning.evaluation import split_metrics_table
from jury_learning.pipeline import prepare_data, run_evaluation

cfg = RunConfig()

bundle = prepare_data(cfg)
_model, metrics = run_evaluation(cfg, bundle)

print(split_metrics_table(metrics))
