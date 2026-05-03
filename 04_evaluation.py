"""Evaluate a saved checkpoint (reloads data for splits — prefer ``run_full_pipeline`` from the notebook)."""

from jury_learning.config import RunConfig
from jury_learning.pipeline import prepare_data, run_evaluation

cfg = RunConfig()
cfg.run_training_stage = False
cfg.run_evaluation_stage = True

bundle = prepare_data(cfg)
run_evaluation(cfg, bundle)
