"""Train on prepared data (reloads data — prefer ``run_full_pipeline`` from the notebook)."""

from jury_learning.config import RunConfig
from jury_learning.pipeline import prepare_data, run_training

cfg = RunConfig()

bundle = prepare_data(cfg)
model, device, history = run_training(cfg, bundle)
