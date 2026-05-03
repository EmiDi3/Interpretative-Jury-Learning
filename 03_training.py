"""Train on prepared data (loads SQL and splits again — prefer ``run_full_pipeline`` from the notebook)."""

from jury_learning.config import RunConfig
from jury_learning.pipeline import prepare_data, run_training

cfg = RunConfig()
# cfg.epochs = 10
# cfg.sql_subset_size = 50_000

bundle = prepare_data(cfg)
run_training(cfg, bundle)
