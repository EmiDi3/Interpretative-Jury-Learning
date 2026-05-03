"""Prepare data only (same logic as the full pipeline). Edit ``RunConfig`` below or import from the notebook."""

from jury_learning.config import RunConfig
from jury_learning.pipeline import prepare_data

cfg = RunConfig()
# Example overrides when running this file directly:
# cfg.sql_subset_size = 500_000
# cfg.export_unique_scenarios = True

prepare_data(cfg)
