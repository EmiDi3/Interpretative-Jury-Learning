from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class RunConfig:
    """Single place to set paths, data limits, model shape, and training knobs."""

    # --- Paths (relative to process cwd unless absolute) ---
    db_path: str = "moral_machine.db"
    scenarios_csv: str = "unique_scenarios.csv"
    model_path: str = "moral_jury_dcn_model.pth"

    # Optional: extract SQLite DB from a zip before running (programmatic; Colab users may prefer
    # the Drive + zip cell in 00_run_all.ipynb and set db_path to the extracted .db).
    extract_db_zip: Optional[str] = None
    extract_db_zip_dest: str = "."

    # --- Data ---
    sql_subset_size: Optional[int] = 100000  # set to None to load the full dataset
    batch_size: int = 1024
    eval_batch_size: int = 64
    random_seed: int = 42

    new_users_holdout_fraction: float = 0.1
    new_groups_holdout_fraction: float = 0.3
    val_fraction: float = 0.15

    rare_scenario_columns: tuple[str, ...] = (
        "Stay_Homeless",
        "Swerve_Homeless",
        "Stay_Stroller",
        "Swerve_Stroller",
    )

    # --- Model ---
    embed_dim: int = 128
    hidden_dim: int = 512
    num_cross_layers: int = 3
    response_encoder_hidden: int = 64

    # --- Training ---
    epochs: int = 50
    lr: float = 1e-3
    lr_phase2: float = 1e-4
    freeze_encoder_epoch_fraction: float = 2 / 3

    # --- Logging ---
    verbose: bool = True
    show_progress_bar: bool = False
    use_wandb: bool = False
    wandb_project: str = "moral-jury-model-training"

    # --- Device ---
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto"

    # --- Pipeline stages ---
    export_unique_scenarios: bool = False
    run_training_stage: bool = True
    run_evaluation_stage: bool = True
