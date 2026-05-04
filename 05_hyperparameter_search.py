"""Hyperparameter search using Optuna.

Searches over model size and training knobs, reporting final validation accuracy
from each trial's training history. Run the best config through the full pipeline
afterward to get all eval-split metrics.

Usage:
    python 05_hyperparameter_search.py                  # 30 trials, default data
    python 05_hyperparameter_search.py --n-trials 60
    python 05_hyperparameter_search.py --epochs 15 --n-trials 50
    python 05_hyperparameter_search.py --db-path /path/to/moral_machine.db
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

from jury_learning.config import RunConfig
from jury_learning.data import build_data_bundle
from jury_learning.pipeline import run_training


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

_EMBED_DIMS = [32, 64, 128, 256]
_HIDDEN_DIMS = [128, 256, 512, 1024]
_RESPONSE_ENCODER_HIDDENS = [32, 64, 128]
_BATCH_SIZES = [256, 512, 1024, 2048]


def _build_trial_cfg(trial: optuna.Trial, base_cfg: RunConfig, trial_epochs: int) -> RunConfig:
    embed_dim = trial.suggest_categorical("embed_dim", _EMBED_DIMS)
    hidden_dim = trial.suggest_categorical("hidden_dim", _HIDDEN_DIMS)
    num_cross_layers = trial.suggest_int("num_cross_layers", 1, 5)
    response_encoder_hidden = trial.suggest_categorical("response_encoder_hidden", _RESPONSE_ENCODER_HIDDENS)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lr_phase2 = trial.suggest_float("lr_phase2", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", _BATCH_SIZES)
    freeze_encoder_epoch_fraction = trial.suggest_float("freeze_encoder_epoch_fraction", 0.3, 0.8)

    return RunConfig(
        # inherit paths and data settings from base config
        db_path=base_cfg.db_path,
        scenarios_csv=base_cfg.scenarios_csv,
        sql_subset_size=base_cfg.sql_subset_size,
        eval_batch_size=base_cfg.eval_batch_size,
        random_seed=base_cfg.random_seed,
        new_users_holdout_fraction=base_cfg.new_users_holdout_fraction,
        new_groups_holdout_fraction=base_cfg.new_groups_holdout_fraction,
        val_fraction=base_cfg.val_fraction,
        rare_scenario_columns=base_cfg.rare_scenario_columns,
        device=base_cfg.device,
        # per-trial model and training settings
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_cross_layers=num_cross_layers,
        response_encoder_hidden=response_encoder_hidden,
        batch_size=batch_size,
        epochs=trial_epochs,
        lr=lr,
        lr_phase2=lr_phase2,
        freeze_encoder_epoch_fraction=freeze_encoder_epoch_fraction,
        # save each trial's weights to a unique temp path
        model_path=f"hp_trial_{trial.number}.pth",
        # suppress per-epoch prints; trial summary is printed below
        verbose=False,
        show_progress_bar=False,
        use_wandb=False,
        run_training_stage=True,
        run_evaluation_stage=False,
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(base_cfg: RunConfig, bundle, trial_epochs: int):
    def objective(trial: optuna.Trial) -> float:
        cfg = _build_trial_cfg(trial, base_cfg, trial_epochs)

        try:
            _, _, history = run_training(cfg, bundle)
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()
        finally:
            # clean up trial weights
            trial_path = Path(cfg.model_path)
            if trial_path.exists():
                trial_path.unlink()

        last = history.last()
        val_acc = last.get("val_accuracy", 0.0)

        print(
            f"  Trial {trial.number:3d} | "
            f"val_acc={val_acc:.4f} "
            f"embed={cfg.embed_dim} hidden={cfg.hidden_dim} "
            f"cross={cfg.num_cross_layers} reh={cfg.response_encoder_hidden} "
            f"lr={cfg.lr:.2e} lr2={cfg.lr_phase2:.2e} "
            f"bs={cfg.batch_size} freeze={cfg.freeze_encoder_epoch_fraction:.2f}"
        )
        return val_acc

    return objective


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search for MoralJuryDCN")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per trial (use fewer than full training)")
    parser.add_argument("--db-path", type=str, default=None, help="Override RunConfig.db_path")
    parser.add_argument("--sql-subset-size", type=int, default=None, help="Override RunConfig.sql_subset_size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna sampler")
    parser.add_argument("--output-dir", type=str, default=".", help="Where to save results CSV and best config JSON")
    args = parser.parse_args()

    base_cfg = RunConfig()
    if args.db_path:
        base_cfg.db_path = args.db_path
    if args.sql_subset_size:
        base_cfg.sql_subset_size = args.sql_subset_size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data once (sql_subset_size={base_cfg.sql_subset_size}) ...")
    t0 = time.time()
    # Share one bundle across all trials to avoid reloading data
    bundle = build_data_bundle(base_cfg)
    print(f"Data loaded in {time.time() - t0:.1f}s\n")

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = make_objective(base_cfg, bundle, trial_epochs=args.epochs)

    print(f"Starting search: {args.n_trials} trials × {args.epochs} epochs each\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # Save results
    results_path = output_dir / "hp_search_results.csv"
    study.trials_dataframe().to_csv(results_path, index=False)
    print(f"\nAll trial results saved to {results_path}")

    # Best trial summary
    best = study.best_trial
    print(f"\nBest trial: #{best.number}  val_accuracy={best.value:.4f}")
    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    best_cfg_dict = {k: v for k, v in best.params.items()}
    best_cfg_path = output_dir / "best_hp_config.json"
    best_cfg_path.write_text(json.dumps(best_cfg_dict, indent=2))
    print(f"Best config saved to {best_cfg_path}")
    print("\nTo train with the best config, construct a RunConfig with these values and run the full pipeline.")


if __name__ == "__main__":
    main()
