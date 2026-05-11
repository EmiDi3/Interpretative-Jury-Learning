from __future__ import annotations

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from jury_learning.config import RunConfig
from jury_learning.data import DataBundle, MoralJuryDataset
from jury_learning.model import MoralJuryDCN
from jury_learning.training import build_model, resolve_device


def accuracy_by_split(
    model: MoralJuryDCN,
    splits: dict[str, pd.DataFrame],
    feature_dict: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Accuracy per named split (fractions in [0, 1])."""
    model.eval()
    results: dict[str, float] = {}

    for name, df_split in splits.items():
        ds = MoralJuryDataset(df_split, feature_dict)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                res_fts = batch["response_features"].to(device)
                group_fts = batch["group_features"].to(device)
                user_ids = batch["ann_id"].to(device)
                labels = batch["label"].to(device)

                outputs = model(res_fts, user_ids, group_fts).view(-1)
                predictions = (outputs > 0.0).float()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        results[name] = correct / total if total > 0 else 0.0

    return results


def accuracy_by_country(
    model: MoralJuryDCN,
    df: pd.DataFrame,
    feature_dict: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """
    Accuracy per ISO alpha-3 country code from ``Cnt_*`` dummy columns.
    Countries with a single row are skipped (returns NaN for that key) to avoid tensor shape issues.
    """
    model.eval()
    results: dict[str, float] = {}

    country_cols = [c for c in feature_dict["group_fts"] if c.startswith("Cnt_")]
    if not country_cols:
        return results

    for col in country_cols:
        code = col.replace("Cnt_", "")
        country_filter_col = f"Cnt_{code}"
        if country_filter_col not in df.columns:
            continue

        df_country = df[df[country_filter_col] == 1].copy()
        if df_country.empty:
            continue
        if len(df_country) == 1:
            results[code] = float("nan")
            continue

        ds = MoralJuryDataset(df_country, feature_dict)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                res_fts = batch["response_features"].to(device)
                group_fts = batch["group_features"].to(device)
                user_ids = batch["ann_id"].to(device)
                labels = batch["label"].to(device)

                outputs = model(res_fts, user_ids, group_fts).view(-1)
                predictions = (outputs > 0.0).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        results[code] = correct / total if total > 0 else float("nan")

    return results


def split_metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    """Single-column table of split → accuracy (easy to display in notebooks)."""
    return (
        pd.DataFrame({"accuracy": metrics})
        .rename_axis("split")
        .reset_index()
    )


def load_trained_model(cfg: RunConfig, bundle: DataBundle, device: torch.device) -> MoralJuryDCN:
    model = build_model(cfg, bundle)
    path = cfg.model_path
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    if cfg.verbose:
        print(f"Loaded model from {path}")
    return model


def run_evaluation(
    cfg: RunConfig,
    bundle: DataBundle,
    model: Optional[MoralJuryDCN] = None,
) -> tuple[MoralJuryDCN, dict[str, float]]:
    """Compute accuracies on validation and held-out test splits; no plotting."""
    device = resolve_device(cfg)
    model = model or load_trained_model(cfg, bundle, device)

    test_splits = {
        "validation": bundle.df_val,
        "new_users": bundle.df_new_users,
        "new_scenarios": bundle.df_new_scenarios,
        "new_groups": bundle.df_new_groups,
        "combined": bundle.df_combined,
    }

    metrics = accuracy_by_split(
        model=model,
        splits=test_splits,
        feature_dict=bundle.feature_dict,
        device=device,
        batch_size=cfg.eval_batch_size,
    )

    if cfg.verbose:
        for name, acc in metrics.items():
            print(f"{name:14} accuracy = {acc:.4f}")

    return model, metrics
