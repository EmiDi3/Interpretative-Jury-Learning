"""
Optional evaluation plots (matplotlib).

Import this module only when you want figures; core metrics stay in ``evaluation`` /
``training`` without a hard dependency on matplotlib in those modules.

Continent aggregation uses ``pycountry_convert`` when installed::

    pip install pycountry_convert matplotlib
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from jury_learning.config import RunConfig
from jury_learning.data import DataBundle, MoralJuryDataset
from jury_learning.evaluation import accuracy_by_country
from jury_learning.model import MoralJuryDCN
from jury_learning.training import TrainingHistory, resolve_device


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Plotting requires matplotlib: pip install matplotlib") from e
    return plt


def aggregate_mean_accuracy_by_continent(country_accuracy: dict[str, float]) -> dict[str, float]:
    """Map alpha-3 codes to continents and average accuracies (skips NaN)."""
    try:
        import pycountry_convert as pc
    except ImportError as e:
        raise ImportError(
            "Continent plots require pycountry_convert: pip install pycountry_convert"
        ) from e

    def continent_for_code(code: str) -> str:
        if code == "AUS":
            return "Australia"
        try:
            iso2 = pc.country_alpha3_to_country_alpha2(code)
            cc = pc.country_alpha2_to_continent_code(iso2)
            return pc.convert_continent_code_to_continent_name(cc)
        except (KeyError, ValueError):
            return "Unknown"

    buckets: dict[str, list[float]] = {}
    for code, acc in country_accuracy.items():
        if acc != acc or pd.isna(acc):  # NaN
            continue
        cont = continent_for_code(code)
        buckets.setdefault(cont, []).append(float(acc))

    out = {}
    for cont, vals in buckets.items():
        if cont == "Unknown" or not vals:
            continue
        out[cont] = float(np.mean(vals))
    return out


def plot_split_generalization(
    split_accuracy: dict[str, float],
    *,
    title: str = "Accuracy by evaluation split",
    ylim: tuple[float, float] = (0.0, 1.0),
    rotation: int = 25,
) -> None:
    """Bar chart for split-level metrics from ``run_evaluation``."""
    plt = _plt()
    names = list(split_accuracy.keys())
    values = [split_accuracy[k] for k in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    ax.bar(names, values, color=colors)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    for i, v in enumerate(values):
        ax.text(i, min(v + 0.02, 0.99), f"{v:.1%}", ha="center", fontsize=9)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.show()


def plot_training_history(
    history: TrainingHistory,
    *,
    title: str = "Training curves",
) -> None:
    """Train vs validation loss and accuracy over epochs."""
    plt = _plt()
    df = history.to_dataframe()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["train_accuracy"], label="train")
    axes[1].plot(df["epoch"], df["val_accuracy"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_continent_accuracy(
    country_accuracy: dict[str, float],
    *,
    title: str = "Mean accuracy by continent",
    sort_ascending: bool = False,
) -> None:
    """Bar chart of mean accuracy per continent (from country-level accuracies)."""
    continent_acc = aggregate_mean_accuracy_by_continent(country_accuracy)
    if not continent_acc:
        print("No continent aggregates to plot (missing pycountry_convert or no valid countries).")
        return

    plt = _plt()
    items = sorted(continent_acc.items(), key=lambda x: x[1], reverse=not sort_ascending)
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, vals, color="steelblue")
    ax.set_ylabel("Mean accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, min(v + 0.02, 0.98), f"{v:.1%}", ha="center", fontsize=9)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()


def plot_country_accuracy(
    country_accuracy: dict[str, float],
    *,
    title: str = "Accuracy by country (top k)",
    max_countries: int = 30,
    priority_codes: Optional[tuple[str, ...]] = None,
) -> None:
    """
    Bar chart over countries with valid accuracies.
    Shows ``priority_codes`` first (if present), then fills up to ``max_countries`` by accuracy.
    """
    plt = _plt()
    valid = {k: v for k, v in country_accuracy.items() if pd.notna(v) and np.isfinite(v)}
    if not valid:
        print("No country-level accuracies to plot.")
        return

    ordered: list[tuple[str, float]] = []
    seen: set[str] = set()
    if priority_codes:
        for c in priority_codes:
            if c in valid:
                ordered.append((c, valid[c]))
                seen.add(c)
    rest = sorted(((k, v) for k, v in valid.items() if k not in seen), key=lambda x: -x[1])
    for pair in rest:
        if len(ordered) >= max_countries:
            break
        ordered.append(pair)

    names = [p[0] for p in ordered]
    vals = [p[1] for p in ordered]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.35), 5))
    ax.bar(names, vals, color="cadetblue")
    ax.set_ylabel("Accuracy")
    ax.set_title(title + f" (n={len(names)})")
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, min(v + 0.015, 0.98), f"{v:.0%}", ha="center", fontsize=8, rotation=0)
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_calibration(
    model: MoralJuryDCN,
    df: pd.DataFrame,
    feature_dict: dict,
    device: torch.device,
    batch_size: int,
    *,
    n_bins: int = 10,
    title: str = "Reliability diagram",
) -> None:
    """Mean predicted probability vs observed frequency per bin (validation-style check)."""
    plt = _plt()
    model.eval()
    ds = MoralJuryDataset(df, feature_dict)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            res = batch["response_features"].to(device)
            grp = batch["group_features"].to(device)
            uid = batch["ann_id"].to(device)
            y = batch["label"].to(device)
            p = model(res, uid, grp).view(-1)
            probs.extend(p.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    p = np.array(probs, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)
    if len(p) == 0:
        print("Calibration: empty dataframe, skipping.")
        return

    bin_ids = np.floor(p * n_bins).astype(int)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    mean_pred: list[float] = []
    mean_obs: list[float] = []
    for b in range(n_bins):
        m = bin_ids == b
        if int(m.sum()) == 0:
            continue
        mean_pred.append(float(p[m].mean()))
        mean_obs.append(float(y[m].mean()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    if mean_pred:
        ax.plot(mean_pred, mean_obs, "o-", color="darkorange", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def country_metrics_table(country_accuracy: dict[str, float]) -> pd.DataFrame:
    """Long table for notebooks."""
    s = pd.Series(country_accuracy, name="accuracy").rename_axis("country_code")
    return s.reset_index().sort_values("accuracy", ascending=False, na_position="last")


def continent_metrics_table(country_accuracy: dict[str, float]) -> pd.DataFrame:
    agg = aggregate_mean_accuracy_by_continent(country_accuracy)
    return (
        pd.Series(agg, name="mean_accuracy")
        .rename_axis("continent")
        .reset_index()
        .sort_values("mean_accuracy", ascending=False)
    )


def run_evaluation_plots(
    cfg: RunConfig,
    bundle: DataBundle,
    model: MoralJuryDCN,
    *,
    split_metrics: Optional[dict[str, float]] = None,
    history: Optional[TrainingHistory] = None,
    plot_calibration_on: str = "validation",
    country_priority: tuple[str, ...] = ("USA", "GBR", "DEU", "CHN", "CAN", "FRA", "BRA"),
) -> None:
    """
    Convenience: split overview, training curves (if history), calibration, continents and countries
    on validation and new-user splits.

    Parameters
    ----------
    plot_calibration_on
        ``\"validation\"`` or ``\"train\"`` — which dataframe to use for the reliability diagram.
    """
    device = resolve_device(cfg)
    model.eval()

    if split_metrics:
        plot_split_generalization(split_metrics, title="Generalization — split accuracy")

    if history is not None and history.epoch:
        plot_training_history(history)

    df_cal = bundle.df_val if plot_calibration_on == "validation" else bundle.df_train
    if len(df_cal) > 0:
        plot_calibration(
            model,
            df_cal,
            bundle.feature_dict,
            device,
            cfg.eval_batch_size,
            title=f"Calibration ({plot_calibration_on})",
        )

    acc_val = accuracy_by_country(model, bundle.df_val, bundle.feature_dict, device, cfg.eval_batch_size)
    if acc_val:
        plot_continent_accuracy(acc_val, title="Continents — validation set")
        plot_country_accuracy(
            acc_val,
            title="Countries — validation set",
            priority_codes=country_priority,
        )

    acc_nu = accuracy_by_country(model, bundle.df_new_users, bundle.feature_dict, device, cfg.eval_batch_size)
    if acc_nu:
        plot_continent_accuracy(acc_nu, title="Continents — new-user split")
        plot_country_accuracy(
            acc_nu,
            title="Countries — new-user split",
            priority_codes=country_priority,
        )
