"""Logistic-regression baseline evaluated across demographic groups.

Uses ``sklearn.linear_model.SGDClassifier`` (logistic loss) with ``partial_fit``
over the existing DataLoader so the full dataset never materialises in memory at
once — consistent with the memory budget used by the neural models.

Typical usage
-------------
    from jury_learning import prepare_data, RunConfig
    from jury_learning.logistic_baseline import train_logistic_baseline

    cfg = RunConfig(sql_subset_size=None)
    bundle = prepare_data(cfg)
    lr_result = train_logistic_baseline(bundle, cfg)
    print(lr_result.metrics_df())
    print(lr_result.group_metrics_df())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from jury_learning.config import RunConfig
from jury_learning.data import DataBundle, MoralJuryDataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batch_to_X(batch: dict) -> np.ndarray:
    """Concatenate response and group features from a DataLoader batch → float32 array."""
    resp  = batch["response_features"].numpy()   # [B, n_resp]
    group = batch["group_features"].numpy()       # [B, n_group]
    return np.hstack([resp, group])               # [B, n_resp+n_group]


def _batch_to_y(batch: dict) -> np.ndarray:
    return batch["label"].numpy().astype(np.int8)


def _predict_batched(
    clf: SGDClassifier,
    scaler: StandardScaler,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, labels) arrays over the entire loader, without OOM risk."""
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in loader:
        X = scaler.transform(_batch_to_X(batch))
        all_preds.append(clf.predict(X))
        all_labels.append(_batch_to_y(batch))
    return np.concatenate(all_preds), np.concatenate(all_labels)


def _accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean()) if len(labels) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LogisticBaselineResult:
    """Holds the fitted model, scaler, and all evaluation metrics.

    Attributes
    ----------
    clf, scaler :
        Fitted ``SGDClassifier`` and ``StandardScaler`` — use them to score
        new data with ``scaler.transform(X)`` then ``clf.predict(X_scaled)``.
    metrics : dict[str, float]
        Accuracy on standard evaluation splits (validation, new_users, …).
    group_metrics : dict[str, dict[str, float]]
        Accuracy keyed by ``{"country": {iso3: acc}, "gender": {label: acc}}``.
    """

    clf:          SGDClassifier
    scaler:       StandardScaler
    metrics:      dict[str, float] = field(default_factory=dict)
    group_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # ----- Convenience methods -----

    def metrics_df(self) -> pd.DataFrame:
        """One-row-per-split accuracy table."""
        return (
            pd.DataFrame({"accuracy": self.metrics})
            .rename_axis("split")
            .reset_index()
            .sort_values("split")
        )

    def group_metrics_df(self, group: str = "country") -> pd.DataFrame:
        """Accuracy table for a specific group dimension (``'country'`` or ``'gender'``).

        Parameters
        ----------
        group : str
            Key from ``group_metrics`` — ``'country'`` or ``'gender'``.
        """
        data = self.group_metrics.get(group, {})
        return (
            pd.DataFrame({"accuracy": data})
            .rename_axis(group)
            .reset_index()
            .dropna(subset=["accuracy"])
            .sort_values("accuracy", ascending=False)
        )

    def combined_df(self) -> pd.DataFrame:
        """All metrics (standard splits + country + gender) in one table."""
        rows = [{"dimension": "split", "group": k, "accuracy": v}
                for k, v in self.metrics.items()]
        for dim, mapping in self.group_metrics.items():
            for label, acc in mapping.items():
                if not np.isnan(acc):
                    rows.append({"dimension": dim, "group": label, "accuracy": acc})
        return pd.DataFrame(rows).sort_values(["dimension", "accuracy"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_logistic_baseline(
    bundle: DataBundle,
    cfg: RunConfig,
    *,
    n_epochs: int = 3,
    alpha: float = 1e-4,   # L2 regularisation strength
    eval_batch_size: Optional[int] = None,
    eval_on_train: bool = False,
) -> LogisticBaselineResult:
    """Train a logistic-regression baseline on the full training set and evaluate
    on every held-out split, broken down by country and gender.

    Parameters
    ----------
    bundle : DataBundle
        Pre-split data bundle (output of ``prepare_data``).
    cfg : RunConfig
        Run configuration (uses ``cfg.random_seed``, ``cfg.eval_batch_size``,
        ``cfg.verbose``).
    n_epochs : int
        Number of passes over the training data.  3 is usually sufficient for
        convergence at the default learning-rate schedule.
    alpha : float
        L2 penalty.  Equivalent to ``C = 1/alpha`` in ``LogisticRegression``.
    eval_batch_size : int | None
        Batch size for evaluation loaders.  Defaults to ``cfg.eval_batch_size``.
    eval_on_train : bool
        Whether to also evaluate on the (sampled) training set.

    Returns
    -------
    LogisticBaselineResult
    """
    verbose  = cfg.verbose
    bs_eval  = eval_batch_size or cfg.eval_batch_size

    # SGDClassifier with log-loss == stochastic logistic regression
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        random_state=cfg.random_seed,
        warm_start=False,
    )
    scaler = StandardScaler()
    classes = np.array([0, 1], dtype=np.int8)

    # ------------------------------------------------------------------
    # Pass 0 — fit the scaler incrementally (one pass, no model update)
    # ------------------------------------------------------------------
    if verbose:
        print("Logistic baseline — fitting scaler (pass 0 / warm-up)…")
    for batch in bundle.train_loader:
        scaler.partial_fit(_batch_to_X(batch))

    # ------------------------------------------------------------------
    # Passes 1..n_epochs — train classifier on scaled batches
    # ------------------------------------------------------------------
    for epoch in range(n_epochs):
        n_correct = n_total = 0
        for batch in bundle.train_loader:
            X = scaler.transform(_batch_to_X(batch))
            y = _batch_to_y(batch)
            clf.partial_fit(X, y, classes=classes)
            preds = clf.predict(X)
            n_correct += int((preds == y).sum())
            n_total   += len(y)
        if verbose:
            print(f"  Epoch {epoch + 1}/{n_epochs} — train accuracy = {n_correct / max(n_total, 1):.4f}")

    # ------------------------------------------------------------------
    # Evaluation — standard splits
    # ------------------------------------------------------------------
    if verbose:
        print("Evaluating on held-out splits…")

    split_dfs: dict[str, pd.DataFrame] = {
        "validation":    bundle.df_val,
        "new_users":     bundle.df_new_users,
        "new_scenarios": bundle.df_new_scenarios,
        "new_groups":    bundle.df_new_groups,
        "combined":      bundle.df_combined,
    }

    metrics: dict[str, float] = {}
    for name, df_split in split_dfs.items():
        if df_split is None or len(df_split) == 0:
            continue
        loader = DataLoader(
            MoralJuryDataset(df_split, bundle.feature_dict),
            batch_size=bs_eval,
            shuffle=False,
        )
        preds, labels = _predict_batched(clf, scaler, loader)
        metrics[name] = _accuracy(preds, labels)
        if verbose:
            print(f"  {name:20s}  accuracy = {metrics[name]:.4f}")

    # Optionally score the training split
    if eval_on_train:
        preds, labels = _predict_batched(clf, scaler, bundle.train_loader)
        metrics["train"] = _accuracy(preds, labels)
        if verbose:
            print(f"  {'train':20s}  accuracy = {metrics['train']:.4f}")

    # ------------------------------------------------------------------
    # Evaluation — across demographic groups (country and gender)
    # ------------------------------------------------------------------
    group_metrics = _eval_across_groups(clf, scaler, bundle, bs_eval, verbose=verbose)

    if verbose:
        print("Logistic baseline complete.")

    return LogisticBaselineResult(
        clf=clf,
        scaler=scaler,
        metrics=metrics,
        group_metrics=group_metrics,
    )


# ---------------------------------------------------------------------------
# Group-wise evaluation
# ---------------------------------------------------------------------------

def _eval_across_groups(
    clf: SGDClassifier,
    scaler: StandardScaler,
    bundle: DataBundle,
    batch_size: int,
    *,
    verbose: bool = False,
) -> dict[str, dict[str, float]]:
    """Evaluate accuracy per country and per gender on the validation split."""
    fd = bundle.feature_dict
    df = bundle.df_val

    country_accs = _eval_by_dummy_prefix(clf, scaler, df, fd, "Cnt_", batch_size)
    gender_accs  = _eval_by_dummy_prefix(clf, scaler, df, fd, "Gen_", batch_size)

    if verbose:
        _print_group_table("Country accuracy (validation)", country_accs, top_n=15)
        _print_group_table("Gender accuracy  (validation)", gender_accs)

    return {
        "country": country_accs,
        "gender":  gender_accs,
    }


def _eval_by_dummy_prefix(
    clf: SGDClassifier,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_dict: dict,
    prefix: str,
    batch_size: int,
) -> dict[str, float]:
    """For each one-hot dummy column starting with *prefix*, score the subset of rows
    where that dummy == 1."""
    cols = [c for c in feature_dict["group_fts"] if c.startswith(prefix)]
    results: dict[str, float] = {}

    for col in cols:
        label = col[len(prefix):]          # e.g. "Cnt_USA" → "USA"
        df_sub = df[df[col] == 1]
        if len(df_sub) < 2:
            results[label] = float("nan")
            continue
        loader = DataLoader(
            MoralJuryDataset(df_sub, feature_dict),
            batch_size=batch_size,
            shuffle=False,
        )
        preds, labels = _predict_batched(clf, scaler, loader)
        results[label] = _accuracy(preds, labels)

    return results


def _print_group_table(title: str, accs: dict[str, float], top_n: Optional[int] = None) -> None:
    valid = {k: v for k, v in accs.items() if not np.isnan(v)}
    ranked = sorted(valid.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    print(f"\n{title}")
    print(f"  {'Group':<12}  Accuracy")
    print(f"  {'-'*12}  --------")
    for label, acc in ranked:
        print(f"  {label:<12}  {acc:.4f}")
