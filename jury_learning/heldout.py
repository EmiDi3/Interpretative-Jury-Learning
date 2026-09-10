"""Held-out jury evaluation — baselines and the jury-size sweep.

Addresses the two problems flagged in RESEARCH_PLAN.md:

(A) Agreement was measured *in-sample*: jurors were selected on the same
    scenarios they were scored on. Everything here selects the jury on a
    *selection* set of scenarios and reports agreement on a *held-out* set.

(B) Raw agreement is meaningless without baselines. Phi-4 swerves 90.5% of
    the time, so an always-swerve jury already scores 0.905. Every number is
    reported alongside the majority-class, random-jury and best-single-juror
    references.

Entry point: ``run_mvp`` — one call does baselines + sweep + figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from jury_learning.llm_jury import (
    JuryResult,
    _jury_agreement,
    find_jury_evolutionary,
)

__all__ = [
    "ScenarioSplit",
    "split_scenarios",
    "compute_baselines",
    "heldout_sweep",
    "plot_sweep",
    "run_mvp",
]


# ---------------------------------------------------------------------------
# Scenario-level train/test split
# ---------------------------------------------------------------------------

@dataclass
class ScenarioSplit:
    """Column indices of pred_matrix, split by scenario.

    Attributes
    ----------
    sel_cols  : Column indices used to *select* the jury.
    held_cols : Column indices used to *evaluate* it (never seen by the search).
    """

    sel_cols:  np.ndarray
    held_cols: np.ndarray

    def __repr__(self) -> str:
        return (f"ScenarioSplit(selection={len(self.sel_cols)}, "
                f"held_out={len(self.held_cols)})")


def split_scenarios(
    scenario_ids: np.ndarray,
    *,
    test_frac: float = 0.3,
    seed: int = 42,
) -> ScenarioSplit:
    """Split pred_matrix columns into a selection set and a held-out set."""
    n = len(scenario_ids)
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_held = max(1, int(round(n * test_frac)))

    return ScenarioSplit(
        sel_cols=np.sort(perm[n_held:]),
        held_cols=np.sort(perm[:n_held]),
    )


def _llm_vector(llm_choices: dict[int, int], scenario_ids: np.ndarray) -> np.ndarray:
    """LLM choice vector aligned to scenario_ids order."""
    return np.array([llm_choices[int(s)] for s in scenario_ids], dtype=np.int8)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def compute_baselines(
    pred_matrix:  np.ndarray,
    llm_choices:  dict[int, int],
    scenario_ids: np.ndarray,
    *,
    jury_sizes:   Sequence[int] = (1, 5, 11, 21, 51),
    n_random:     int = 200,
    seed:         int = 0,
) -> pd.DataFrame:
    """Reference points every jury result must be read against.

    Returns a tidy DataFrame with one row per baseline:
      - ``majority_class``    — always predict the LLM's modal choice
      - ``best_single_juror`` — best row of the prediction matrix (an oracle;
                                in-sample by construction, so it is an upper
                                bound on what a size-1 jury could achieve)
      - ``random_jury_K``     — mean +/- std over ``n_random`` random juries
      - ``full_population``   — all sampled users vote
    """
    rng     = np.random.default_rng(seed)
    llm_vec = _llm_vector(llm_choices, scenario_ids)
    n_users = pred_matrix.shape[0]
    dec     = (pred_matrix >= 0.5).astype(np.int8)

    rows: list[dict] = []

    # Majority class — the number any jury must beat to be interesting.
    swerve_rate = float(llm_vec.mean())
    rows.append({
        "baseline":  "majority_class",
        "jury_size": np.nan,
        "agreement": max(swerve_rate, 1.0 - swerve_rate),
        "std":       np.nan,
        "note":      f"LLM swerve rate = {swerve_rate:.3f}",
    })

    # Best single juror (oracle — selected and scored on the same columns).
    per_user = (dec == llm_vec).mean(axis=1)
    rows.append({
        "baseline":  "best_single_juror",
        "jury_size": 1,
        "agreement": float(per_user.max()),
        "std":       np.nan,
        "note":      "oracle, in-sample upper bound",
    })

    # Random juries — what you get with no search at all.
    for k in jury_sizes:
        k = int(min(k, n_users))
        scores = [
            _jury_agreement(pred_matrix, rng.choice(n_users, k, replace=False), llm_vec)
            for _ in range(n_random)
        ]
        rows.append({
            "baseline":  f"random_jury_{k}",
            "jury_size": k,
            "agreement": float(np.mean(scores)),
            "std":       float(np.std(scores)),
            "note":      f"mean +/- std over {n_random} draws",
        })

    # Whole sampled population votes.
    rows.append({
        "baseline":  "full_population",
        "jury_size": n_users,
        "agreement": _jury_agreement(pred_matrix, np.arange(n_users), llm_vec),
        "std":       np.nan,
        "note":      "all sampled users vote",
    })

    return pd.DataFrame(rows)


def random_jury_band(
    pred_matrix:  np.ndarray,
    llm_choices:  dict[int, int],
    scenario_ids: np.ndarray,
    sizes:        Sequence[int],
    *,
    n_random: int = 100,
    seed:     int = 0,
) -> pd.DataFrame:
    """Random-jury mean/std at each size — the band drawn under the sweep."""
    rng     = np.random.default_rng(seed)
    llm_vec = _llm_vector(llm_choices, scenario_ids)
    n_users = pred_matrix.shape[0]

    rows = []
    for k in sizes:
        k = int(min(k, n_users))
        scores = [
            _jury_agreement(pred_matrix, rng.choice(n_users, k, replace=False), llm_vec)
            for _ in range(n_random)
        ]
        rows.append({
            "jury_size": k,
            "random_mean": float(np.mean(scores)),
            "random_std":  float(np.std(scores)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Jury-size sweep, held out
# ---------------------------------------------------------------------------

def heldout_sweep(
    pred_matrix:  np.ndarray,
    llm_choices:  dict[int, int],
    scenario_ids: np.ndarray,
    *,
    sizes:         Sequence[int] = (1, 3, 5, 7, 9, 11, 15, 21, 31, 51),
    split:         Optional[ScenarioSplit] = None,
    test_frac:     float = 0.3,
    split_seed:    int   = 42,
    search_seed:   int   = 0,
    pop_size:      int   = 60,
    n_generations: int   = 120,
    sampled_uids:  Optional[np.ndarray] = None,
    verbose:       bool  = True,
) -> tuple[pd.DataFrame, dict[int, JuryResult], ScenarioSplit]:
    """Select each jury on the selection scenarios, score it on both sets.

    The search never sees ``split.held_cols``, so the held-out column is an
    honest estimate of how well the jury reproduces the LLM on scenarios it
    was not fitted to.

    Returns ``(tidy_df, {jury_size: JuryResult}, split)``.
    """
    if split is None:
        split = split_scenarios(scenario_ids, test_frac=test_frac, seed=split_seed)

    sel_ids,  held_ids  = scenario_ids[split.sel_cols], scenario_ids[split.held_cols]
    sel_mat,  held_mat  = pred_matrix[:, split.sel_cols], pred_matrix[:, split.held_cols]
    sel_vec,  held_vec  = _llm_vector(llm_choices, sel_ids), _llm_vector(llm_choices, held_ids)

    n_users = pred_matrix.shape[0]
    if verbose:
        print(f"{split}  |  {n_users} candidate jurors")
        print(f"{'size':>5} {'in-sample':>11} {'held-out':>10} {'gap':>8}")

    rows, results = [], {}
    for k in sizes:
        k = int(min(k, n_users))
        res = find_jury_evolutionary(
            sel_mat, llm_choices, sel_ids,
            jury_size=k,
            pop_size=pop_size,
            n_generations=n_generations,
            seed=search_seed,
            sampled_uids=sampled_uids,
            verbose=False,
        )
        in_sample = _jury_agreement(sel_mat,  res.jury_indices, sel_vec)
        held_out  = _jury_agreement(held_mat, res.jury_indices, held_vec)

        rows.append({
            "jury_size":          k,
            "in_sample":          in_sample,
            "held_out":           held_out,
            "gap":                in_sample - held_out,
            "n_sel_scenarios":    len(sel_ids),
            "n_held_scenarios":   len(held_ids),
        })
        results[k] = res
        if verbose:
            print(f"{k:>5} {in_sample:>11.4f} {held_out:>10.4f} {in_sample - held_out:>8.4f}")

    return pd.DataFrame(rows), results, split


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_sweep(
    sweep_df:  pd.DataFrame,
    *,
    band_df:   Optional[pd.DataFrame] = None,
    majority:  Optional[float] = None,
    title:     str = "Jury size vs agreement with LLM",
    save_path: Optional[str] = None,
):
    """Two curves (in-sample vs held-out) with the random-jury band beneath."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    if band_df is not None and len(band_df):
        ax.fill_between(
            band_df.jury_size,
            band_df.random_mean - band_df.random_std,
            band_df.random_mean + band_df.random_std,
            alpha=0.18, color="grey", label="random jury (±1 sd)", zorder=1,
        )
        ax.plot(band_df.jury_size, band_df.random_mean,
                color="grey", ls=":", lw=1.5, zorder=2)

    if majority is not None:
        ax.axhline(majority, color="firebrick", ls="--", lw=1.5, zorder=3,
                   label=f"majority class ({majority:.3f})")

    ax.plot(sweep_df.jury_size, sweep_df.in_sample, marker="o", lw=2,
            color="#4C72B0", label="in-sample (selection scenarios)", zorder=4)
    ax.plot(sweep_df.jury_size, sweep_df.held_out, marker="s", lw=2,
            color="#DD8452", label="held-out scenarios", zorder=5)

    ax.set_xlabel("Jury size (K)")
    ax.set_ylabel("Agreement with LLM")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xticks(sweep_df.jury_size)
    ax.get_xaxis().set_major_formatter(__import__("matplotlib").ticker.ScalarFormatter())
    ax.grid(alpha=0.3, zorder=0)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"figure saved -> {save_path}")
    return fig, ax


# ---------------------------------------------------------------------------
# One-call MVP
# ---------------------------------------------------------------------------

def run_mvp(
    pred_matrix:  np.ndarray,
    llm_choices:  dict[int, int],
    scenario_ids: np.ndarray,
    *,
    label:        str = "llm",
    sizes:        Sequence[int] = (1, 3, 5, 7, 9, 11, 15, 21, 31, 51),
    sampled_uids: Optional[np.ndarray] = None,
    test_frac:    float = 0.3,
    split_seed:   int   = 42,
    pop_size:     int   = 60,
    n_generations: int  = 120,
    n_random:     int   = 100,
    out_dir:      str   = "results",
    make_plot:    bool  = True,
) -> dict:
    """Baselines + held-out sweep + figure in one call.

    Returns a dict with ``baselines``, ``sweep``, ``band``, ``results``,
    ``split`` and ``summary`` (the headline numbers, ready to quote).
    """
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== {label} ===")
    base_df = compute_baselines(
        pred_matrix, llm_choices, scenario_ids,
        jury_sizes=[s for s in sizes if s > 1][:5],
        n_random=n_random,
    )
    print("\nBaselines:")
    print(base_df.to_string(index=False))

    print("\nHeld-out sweep:")
    sweep_df, results, split = heldout_sweep(
        pred_matrix, llm_choices, scenario_ids,
        sizes=sizes, test_frac=test_frac, split_seed=split_seed,
        pop_size=pop_size, n_generations=n_generations,
        sampled_uids=sampled_uids,
    )

    band_df = random_jury_band(
        pred_matrix[:, split.held_cols], llm_choices,
        scenario_ids[split.held_cols], sizes, n_random=n_random,
    )

    majority = float(base_df.loc[base_df.baseline == "majority_class", "agreement"].iloc[0])

    # Headline numbers.
    best_row  = sweep_df.loc[sweep_df.held_out.idxmax()]
    plateau   = float(best_row.held_out)
    at_95     = sweep_df[sweep_df.held_out >= 0.95 * plateau]
    knee      = int(at_95.jury_size.min()) if len(at_95) else int(best_row.jury_size)
    lift      = plateau - float(
        band_df.loc[band_df.jury_size == best_row.jury_size, "random_mean"].iloc[0]
    )
    monotonic = bool(sweep_df.held_out.is_monotonic_increasing)

    summary = {
        "llm":                    label,
        "majority_class":         majority,
        "best_heldout":           plateau,
        "best_heldout_size":      int(best_row.jury_size),
        "smallest_size_at_95pct": knee,
        "lift_over_random":       lift,
        "lift_over_majority":     plateau - majority,
        "heldout_monotonic":      monotonic,
        "max_in_sample_gap":      float(sweep_df.gap.max()),
    }

    sweep_df.to_csv(f"{out_dir}/size_sweep_{label}.csv", index=False)
    base_df.to_csv(f"{out_dir}/baselines_{label}.csv", index=False)

    if make_plot:
        try:
            plot_sweep(
                sweep_df, band_df=band_df, majority=majority,
                title=f"Jury size vs agreement — {label}",
                save_path=f"{out_dir}/size_sweep_{label}.png",
            )
        except ImportError:
            print("matplotlib not available — skipping figure")

    print("\n--- headline ---")
    for k, v in summary.items():
        print(f"  {k:<24} {v}")

    return {
        "baselines": base_df, "sweep": sweep_df, "band": band_df,
        "results": results, "split": split, "summary": summary,
    }
