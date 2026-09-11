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
    "sanity_checks",
    "jury_demographics",
    "compare_demographics",
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
    user_df:      Optional[pd.DataFrame] = None,
    demographics_at: Optional[int] = None,
    top_countries:   int = 12,
) -> dict:
    """Checks + baselines + held-out sweep + demographics + figure, in one call.

    Parameters
    ----------
    user_df : user-level frame with UserID and the group-feature columns
              (``bundle.df_train``). When given — together with
              ``sampled_uids`` — juror demographics are computed and saved.
    demographics_at : jury size to characterise. Defaults to the size with the
              best held-out agreement, which is the only size the result
              justifies talking about.

    Returns a dict with ``checks``, ``baselines``, ``sweep``, ``band``,
    ``results``, ``split``, ``demographics`` and ``summary``.
    """
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== {label} ===")

    checks = sanity_checks(
        pred_matrix, llm_choices, scenario_ids,
        sampled_uids=sampled_uids, label=label,
    )
    if checks.get("problems") and any(
        "NOT aligned" in p or "no LLM choice" in p for p in checks["problems"]
    ):
        raise ValueError(
            f"[{label}] prediction matrix and LLM choices are not aligned; "
            f"agreement numbers would be meaningless. See the checks above."
        )

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
        "population_agreement":   checks.get("population_agreement"),
        "median_juror_agreement": checks.get("per_user_agreement_median"),
        "condorcet_holds":        checks.get("condorcet_precondition_holds"),
        "scenario_fingerprint":   checks.get("scenario_fingerprint"),
    }

    sweep_df.to_csv(f"{out_dir}/size_sweep_{label}.csv", index=False)
    base_df.to_csv(f"{out_dir}/baselines_{label}.csv", index=False)
    pd.Series(checks).to_json(f"{out_dir}/checks_{label}.json", indent=2)

    # --- demographics of the jury the result actually justifies ----------
    demo_df = None
    if user_df is not None and sampled_uids is not None:
        k = int(demographics_at or best_row.jury_size)
        if k not in results:
            k = int(best_row.jury_size)
        print(f"\nDemographics of the K={k} jury (best held-out):")
        try:
            demo_df = jury_demographics(
                results[k], user_df, sampled_uids, top_countries=top_countries,
            )
            demo_df.insert(0, "llm", label)
            demo_df.insert(1, "jury_size", k)
            demo_df.to_csv(f"{out_dir}/demographics_{label}.csv", index=False)
            print(demo_df.drop(columns=["llm"]).to_string(index=False))
            print(f"  saved -> {out_dir}/demographics_{label}.csv")
            summary["demographics_at"] = k
        except (ValueError, KeyError) as e:
            print(f"  demographics skipped: {e}")
    elif user_df is None:
        print("\n(no user_df passed — skipping demographics; "
              "pass user_df=bundle.df_train to get them)")

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
        "checks": checks, "baselines": base_df, "sweep": sweep_df,
        "band": band_df, "results": results, "split": split,
        "demographics": demo_df, "summary": summary,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def sanity_checks(
    pred_matrix:  np.ndarray,
    llm_choices:  dict[int, int],
    scenario_ids: np.ndarray,
    *,
    sampled_uids: Optional[np.ndarray] = None,
    label:        str  = "",
    verbose:      bool = True,
) -> dict:
    """Checks that catch the failures which look like real results.

    The one that matters most is scenario alignment. If pred_matrix columns do
    not correspond to the scenario_ids they are indexed by, every agreement
    number is still a plausible-looking fraction near chance -- there is no
    exception, no NaN, nothing to notice. Population agreement below 0.5 is
    the symptom, and this is what tells the two explanations apart.

    Returns a dict of findings; ``problems`` lists anything that warrants a
    second look before the numbers are quoted.
    """
    out: dict = {"label": label}
    problems: list[str] = []

    n_users, n_scen = pred_matrix.shape
    out["n_users"], out["n_scenarios"] = n_users, n_scen

    # --- alignment -------------------------------------------------------
    if len(scenario_ids) != n_scen:
        problems.append(
            f"scenario_ids has {len(scenario_ids)} entries but pred_matrix has "
            f"{n_scen} columns — columns are NOT aligned to scenario ids"
        )
    dupes = len(scenario_ids) - len(np.unique(scenario_ids))
    out["duplicate_scenario_ids"] = int(dupes)
    if dupes:
        problems.append(f"{dupes} duplicate scenario_ids — a column is counted twice")

    missing = [int(s) for s in scenario_ids if int(s) not in llm_choices]
    out["scenario_ids_missing_from_llm"] = len(missing)
    if missing:
        problems.append(
            f"{len(missing)} scenario_ids have no LLM choice (e.g. {missing[:5]})"
        )

    out["llm_choices_total"] = len(llm_choices)
    out["llm_choices_used"]  = int(np.isin(list(llm_choices.keys()), scenario_ids).sum())
    if out["llm_choices_used"] < len(llm_choices):
        out["llm_choices_dropped"] = len(llm_choices) - out["llm_choices_used"]

    # Fingerprint of the scenario set, so two runs can be compared afterwards.
    out["scenario_fingerprint"] = _fingerprint(scenario_ids)

    if missing or len(scenario_ids) != n_scen:
        if verbose:
            _report(out, problems)
        out["problems"] = problems
        return out

    llm_vec = _llm_vector(llm_choices, scenario_ids)
    dec     = (pred_matrix >= 0.5).astype(np.int8)

    # --- what the LLM and the modelled population actually do ------------
    out["llm_swerve_rate"] = float(llm_vec.mean())

    pop_vote = (dec.mean(axis=0) >= 0.5).astype(np.int8)
    out["population_swerve_rate"]   = float(pop_vote.mean())
    out["population_agreement"]     = float((pop_vote == llm_vec).mean())

    # If the population votes one way almost always, the "population baseline"
    # is just the LLM's own class balance and carries no information.
    if out["population_swerve_rate"] < 0.02 or out["population_swerve_rate"] > 0.98:
        problems.append(
            f"population majority votes one way on "
            f"{max(out['population_swerve_rate'], 1 - out['population_swerve_rate']):.1%} "
            f"of scenarios — the population baseline is degenerate, not a real consensus"
        )

    # --- is the DCN saying anything at all? ------------------------------
    out["pred_min"]  = float(pred_matrix.min())
    out["pred_max"]  = float(pred_matrix.max())
    out["pred_mean"] = float(pred_matrix.mean())
    out["pred_std"]  = float(pred_matrix.std())
    out["frac_pred_above_half"] = float((pred_matrix >= 0.5).mean())

    per_user_std = pred_matrix.std(axis=1)
    out["users_constant_across_scenarios"] = int((per_user_std < 1e-6).sum())
    if out["users_constant_across_scenarios"] > 0.5 * n_users:
        problems.append(
            f"{out['users_constant_across_scenarios']}/{n_users} users give the same "
            f"probability to every scenario — the model is ignoring scenario features"
        )

    per_scen_std = pred_matrix.std(axis=0)
    out["scenarios_with_no_disagreement"] = int((per_scen_std < 1e-6).sum())

    # --- is any individual better than chance? ---------------------------
    per_user_agree = (dec == llm_vec).mean(axis=1)
    out["per_user_agreement_min"]    = float(per_user_agree.min())
    out["per_user_agreement_median"] = float(np.median(per_user_agree))
    out["per_user_agreement_max"]    = float(per_user_agree.max())
    out["frac_users_above_chance"]   = float((per_user_agree > 0.5).mean())

    # Condorcet's precondition. Without it, majority voting makes things worse
    # as the jury grows -- which is exactly the declining curve we observe.
    out["condorcet_precondition_holds"] = bool(out["per_user_agreement_median"] > 0.5)
    if not out["condorcet_precondition_holds"]:
        problems.append(
            f"median juror agrees with the LLM only "
            f"{out['per_user_agreement_median']:.3f} of the time (<= chance) — "
            f"majority voting will DEGRADE agreement as the jury grows; this "
            f"explains a declining size curve and is not a bug by itself"
        )

    # Is there any per-user signal at all?
    #
    # Misaligning pred_matrix columns against scenario_ids cannot be caught
    # structurally -- lengths and ids still match, nothing raises. What it
    # destroys is the signal, so test for that directly.
    majority = max(out["llm_swerve_rate"], 1.0 - out["llm_swerve_rate"])
    out["majority_class"] = float(majority)

    # Best of n_users independent coin-flippers over n_scen scenarios.
    z = float(np.sqrt(2.0 * np.log(max(n_users, 2))))
    out["best_juror_null_expectation"] = float(0.5 + z * np.sqrt(0.25 / n_scen))

    out["best_juror_beats_majority"] = bool(out["per_user_agreement_max"] > majority)
    if not out["best_juror_beats_majority"]:
        problems.append(
            f"best single juror ({out['per_user_agreement_max']:.3f}) does not beat "
            f"the majority-class baseline ({majority:.3f}) — no individual matches "
            f"the LLM better than always guessing its modal choice. Either the LLM "
            f"is too class-skewed to fit, or pred_matrix columns are misaligned "
            f"with scenario_ids (which preserves shapes and raises nothing)"
        )
    if out["per_user_agreement_max"] < out["best_juror_null_expectation"]:
        problems.append(
            f"best single juror ({out['per_user_agreement_max']:.3f}) is at or below "
            f"what {n_users} coin-flippers would reach by chance "
            f"({out['best_juror_null_expectation']:.3f}) — there is no per-user "
            f"signal here; suspect column misalignment before interpreting anything"
        )

    if sampled_uids is not None:
        out["sampled_uids_len"] = len(sampled_uids)
        if len(sampled_uids) != n_users:
            problems.append(
                f"sampled_uids has {len(sampled_uids)} entries but pred_matrix has "
                f"{n_users} rows — juror UserIDs will be wrong"
            )
        if len(np.unique(sampled_uids)) != len(sampled_uids):
            problems.append("sampled_uids contains duplicates")

    out["problems"] = problems
    if verbose:
        _report(out, problems)
    return out


def _fingerprint(scenario_ids: np.ndarray) -> str:
    """Stable short hash of the scenario set, to compare runs across LLMs."""
    import hashlib
    arr = np.asarray(sorted(int(s) for s in scenario_ids), dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def _report(out: dict, problems: list[str]) -> None:
    print("  --- sanity checks ---")
    for k in ("n_users", "n_scenarios", "duplicate_scenario_ids",
              "scenario_ids_missing_from_llm", "scenario_fingerprint",
              "llm_swerve_rate", "population_swerve_rate", "population_agreement",
              "pred_mean", "pred_std", "frac_pred_above_half",
              "users_constant_across_scenarios", "per_user_agreement_median",
              "per_user_agreement_max", "frac_users_above_chance",
              "majority_class", "best_juror_null_expectation",
              "best_juror_beats_majority", "condorcet_precondition_holds"):
        if k in out:
            v = out[k]
            print(f"    {k:<34} {v:.4f}" if isinstance(v, float) else f"    {k:<34} {v}")
    if problems:
        print("  !! worth a second look:")
        for p in problems:
            print(f"     - {p}")
    else:
        print("    no problems flagged")


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

def jury_demographics(
    jury_result:  JuryResult,
    user_df:      pd.DataFrame,
    sampled_uids: np.ndarray,
    *,
    top_countries: int = 12,
) -> pd.DataFrame:
    """Juror demographics vs the population the jury was drawn from.

    Differs from ``llm_jury.analyze_jury_demographics`` in one way that matters
    for comparing LLMs: the country rows are the ``top_countries`` largest by
    *population* share, not the largest by this jury's delta. The row set is
    therefore identical for every LLM drawn from the same sample, so the
    per-model tables line up and no row is NaN in one model and present in
    another -- which is what made the May figure unreadable.
    """
    from jury_learning.data import _NUMERIC_GROUP_FTS

    df = user_df
    if df.duplicated("UserID").any():
        df = df.groupby("UserID", sort=False).first().reset_index()

    pop_df = df[df["UserID"].isin({int(u) for u in sampled_uids})]
    if pop_df.empty:
        raise ValueError(
            "No sampled users found in user_df — check that user_df covers the "
            "same split the jury was sampled from (bundle.df_train)."
        )

    juror_df = pop_df[pop_df["UserID"].isin({int(u) for u in jury_result.jury_user_ids})]
    if juror_df.empty:
        raise ValueError("None of the jury's UserIDs are in the sampled population.")

    numeric = [c for c in _NUMERIC_GROUP_FTS if c in pop_df.columns]
    gender  = [c for c in pop_df.columns if c.startswith("Gen_")]
    country = [c for c in pop_df.columns if c.startswith("Cnt_")]
    # Stable, population-driven country set.
    country = (pop_df[country].mean().sort_values(ascending=False)
               .head(top_countries).index.tolist())

    rows = []
    for col, pretty in ([(c, c) for c in numeric]
                        + [(c, c.replace("Gen_", "gender=")) for c in gender]
                        + [(c, c.replace("Cnt_", "country=")) for c in country]):
        jm, pm = float(juror_df[col].mean()), float(pop_df[col].mean())
        rows.append({"feature": pretty, "juror_mean": jm,
                     "population_mean": pm, "delta": jm - pm})

    out = pd.DataFrame(rows)
    out.attrs["jury_size"]   = jury_result.jury_size
    out.attrs["n_jurors"]    = len(juror_df)
    out.attrs["n_population"] = len(pop_df)
    return out


def compare_demographics(
    labels:  Sequence[str],
    *,
    out_dir: str = "results",
    save:    bool = True,
):
    """Merge saved per-LLM demographic tables into one comparison.

    Reads ``<out_dir>/demographics_<label>.csv`` for each label. Because
    jury_demographics() uses a population-driven row set, the tables align on
    ``feature`` with no missing rows.
    """
    from pathlib import Path

    frames = {}
    for lab in labels:
        p = Path(out_dir) / f"demographics_{lab}.csv"
        if not p.exists():
            print(f"  (skipping {lab}: {p} not found)")
            continue
        frames[lab] = pd.read_csv(p)
    if not frames:
        raise FileNotFoundError(f"No demographics_*.csv found in {out_dir}/")

    first = next(iter(frames.values()))
    merged = first[["feature", "population_mean"]].copy()
    for lab, df in frames.items():
        merged = merged.merge(
            df[["feature", "juror_mean", "delta"]].rename(
                columns={"juror_mean": f"juror_{lab}", "delta": f"delta_{lab}"}),
            on="feature", how="outer",
        )

    if save:
        dest = Path(out_dir) / "demographics_compare.csv"
        merged.to_csv(dest, index=False)
        print(f"  saved -> {dest}")

    fig = None
    try:
        import matplotlib.pyplot as plt
        delta_cols = [c for c in merged.columns if c.startswith("delta_")]
        if delta_cols:
            h = max(4.0, 0.34 * len(merged))
            fig, ax = plt.subplots(figsize=(9, h))
            y = np.arange(len(merged))
            bar_h = 0.8 / len(delta_cols)
            for i, c in enumerate(delta_cols):
                ax.barh(y + i * bar_h, merged[c], height=bar_h,
                        label=c.replace("delta_", ""))
            ax.set_yticks(y + 0.4 - bar_h / 2)
            ax.set_yticklabels(merged.feature)
            ax.invert_yaxis()
            ax.axvline(0, color="black", lw=1)
            ax.set_xlabel("juror mean − population mean")
            ax.set_title("Jury composition vs population, by LLM")
            ax.legend()
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            if save:
                dest = Path(out_dir) / "demographics_compare.png"
                fig.savefig(dest, dpi=150, bbox_inches="tight")
                print(f"  saved -> {dest}")
    except ImportError:
        print("  matplotlib not available — table only")

    return merged, fig
