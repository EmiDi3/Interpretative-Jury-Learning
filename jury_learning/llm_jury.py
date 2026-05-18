"""Counterfactual Jury Learning for LLM responses.

Given a set of LLM decisions on Moral Machine scenarios, this module finds
the *smallest jury of real human participants* (selected from the Moral Machine
survey population via the trained DCN model) whose majority vote best replicates
the LLM's pattern of choices.

Two search strategies are provided:
  - Evolutionary Algorithm (EA)  — genetic search, no gradient assumptions
  - ML / Greedy + L1             — greedy forward selection + sparse logistic regression

Typical usage
-------------
    from jury_learning.llm_jury import (
        load_llm_responses, build_prediction_matrix,
        find_jury_evolutionary, find_jury_ml, JuryResult,
    )
    from jury_learning import prepare_data, RunConfig
    from jury_learning.evaluation import load_trained_model

    cfg = RunConfig(sql_subset_size=None)
    bundle = prepare_data(cfg)
    model  = load_trained_model(cfg, bundle, device)

    llm_choices = load_llm_responses("phi4.json")          # {scenario_id: 0/1}
    pred_matrix, user_ids, scenario_ids = build_prediction_matrix(
        model, bundle, llm_choices, device, n_users=2000
    )

    result_ea  = find_jury_evolutionary(pred_matrix, llm_choices,
                                        scenario_ids, jury_size=12)
    result_ml  = find_jury_ml(pred_matrix, llm_choices,
                               scenario_ids, jury_size=12)

    print(result_ea)
    print(result_ml)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from jury_learning.data import DataBundle, _NUMERIC_GROUP_FTS


# ---------------------------------------------------------------------------
# Character-name → column mapping (matches data.py's character_cols)
# ---------------------------------------------------------------------------

_CHAR_MAP: dict[str, str] = {
    "man":                 "Man",
    "men":                 "Man",
    "woman":               "Woman",
    "women":               "Woman",
    "boy":                 "Boy",
    "boys":                "Boy",
    "girl":                "Girl",
    "girls":               "Girl",
    "elderly man":         "OldMan",
    "elderly men":         "OldMan",
    "elderly woman":       "OldWoman",
    "elderly women":       "OldWoman",
    "pregnant woman":      "Pregnant",
    "pregnant women":      "Pregnant",
    "baby in a stroller":  "Stroller",
    "babies in strollers": "Stroller",
    "homeless person":     "Homeless",
    "homeless people":     "Homeless",
    "large man":           "LargeMan",
    "large men":           "LargeMan",
    "large woman":         "LargeWoman",
    "large women":         "LargeWoman",
    "criminal":            "Criminal",
    "criminals":           "Criminal",
    "male executive":      "MaleExecutive",
    "male executives":     "MaleExecutive",
    "female executive":    "FemaleExecutive",
    "female executives":   "FemaleExecutive",
    "male athlete":        "MaleAthlete",
    "male athletes":       "MaleAthlete",
    "female athlete":      "FemaleAthlete",
    "female athletes":     "FemaleAthlete",
    "male doctor":         "MaleDoctor",
    "male doctors":        "MaleDoctor",
    "female doctor":       "FemaleDoctor",
    "female doctors":      "FemaleDoctor",
    "dog":                 "Dog",
    "dogs":                "Dog",
    "cat":                 "Cat",
    "cats":                "Cat",
}

_CHAR_COLS = [
    "NumberOfCharacters", "Pedped", "Barrier", "CrossingSignal",
    "Man", "Woman", "Pregnant", "Stroller", "OldMan", "OldWoman",
    "Boy", "Girl", "Homeless", "LargeWoman", "LargeMan", "Criminal",
    "MaleExecutive", "FemaleExecutive", "FemaleAthlete", "MaleAthlete",
    "FemaleDoctor", "MaleDoctor", "Dog", "Cat",
]

_RESPONSE_FTS = [f"Stay_{c}" for c in _CHAR_COLS] + [f"Swerve_{c}" for c in _CHAR_COLS]


# ---------------------------------------------------------------------------
# Prompt parser — extract scenario feature vector from natural-language prompt
# ---------------------------------------------------------------------------

def _parse_side(kills_text: str, signal_text: str) -> dict[str, int]:
    """Parse one side (Stay/Swerve) of an LLM prompt into character counts."""
    counts: dict[str, int] = {c: 0 for c in _CHAR_COLS}

    # CrossingSignal: 0=none, 1=green, 2=red
    if "green light" in signal_text:
        counts["CrossingSignal"] = 1
    elif "red light" in signal_text:
        counts["CrossingSignal"] = 2

    # Barrier (rarely appears in generated prompts — default 0)
    counts["Barrier"] = 0

    # PedPed — set later at scenario level
    counts["Pedped"] = 1   # all LLM scenarios are ped-vs-ped

    # Character counts — match longest phrases first to avoid partial matches
    sorted_names = sorted(_CHAR_MAP.keys(), key=len, reverse=True)
    remaining = kills_text.lower()

    for name in sorted_names:
        pattern = rf"(\d+)\s+{re.escape(name)}"
        for m in re.finditer(pattern, remaining):
            col = _CHAR_MAP[name]
            counts[col] += int(m.group(1))
            # blank out so we don't double-count
            remaining = remaining[:m.start()] + " " * len(m.group(0)) + remaining[m.end():]

    counts["NumberOfCharacters"] = sum(
        counts[c] for c in _CHAR_COLS
        if c not in ("NumberOfCharacters", "Pedped", "Barrier", "CrossingSignal")
    )
    return counts


def parse_prompt_to_features(prompt: str) -> dict[str, int]:
    """Extract Stay_* and Swerve_* feature dict from one LLM prompt string."""
    # Split on the two bold headings
    stay_m  = re.search(r"\*\*Stay on course\*\*\s*[—-]\s*kills:\s*(.+?)(?:\n|$)", prompt)
    swerve_m = re.search(r"\*\*Swerve\*\*\s*[—-]\s*kills:\s*(.+?)(?:\n|$)", prompt)

    if not stay_m or not swerve_m:
        return {f: 0 for f in _RESPONSE_FTS}

    stay_kills  = stay_m.group(1).strip()
    swerve_kills = swerve_m.group(1).strip()

    # Signal context comes from the sentence after the kills list
    stay_ctx   = stay_kills
    swerve_ctx = swerve_kills

    stay_feats   = _parse_side(stay_kills,   stay_ctx)
    swerve_feats = _parse_side(swerve_kills, swerve_ctx)

    row: dict[str, int] = {}
    for col in _CHAR_COLS:
        row[f"Stay_{col}"]   = stay_feats[col]
        row[f"Swerve_{col}"] = swerve_feats[col]
    return row


def load_scenario_features(
    llm_path: str | Path,
    unique_scenarios_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Return a DataFrame (scenario_id, Stay_*, Swerve_*) aligned to the LLM file.

    If *unique_scenarios_csv* is provided and contains all needed scenario IDs, it
    is used directly (faster, exact).  Otherwise the prompts in the LLM file are
    parsed to reconstruct the feature matrix.
    """
    records = json.load(open(llm_path))
    scenario_ids = [r["scenario_id"] for r in records]
    prompts      = {r["scenario_id"]: r["prompt"] for r in records}

    if unique_scenarios_csv and Path(unique_scenarios_csv).is_file():
        df = pd.read_csv(unique_scenarios_csv)
        # Filter to the scenario IDs we need (by row index = scenario_id)
        df = df.iloc[scenario_ids].copy()
        df.insert(0, "scenario_id", scenario_ids)
        # Keep only the response feature columns
        feat_cols = [c for c in _RESPONSE_FTS if c in df.columns]
        return df[["scenario_id"] + feat_cols].reset_index(drop=True)

    # Fall back to prompt parsing
    rows = []
    for sid in scenario_ids:
        row = {"scenario_id": sid}
        row.update(parse_prompt_to_features(prompts[sid]))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LLM response loader
# ---------------------------------------------------------------------------

def load_llm_responses(path: str | Path) -> dict[int, int]:
    """Return {scenario_id: 0 (Stay) / 1 (Swerve)} for each record in the JSON file.

    Records with missing or invalid choices are skipped.
    """
    records = json.load(open(path))
    result: dict[int, int] = {}
    for r in records:
        choice = r.get("choice", "").strip()
        if choice == "Swerve":
            result[r["scenario_id"]] = 1
        elif choice == "Stay":
            result[r["scenario_id"]] = 0
    return result


# ---------------------------------------------------------------------------
# Prediction matrix builder
# ---------------------------------------------------------------------------

def _sample_users(bundle: DataBundle, n_users: int, seed: int = 42) -> pd.DataFrame:
    """Sample *n_users* unique users from the training data, one row per user."""
    df = bundle.df_train
    users = df.groupby("UserID", sort=False).first().reset_index()
    if len(users) > n_users:
        users = users.sample(n_users, random_state=seed)
    return users.reset_index(drop=True)


def build_prediction_matrix(
    model: torch.nn.Module,
    bundle: DataBundle,
    llm_choices: dict[int, int],
    device: torch.device,
    *,
    n_users: int = 2000,
    batch_size: int = 512,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a (n_users × n_scenarios) matrix of swerve probabilities.

    Uses the trained model to predict P(swerve) for every (user, scenario) pair.

    Parameters
    ----------
    model      : Trained MoralJuryDCN (or compatible).
    bundle     : DataBundle from prepare_data().
    llm_choices: {scenario_id: 0/1} dict — only scenarios in this dict are used.
    device     : torch.device.
    n_users    : Number of users to sample from the training set.
    batch_size : Inference batch size (per-user).
    seed       : Random seed for user sampling.
    verbose    : Print progress.

    Returns
    -------
    pred_matrix  : float32 ndarray shape (n_users, n_scenarios)
    sampled_uids : int64 ndarray shape (n_users,) — UserIDs of sampled users
    scenario_ids : int64 ndarray shape (n_scenarios,) — ordered scenario IDs
    """
    model.eval()
    fd = bundle.feature_dict

    # --- scenario features (parsed from prompts in the bundle's scenario CSV or LLM JSON) ---
    scenario_ids = np.array(sorted(llm_choices.keys()), dtype=np.int64)
    n_scenarios = len(scenario_ids)

    # Build scenario feature tensor from the DataBundle's scenario data
    # We look up scenario_id rows from the DataBundle's stored feature dict
    # Fallback: we need unique_scenarios.csv or we parse inline.
    # Here we use a helper that checks for the CSV first.
    llm_files = list(Path(".").glob("*.json"))
    llm_file = llm_files[0] if llm_files else None

    if verbose:
        print(f"Building prediction matrix: {n_users} users × {n_scenarios} scenarios …")

    # --- sample users ---
    user_df = _sample_users(bundle, n_users, seed=seed)
    sampled_uids = user_df["UserID"].to_numpy(dtype=np.int64)
    n_users_actual = len(user_df)

    gf_cols  = fd["group_fts"]
    num_cols = [c for c in gf_cols if c in _NUMERIC_GROUP_FTS]
    dum_cols = [c for c in gf_cols if c not in _NUMERIC_GROUP_FTS]

    # --- scenario response features ---
    # Try to retrieve from bundle's training data by matching scenario content,
    # or use the pre-parsed scenario feature rows.
    # Since we don't have scenario_id in the DataBundle's df, we use prompt parsing.
    # We'll pass a placeholder and let the caller provide scenario_features_df.
    # For now raise a helpful error.
    raise RuntimeError(
        "build_prediction_matrix requires scenario_features_df — "
        "call build_prediction_matrix_from_features() directly."
    )


def build_prediction_matrix_from_features(
    model: torch.nn.Module,
    bundle: DataBundle,
    scenario_features_df: pd.DataFrame,
    llm_choices: dict[int, int],
    device: torch.device,
    *,
    n_users: int = 2000,
    batch_size: int = 1024,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (n_users × n_scenarios) prediction matrix given scenario features.

    Parameters
    ----------
    scenario_features_df : DataFrame with columns [scenario_id, Stay_*, Swerve_*].
                           Rows need not be sorted; will be aligned to llm_choices.

    Returns
    -------
    pred_matrix  : float32 ndarray (n_users, n_scenarios) — P(swerve)
    sampled_uids : int64 ndarray (n_users,)
    scenario_ids : int64 ndarray (n_scenarios,) — scenario order matches pred_matrix cols
    """
    model.eval()
    fd = bundle.feature_dict

    # Align scenarios
    scenario_ids = np.array(sorted(llm_choices.keys()), dtype=np.int64)
    n_scenarios  = len(scenario_ids)

    sid_to_row = {int(sid): i for i, sid in enumerate(scenario_features_df["scenario_id"])}
    row_order  = [sid_to_row[int(s)] for s in scenario_ids]
    scen_df    = scenario_features_df.iloc[row_order].reset_index(drop=True)

    # Build response feature tensor (n_scenarios, n_resp)
    resp_cols    = [c for c in _RESPONSE_FTS if c in scen_df.columns]
    scen_resp_np = scen_df[resp_cols].to_numpy(dtype=np.float32)   # (S, n_resp)
    scen_resp_t  = torch.tensor(scen_resp_np, dtype=torch.float32).to(device)

    # Sample users
    user_df = _sample_users(bundle, n_users, seed=seed)
    n_users_actual = len(user_df)
    sampled_uids   = user_df["UserID"].to_numpy(dtype=np.int64)

    gf_cols  = fd["group_fts"]
    num_cols = [c for c in gf_cols if c in _NUMERIC_GROUP_FTS]
    dum_cols = [c for c in gf_cols if c not in _NUMERIC_GROUP_FTS]

    pred_matrix = np.empty((n_users_actual, n_scenarios), dtype=np.float32)

    if verbose:
        print(f"  {n_users_actual} users × {n_scenarios} scenarios "
              f"— running model inference …")
        t0 = time.time()

    with torch.no_grad():
        for u_idx, row in user_df.iterrows():
            uid = int(row["UserID"])
            # Group features for this user (repeated for every scenario)
            g_num = row[num_cols].to_numpy(dtype=np.float32)
            g_dum = row[dum_cols].to_numpy(dtype=np.float32) if dum_cols else np.array([], dtype=np.float32)
            g_vec = np.concatenate([g_num, g_dum])            # (n_group,)
            g_t   = torch.tensor(g_vec, dtype=torch.float32).unsqueeze(0).to(device)  # (1, n_group)

            uid_t = torch.tensor([uid], dtype=torch.long).to(device)

            # Process in sub-batches over scenarios
            probs: list[np.ndarray] = []
            for s_start in range(0, n_scenarios, batch_size):
                s_end  = min(s_start + batch_size, n_scenarios)
                B      = s_end - s_start
                r_t    = scen_resp_t[s_start:s_end]                          # (B, n_resp)
                g_b    = g_t.expand(B, -1)                                   # (B, n_group)
                uid_b  = uid_t.expand(B)                                     # (B,)
                logits = model(r_t, uid_b, g_b).squeeze(-1)                  # (B,)
                probs.append(torch.sigmoid(logits).cpu().numpy())

            pred_matrix[u_idx] = np.concatenate(probs)

            if verbose and (u_idx + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  … {u_idx + 1}/{n_users_actual} users  "
                      f"({elapsed:.1f}s elapsed)")

    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s. "
              f"Matrix shape: {pred_matrix.shape}")

    return pred_matrix, sampled_uids, scenario_ids


# ---------------------------------------------------------------------------
# Jury evaluation helpers
# ---------------------------------------------------------------------------

def _jury_agreement(
    pred_matrix: np.ndarray,
    jury_indices: np.ndarray,
    llm_vec: np.ndarray,
) -> float:
    """Fraction of scenarios where jury majority vote matches LLM."""
    votes       = pred_matrix[jury_indices].mean(axis=0)   # (S,)
    jury_dec    = (votes >= 0.5).astype(np.int8)            # (S,)
    return float((jury_dec == llm_vec).mean())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class JuryResult:
    """Outcome of one jury-selection run.

    Attributes
    ----------
    jury_indices   : Indices into *sampled_uids* (rows of pred_matrix).
    jury_user_ids  : Actual UserIDs of selected jurors.
    agreement      : Fraction of scenarios where majority vote matches LLM.
    approach       : "evolutionary" | "ml_greedy" | "ml_l1"
    jury_size      : K (number of jurors).
    scenario_ids   : Ordered scenario IDs (columns of pred_matrix).
    per_scenario   : Per-scenario 0/1 array (1 = jury agrees with LLM).
    history        : Training curve (e.g., best fitness per generation for EA).
    """

    jury_indices:  np.ndarray
    jury_user_ids: np.ndarray
    agreement:     float
    approach:      str
    jury_size:     int
    scenario_ids:  np.ndarray
    per_scenario:  np.ndarray = field(default_factory=lambda: np.array([]))
    history:       list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.approach}] jury_size={self.jury_size}  "
            f"agreement={self.agreement:.4f}  "
            f"user_ids={self.jury_user_ids.tolist()}"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# 1. Evolutionary Algorithm
# ---------------------------------------------------------------------------

def find_jury_evolutionary(
    pred_matrix:  np.ndarray,
    llm_choices:  dict[int, int],
    scenario_ids: np.ndarray,
    *,
    jury_size:       int   = 12,
    pop_size:        int   = 200,
    n_generations:   int   = 500,
    mutation_rate:   float = 0.15,
    tournament_k:    int   = 5,
    elite_frac:      float = 0.05,
    seed:            int   = 0,
    sampled_uids:    Optional[np.ndarray] = None,
    verbose:         bool  = True,
) -> JuryResult:
    """Find the best jury using a genetic algorithm.

    Chromosome: array of *jury_size* distinct indices into pred_matrix rows.
    Fitness: fraction of scenarios where jury majority vote matches LLM.

    Parameters
    ----------
    pred_matrix  : (n_users, n_scenarios) float32 — P(swerve) from DCN.
    llm_choices  : {scenario_id: 0/1} dict.
    scenario_ids : Ordered scenario IDs (same order as pred_matrix columns).
    jury_size    : K — how many jurors (configurable by moderator).
    pop_size     : EA population size.
    n_generations: Number of generations.
    mutation_rate: Fraction of jury slots randomly swapped per mutation.
    tournament_k : Tournament size for parent selection.
    elite_frac   : Fraction of population kept unchanged (elitism).
    seed         : Random seed.
    sampled_uids : UserIDs matching pred_matrix rows (for result annotation).
    verbose      : Print progress every 50 generations.
    """
    rng = np.random.default_rng(seed)
    N, S = pred_matrix.shape

    # Build LLM choice vector aligned to scenario_ids order
    llm_vec = np.array([llm_choices[int(sid)] for sid in scenario_ids], dtype=np.int8)

    # Precompute binary decisions for each user (avoids repeated thresholding)
    user_dec = (pred_matrix >= 0.5).astype(np.int8)   # (N, S)

    def fitness(idx: np.ndarray) -> float:
        votes = user_dec[idx].mean(axis=0)             # (S,)
        return float(((votes >= 0.5).astype(np.int8) == llm_vec).mean())

    def tournament(pop: list, fits: list) -> np.ndarray:
        k    = min(tournament_k, len(pop))
        cands = rng.choice(len(pop), k, replace=False)
        best  = cands[np.argmax([fits[c] for c in cands])]
        return pop[best].copy()

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Union of both parents; keep jury_size unique indices at random."""
        pool   = np.unique(np.concatenate([p1, p2]))
        if len(pool) >= jury_size:
            return rng.choice(pool, jury_size, replace=False)
        # If pool too small, pad with random new users
        extra = np.setdiff1d(np.arange(N), pool)
        pad   = rng.choice(extra, jury_size - len(pool), replace=False)
        return np.concatenate([pool, pad])

    def mutate(ind: np.ndarray) -> np.ndarray:
        n_swap = max(1, int(round(jury_size * mutation_rate)))
        out    = ind.copy()
        in_set = set(out.tolist())
        all_u  = np.arange(N)
        for _ in range(n_swap):
            pos      = rng.integers(jury_size)
            old_uid  = out[pos]
            candidates = np.setdiff1d(all_u, list(in_set))
            if len(candidates) == 0:
                break
            new_uid  = rng.choice(candidates)
            in_set.discard(old_uid)
            in_set.add(new_uid)
            out[pos] = new_uid
        return out

    # Initialise population
    population = [rng.choice(N, jury_size, replace=False) for _ in range(pop_size)]
    n_elite    = max(1, int(pop_size * elite_frac))
    history: list[float] = []

    if verbose:
        print(f"EA: pop={pop_size}, generations={n_generations}, "
              f"jury_size={jury_size}")

    for gen in range(n_generations):
        fits = [fitness(ind) for ind in population]
        ranked = sorted(zip(fits, range(len(population))), reverse=True)
        best_fit = ranked[0][0]
        history.append(best_fit)

        if verbose and gen % 50 == 0:
            print(f"  Gen {gen:>4d}/{n_generations}  best={best_fit:.4f}")

        # Build next generation
        elites   = [population[ranked[i][1]].copy() for i in range(n_elite)]
        children = elites[:]
        while len(children) < pop_size:
            p1 = tournament(population, fits)
            p2 = tournament(population, fits)
            child = crossover(p1, p2)
            child = mutate(child)
            children.append(child)

        population = children

    # Final evaluation
    fits = [fitness(ind) for ind in population]
    best_idx = int(np.argmax(fits))
    best_ind = population[best_idx]

    votes      = user_dec[best_ind].mean(axis=0)
    per_scene  = ((votes >= 0.5).astype(np.int8) == llm_vec).astype(np.int8)
    uids       = sampled_uids[best_ind] if sampled_uids is not None else best_ind

    if verbose:
        print(f"EA complete. Best agreement = {fits[best_idx]:.4f}")

    return JuryResult(
        jury_indices  = best_ind,
        jury_user_ids = uids,
        agreement     = fits[best_idx],
        approach      = "evolutionary",
        jury_size     = jury_size,
        scenario_ids  = scenario_ids,
        per_scenario  = per_scene,
        history       = history,
    )


# ---------------------------------------------------------------------------
# 2. ML Approach — Greedy Forward Selection + L1 Logistic Regression
# ---------------------------------------------------------------------------

def find_jury_ml(
    pred_matrix:  np.ndarray,
    llm_choices:  dict[int, int],
    scenario_ids: np.ndarray,
    *,
    jury_size:       int   = 12,
    sampled_uids:    Optional[np.ndarray] = None,
    l1_C:           float = 0.1,
    verbose:         bool  = True,
) -> tuple[JuryResult, JuryResult]:
    """Find the best jury using two ML strategies.

    **Strategy A — Greedy forward selection**
    Iteratively adds the user whose inclusion maximally improves jury agreement.
    O(K × N × S) — fast and interpretable.

    **Strategy B — L1 logistic regression (soft jury)**
    Treats user predictions as features: X[s, u] = P(user_u swerves | scenario_s).
    Fits a sparse logistic regression predicting LLM's choice, then selects
    the K users with the largest |coefficient|.

    Parameters
    ----------
    pred_matrix  : (n_users, n_scenarios) float32.
    llm_choices  : {scenario_id: 0/1}.
    scenario_ids : Ordered scenario IDs.
    jury_size    : K.
    l1_C         : Inverse regularisation for L1 logistic regression (smaller → sparser).
    sampled_uids : UserIDs matching pred_matrix rows.
    verbose      : Print progress.

    Returns
    -------
    (greedy_result, l1_result) — two JuryResult objects.
    """
    N, S     = pred_matrix.shape
    llm_vec  = np.array([llm_choices[int(sid)] for sid in scenario_ids], dtype=np.int8)
    user_dec = (pred_matrix >= 0.5).astype(np.int8)   # (N, S) binary

    def jury_agreement(indices: list[int]) -> float:
        if not indices:
            return 0.0
        votes = user_dec[indices].mean(axis=0)
        return float(((votes >= 0.5).astype(np.int8) == llm_vec).mean())

    # ------------------------------------------------------------------
    # Strategy A: Greedy forward selection
    # ------------------------------------------------------------------
    if verbose:
        print(f"ML Greedy: selecting {jury_size} users from {N} candidates …")

    selected: list[int] = []
    remaining = list(range(N))

    for step in range(jury_size):
        best_u, best_gain = -1, -1.0
        for u in remaining:
            gain = jury_agreement(selected + [u])
            if gain > best_gain:
                best_gain = gain
                best_u    = u
        selected.append(best_u)
        remaining.remove(best_u)
        if verbose:
            print(f"  Step {step + 1}/{jury_size}: added user_idx={best_u}  "
                  f"agreement={best_gain:.4f}")

    g_votes    = user_dec[selected].mean(axis=0)
    g_per_scen = ((g_votes >= 0.5).astype(np.int8) == llm_vec).astype(np.int8)
    g_uids     = sampled_uids[selected] if sampled_uids is not None else np.array(selected)

    greedy_result = JuryResult(
        jury_indices  = np.array(selected),
        jury_user_ids = g_uids,
        agreement     = jury_agreement(selected),
        approach      = "ml_greedy",
        jury_size     = jury_size,
        scenario_ids  = scenario_ids,
        per_scenario  = g_per_scen,
    )

    # ------------------------------------------------------------------
    # Strategy B: L1 Logistic Regression
    # ------------------------------------------------------------------
    if verbose:
        print(f"\nML L1-LogReg: fitting sparse logistic regression (C={l1_C}) …")

    from sklearn.linear_model import LogisticRegression as _LR

    # Feature matrix: rows = scenarios, cols = user predictions
    X = pred_matrix.T.astype(np.float32)   # (S, N)
    y = llm_vec.astype(int)

    clf = _LR(
        penalty="l1", solver="liblinear", C=l1_C,
        max_iter=2000, random_state=0,
    )
    clf.fit(X, y)
    coef     = clf.coef_[0]                        # (N,)
    top_k    = np.argsort(np.abs(coef))[-jury_size:]  # largest magnitude

    l1_votes    = user_dec[top_k].mean(axis=0)
    l1_per_scen = ((l1_votes >= 0.5).astype(np.int8) == llm_vec).astype(np.int8)
    l1_uids     = sampled_uids[top_k] if sampled_uids is not None else top_k

    l1_agreement = float(((l1_votes >= 0.5).astype(np.int8) == llm_vec).mean())

    if verbose:
        print(f"  L1 agreement = {l1_agreement:.4f}")

    l1_result = JuryResult(
        jury_indices  = top_k,
        jury_user_ids = l1_uids,
        agreement     = l1_agreement,
        approach      = "ml_l1",
        jury_size     = jury_size,
        scenario_ids  = scenario_ids,
        per_scenario  = l1_per_scen,
    )

    return greedy_result, l1_result


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def analyze_jury_demographics(
    jury_result: JuryResult,
    user_df: pd.DataFrame,
    sampled_uids: np.ndarray,
) -> pd.DataFrame:
    """Summarise demographic features of selected jurors vs. the full sample.

    Parameters
    ----------
    user_df      : DataFrame of sampled users (one row per user, from _sample_users).
    sampled_uids : UserIDs in the same order as pred_matrix rows.

    Returns a DataFrame comparing juror demographics to the full sample mean.
    """
    uid_to_row = {int(uid): i for i, uid in enumerate(sampled_uids)}
    juror_rows = [uid_to_row[int(uid)] for uid in jury_result.jury_user_ids
                  if int(uid) in uid_to_row]
    juror_df   = user_df.iloc[juror_rows]

    numeric_cols = [c for c in _NUMERIC_GROUP_FTS if c in user_df.columns]
    country_cols = [c for c in user_df.columns if c.startswith("Cnt_")]
    gender_cols  = [c for c in user_df.columns if c.startswith("Gen_")]

    rows = []
    for col in numeric_cols:
        rows.append({
            "feature":       col,
            "juror_mean":    juror_df[col].mean(),
            "population_mean": user_df[col].mean(),
            "delta":         juror_df[col].mean() - user_df[col].mean(),
        })

    for col in gender_cols:
        rows.append({
            "feature":       col.replace("Gen_", "gender="),
            "juror_mean":    juror_df[col].mean(),
            "population_mean": user_df[col].mean(),
            "delta":         juror_df[col].mean() - user_df[col].mean(),
        })

    # Top-5 over-represented countries
    country_deltas = []
    for col in country_cols:
        d = juror_df[col].mean() - user_df[col].mean()
        country_deltas.append((col.replace("Cnt_", "country="), d,
                               juror_df[col].mean(), user_df[col].mean()))
    country_deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, delta, jm, pm in country_deltas[:10]:
        rows.append({"feature": name, "juror_mean": jm,
                     "population_mean": pm, "delta": delta})

    return pd.DataFrame(rows).set_index("feature")


def compare_results(
    results: list[JuryResult],
    llm_choices: dict[int, int],
    scenario_ids: np.ndarray,
) -> pd.DataFrame:
    """Side-by-side comparison table for multiple JuryResult objects."""
    llm_vec = np.array([llm_choices[int(s)] for s in scenario_ids], dtype=np.int8)
    rows = []
    for r in results:
        rows.append({
            "approach":    r.approach,
            "jury_size":   r.jury_size,
            "agreement":   r.agreement,
            "llm_swerve_rate": float(llm_vec.mean()),
            "jury_swerve_rate": float(
                (r.per_scenario.sum() / len(r.per_scenario)) if len(r.per_scenario) else float("nan")
            ),
        })
    return pd.DataFrame(rows).set_index("approach")


def plot_ea_history(result: JuryResult) -> None:
    """Plot fitness over EA generations (requires matplotlib)."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3))
    plt.plot(result.history)
    plt.xlabel("Generation")
    plt.ylabel("Agreement with LLM")
    plt.title(f"EA convergence — jury_size={result.jury_size}")
    plt.tight_layout()
    plt.show()


def plot_agreement_distribution(results: list[JuryResult]) -> None:
    """Bar chart: per-scenario agreement for each approach."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        ax.bar(["agree", "disagree"],
               [(r.per_scenario == 1).mean(), (r.per_scenario == 0).mean()])
        ax.set_title(f"{r.approach}\nagreement={r.agreement:.3f}")
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
