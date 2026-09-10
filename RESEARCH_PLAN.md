# Research Plan — Finding a Human Jury that Represents LLM Moral Choices

## Goal

Given an LLM's choices on Moral Machine scenarios, determine:

1. **Can** we find a jury of real Moral Machine participants whose majority vote reproduces the LLM's choices?
2. **How big** does that jury need to be?
3. **What demographics** characterize that jury — and are those conclusions stable?

We answer this with the trained `MoralJuryDCN`, which lets us predict how *any* surveyed human would have voted on *any* scenario, and the jury-search code in `jury_learning/llm_jury.py`.

---

## Two things to fix before any result is trustworthy

**(A) The jury pipeline is not connected.** `llm_jury.py` is implemented but nothing calls it — `00_run_all.ipynb` ends at plots, and `build_prediction_matrix()` raises `RuntimeError` on purpose (you must use `build_prediction_matrix_from_features()` with a `scenario_features_df` from `load_scenario_features()`). Step 0 closes this.

**(B) Agreement is currently measured in-sample.** Jurors are selected to match the LLM *and* scored on the same scenarios. That inflates agreement and makes "bigger jury = better" ambiguous. **Every headline number from Step 2 onward must use a scenario-level train/test split**: select the jury on a *selection* set of scenarios, report agreement on a *held-out* set. In-sample numbers are kept only as a diagnostic.

### On your mentor's question: "shouldn't agreement go up as jurors increase?"

Not necessarily, and the answer is the interesting part of the project:

- **In-sample, a jury of size 1 can look near-perfect** — pick the single human whose predicted votes happen to match the LLM. Adding jurors via majority vote can *dilute* that, so in-sample agreement can be flat or even fall before it stabilizes. This is overfitting, not a real effect.
- **Out-of-sample**, the Condorcet intuition (more jurors → better) only holds if jurors are *independent* and each *individually better than chance* at matching the LLM. Selected jurors are correlated (the DCN clusters similar people) and chosen to fit, so the held-out curve typically **rises then plateaus** — there's a finite "enough" size, which is exactly the "how big" answer we want.

So the jury-size sweep (Step 2), done held-out, is what actually resolves the mentor's question — and the in-sample vs held-out gap is itself a result worth showing.

---

## Where to run each step

- **Repo / Claude Code**: all code changes, refactors, small CPU experiments (e.g. jury search on a cached prediction matrix — the search itself is NumPy and fast).
- **Colab GPU**: anything that trains the DCN or builds the prediction matrix on the full dataset (`sql_subset_size=None`, ~4M rows). Cache the prediction matrix to a `.npz` so the expensive GPU step runs once and all downstream jury experiments run cheaply.

Suggested layout to keep results reproducible: a new `experiments/` package for scripts, and a `results/` dir (gitignored except small summary CSVs/figures) for matrices, per-seed CSVs, and plots.

---

## Step 0 — Wire up and smoke-test the jury pipeline

**Why:** nothing currently runs the jury search end to end; fix the `build_prediction_matrix` gap and get one full pass on a small subset.

**Claude Code prompt:**

> Read `jury_learning/llm_jury.py`, `jury_learning/data.py`, `jury_learning/evaluation.py`, and `jury_learning/pipeline.py` to understand the current interfaces.
>
> Create `experiments/run_jury.py` that runs the LLM-jury pipeline end to end and is importable as functions (no work at import time):
> 1. Build a `RunConfig`; expose CLI/argparse flags for `--llm-file`, `--n-users`, `--jury-size`, `--sql-subset-size`, `--seed`, `--device`, and `--matrix-cache` (path to a `.npz`).
> 2. `prepare_data(cfg)`; load the trained DCN with `evaluation.load_trained_model` (train it first only if no checkpoint exists).
> 3. Load the LLM choices with `load_llm_responses` and scenario features with `load_scenario_features` (pass `unique_scenarios.csv` when present).
> 4. Build the prediction matrix with `build_prediction_matrix_from_features` — NOT `build_prediction_matrix`. If `--matrix-cache` exists, load it; otherwise build and save `pred_matrix`, `sampled_uids`, `scenario_ids`, plus the sampled `user_df` (needed for demographics).
> 5. Run `find_jury_evolutionary` and `find_jury_ml`, print a `compare_results` table, and run `analyze_jury_demographics`.
>
> Then run it on a small smoke-test config (`--sql-subset-size 100000 --n-users 500 --jury-size 12 --llm-file phi4.json`) and confirm it completes and prints non-trivial agreement. Fix any bugs you hit in `llm_jury.py` (the `build_prediction_matrix` RuntimeError path, scenario alignment, demographics indexing). Do not commit large artifacts; add `results/` and `*.npz` to `.gitignore`. Report the agreement numbers and any code changes you made.

---

## Step 1 — Baselines: define what "accurate" means

**Why:** the LLM files look heavily skewed toward "Swerve". If an LLM swerves 85% of the time, a jury that always swerves already scores 0.85 — so raw agreement is meaningless without baselines.

**Claude Code prompt:**

> Add `experiments/baselines.py`. For each LLM file (`phi4.json`, `llama3.1_8b.json`, `qwen2.5_14b.json`) compute and save to `results/baselines.csv`:
> - the LLM's swerve rate (class balance) and the majority-class agreement;
> - the **single best juror** agreement (best row of the prediction matrix);
> - the **random-jury** agreement for several jury sizes (mean ± std over ≥200 random juries each);
> - the **full-population majority** agreement (all sampled users vote).
>
> Reuse the prediction-matrix builder from `experiments/run_jury.py` (load the cached `.npz`). Print a table per LLM and note the gap between the random-jury baseline and the searched jury from Step 0. These baselines are the reference every later result is reported against (report "agreement" AND "lift over random jury of same size").

---

## Step 2 — Jury-size sweep with held-out scenarios (core experiment)

**Why:** directly answers "how big a jury" and resolves the monotonicity question. This is the main figure of the project.

**Claude Code prompt:**

> Add `experiments/jury_size_sweep.py`. Implement a **scenario-level train/test split**: split the scenario IDs into a selection set and a held-out evaluation set (default 70/30, configurable, seeded). Add a helper so jury search runs only on selection-set columns of the prediction matrix, and agreement is then computed separately on selection and held-out columns.
>
> Sweep `jury_size` over `[1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 75, 101]` (odd sizes to avoid ties; clip to available users). For each size and each search method (`evolutionary`, `ml_greedy`, `ml_l1`), record in-sample and held-out agreement. Save tidy results to `results/size_sweep_<llm>.csv` and produce `results/size_sweep_<llm>.png` plotting agreement vs jury size with **two curves (in-sample vs held-out)** per method, plus the random-jury baseline band from Step 1.
>
> Run for all three LLMs on the cached full-dataset matrix. In a short `results/size_sweep_findings.md`, report: the held-out agreement plateau and the smallest jury size that reaches ~95% of it ("how big"), and explicitly discuss whether held-out agreement increases monotonically with size and why it diverges from the in-sample curve.

---

## Step 3 — Robustness across seeds and methods

**Why:** your mentor asked whether repeating the algorithm gives similar results. Quantify run-to-run variance and method agreement.

**Claude Code prompt:**

> Add `experiments/robustness.py`. For a fixed jury size near the Step-2 plateau, repeat jury selection across ≥20 seeds for each method (`evolutionary`, `ml_greedy`, `ml_l1`), using the held-out split from Step 2. The EA seed varies its random search; for greedy/L1 vary the user sample (`--seed` feeding `_sample_users`) so we capture sampling variance too.
>
> Report to `results/robustness_<llm>.csv` and a summary `results/robustness_findings.md`:
> - held-out agreement: mean, std, 95% CI per method;
> - **jury composition stability**: mean pairwise Jaccard overlap of selected `jury_user_ids` across seeds, and how often individual jurors recur;
> - **demographic stability**: std across seeds of the key juror demographic means from `analyze_jury_demographics`.
>
> Conclude whether results are robust (tight CIs, stable demographics) or seed-sensitive, and which search method is most stable. Use `tqdm` and keep runtime reasonable by reusing the cached prediction matrix.

---

## Step 4 — Demographic characterization

**Why:** the third research question — who is on the jury, and is the demographic story consistent across seeds and across LLMs.

**Claude Code prompt:**

> Add `experiments/demographics.py`. At the chosen jury size, aggregate `analyze_jury_demographics` over the Step-3 seeds (so each demographic delta has a mean and CI, not a single noisy run) for each of the three LLMs.
>
> Produce `results/demographics_<llm>.csv` and a comparison figure `results/demographics_compare.png` showing, per LLM, which demographics are consistently over-/under-represented in the jury vs the population (age, education, income, political, religious, gender, top countries). In `results/demographics_findings.md` summarize: which demographic features are stably over/under-represented; whether the three LLMs select demographically different juries (and if so, how); and any caveats (e.g. country dummies dominated by sample composition). Flag any finding that isn't stable across seeds as inconclusive.

---

## Step 5 — Consolidate into a writeup

**Why:** turn results into something presentable to your mentor.

**Claude Code prompt:**

> Create `results/SUMMARY.md` pulling together Steps 1–4: a short methods note (held-out scenario split, baselines, search methods), the headline numbers (can we match the LLM and at what lift over baseline; the "how big" plateau size per LLM; robustness CIs; demographic conclusions), and embed the key figures. Explicitly address the two mentor questions: (1) robustness across repeated runs, with the CI evidence; (2) the agreement-vs-jury-size relationship, contrasting in-sample and held-out behavior and explaining the mechanism. Keep it tight and figure-led.

---

## Open questions to settle as we go

- **`n_users` sensitivity** — does the candidate pool size (2000 default) change which jury is found or its agreement? Worth a small sweep.
- **DCN fidelity ceiling** — jury agreement is bounded by how well the DCN models individuals; report the DCN's own held-out accuracy alongside jury numbers so we don't over-attribute error to the jury.
- **Tie handling** — majority vote with even juries; we use odd sizes, but confirm `_jury_agreement` tie behavior is what we want.
- **Scenario coverage** — each LLM file covers a specific set of scenario_ids; confirm overlap if we ever compare juries head-to-head across LLMs.
