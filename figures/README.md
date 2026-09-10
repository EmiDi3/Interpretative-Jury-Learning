# Figures

Exploratory jury-demographics output from Section 5 of `00_run_all.ipynb`.

| File | Content |
|---|---|
| `jury_demographics_phi4.png` | Juror vs population demographic means and deltas, Phi-4 jury |
| `jury_demographics_qwen2.5_14b.png` | Same, Qwen2.5-14B jury |
| `jury_demographics_phi4_vs_qwen.png` | Side-by-side Phi-4 vs Qwen jury, plus country representation |

## Caveats — read before citing these

These were generated on **2026-05-26 ~02:00**, which is **before** commit `06427d6`
("Fix NaN in analyze_jury_demographics", 2026-05-26 08:37). They therefore carry
two known problems:

1. **Pre-fix demographics code.** The population baseline was computed over the
   whole `user_df` rather than the sampled candidate pool, and jurors were looked
   up positionally instead of by `UserID`. The visible symptom is the `nan` cells
   in the Qwen column of `jury_demographics_phi4_vs_qwen.png`.

2. **In-sample agreement.** Jurors were selected and scored on the same scenarios,
   with no held-out split. See `RESEARCH_PLAN.md` §"Two things to fix" — every
   headline number is supposed to come from a scenario-level train/test split.

Treat these as provisional. They are kept for provenance and should be regenerated
once the experiments in `RESEARCH_PLAN.md` (Steps 2–4) are run.
