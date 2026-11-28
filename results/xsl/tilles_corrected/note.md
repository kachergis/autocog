# Corrected tilles vs the old port

The port now follows XSLmodels `b44b9fa` (`20826a50`). `incumbents.json` and the R bridge
use xsl-bakeoff's refit of the corrected model (`e5fd145b`). This directory refits it under
the random-ability likelihood (`robust_refit.py`, full fit popsize 12 / maxiter 60 /
3 restarts) and scores it with the new default gate, cold-started CV on folds grouped by
experiment family (fold budget 12 / 60 / 2). `old_model_cv/` scores the old port the same
way (`cv_checks.py`, fold budget 10 / 40 / 2, the budget the other incumbents had in
[../cv_checks](../cv_checks/comparison.md)).

| | full-corpus NLL | held-out NLL, cold grouped |
|---|---|---|
| tilles, old port | 17,313.6 | 17,488.3 |
| tilles, corrected | 17,293.0 | 17,463.1 |
| inc_fazly | | 17,419.6 |
| kachergis | | 17,213.7 |
| xsl_14 | | 17,008.5 |

Corrected fit: x .143, b .244, alpha_0 .960, ability_sd 1.54, lapse .001 (at its floor).
Held-out wSSE .711.

Held-out NLL by family (grouped folds):

| | filt | freq369 | temporal_contiguity | singletons |
|---|---|---|---|---|
| tilles, old | 4569.3 | 3127.5 | 1592.8 | 8198.6 |
| tilles, corrected | 4565.6 | 3125.9 | 1592.9 | 8178.7 |
| inc_fazly | 4570.7 | 3147.8 | 1586.6 | 8114.5 |
| kachergis | 4294.7 | 3202.9 | 1577.3 | 8138.8 |
| xsl_14 | 4226.6 | 3209.7 | 1570.1 | 8002.1 |

- The correction helps tilles a little: 21 NLL on the full corpus and 25 held out, almost all
  of it on the singleton conditions. That is the size of fold-level optimiser noise, and the
  3-parameter model is easy to fit, so the gain is probably real but small.
- Its place does not change. tilles is still the weakest of these incumbents: 44 behind
  fazly, 249 behind kachergis and 455 behind xsl_14. Its stored leaderboard row (17,357.2,
  warm-started random folds, old model) is replaced by these numbers when the leaderboard is
  re-scored under the new gate.
- Like fazly, tilles is about 270 NLL worse than kachergis on the filt family when that
  family is held out whole, and better than kachergis and xsl_14 on freq369.
- Phase 3's frozen predictions (retroactive reappearance, sparse vs dense, ME vs
  co-occurrence, hub vs private, synonym) were made with the old tilles. They were not
  recomputed.
