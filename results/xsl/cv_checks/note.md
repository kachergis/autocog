# Do the leaderboard order or the gaps to the incumbents change?

Tables are in [comparison.md](comparison.md). Four theories were refitted (xsl_14, xsl_09,
kachergis, inc_fazly); the other 15 only have the wSSE recomputation.

**Short answer.** The generated theories' lead over the incumbents holds in every variant
(at least 205 NLL). The order inside the top cluster (xsl_09 to xsl_14) does not: a
larger-budget rerun ([../cv_checks_robust/comparison.md](../cv_checks_robust/comparison.md),
see the last section) puts all five within 30 NLL under every split, and their order changes
with the split and the optimiser budget. The order of the two incumbents also depends on the
variant.

| held-out NLL | warm, random folds (stored) | cold, random folds | cold, grouped folds |
|---|---|---|---|
| xsl_14 | 16880.2 | 16914.6 | 17008.5 |
| xsl_09 | 16921.7 | 16929.9 | 16983.3 |
| kachergis | 17135.1 | 17167.9 | 17213.7 |
| inc_fazly | 17153.6 | 17153.8 | 17419.6 |

## 1. wSSE (population mean instead of median person)

- NLL does not use this prediction, so the leaderboard order (held-out NLL) is unchanged.
- wSSE falls for every theory except inc_fazly (full fit .481 -> .488, held-out .504 -> .505).
  kachergis: full fit .637 -> .527, held-out .689 -> .564.
- xsl_14 is still first on both wSSE columns (held-out .400 -> .363).
- inc_fazly's held-out wSSE rank drops from 10 to 13: it had looked better than xsl_07/08/03
  on wSSE only because the median-person prediction penalised the others more. kachergis's
  wSSE is still worse than fazly's (.564 vs .505), but the gap is a third of what was reported
  (.185 -> .060).

## 2. Cold start vs warm start (same folds)

- inc_fazly: no change (+0.2). xsl_09: +8.2 with an identical training NLL.
- kachergis: +32.8, almost all from one fold (fold 3, +30.7), where the cold fit found a
  training optimum 4.4 NLL *better* than the warm one that predicts the held-out conditions
  worse. The warm start had kept that fold in the full-corpus basin, so the stored value was
  optimistic by about 30.
- xsl_14: +34.4, but its cold training NLL is also 31.6 higher than the warm one (folds 2 and
  4 account for 24 of it). So the cold optimiser stopped short of the warm solution, and this
  difference mixes optimiser error with removed leakage; it is an upper bound on the bias, not
  an estimate of it. Cold fold parameters differ from the warm ones by up to ~80% of a
  parameter's range for xsl_14 and xsl_09, so the "~1% shift" under warm starts reflected the
  warm start, not a well-identified optimum.
- Effect on order: xsl_14 stays ahead of xsl_09 but the gap shrinks from 41.6 to 15.3, which
  is no larger than xsl_14's own optimiser shortfall. kachergis and inc_fazly swap (fazly now
  14 ahead). Gaps to xsl_14: kachergis 255 -> 253, fazly 274 -> 239.

## 3. Grouped folds (families held out whole, cold)

- Every theory is worse when a family's siblings are not in training: xsl_14 +94, xsl_09 +53,
  kachergis +46, inc_fazly +266.
- inc_fazly's loss is almost entirely the filtering family (4570.7, against 4212-4295 for the
  others): its fit to the `filt*` conditions depended on having other `filt*` conditions in
  training. On the freq369 family fazly is the best of the four.
- At this budget xsl_09 beats xsl_14 by 25.2, but that does not survive the larger-budget
  rerun (below), where xsl_14 is 1.9 ahead.
- Gaps to xsl_14: kachergis 253 -> 205, fazly 239 -> 411. kachergis is clearly ahead of fazly
  (206 NLL), the reverse of the cold random-fold order.

## Caveats

- One fold seed and one grouped split; the 12-condition filt fold dominates the grouped score.
- The cold budget (popsize 10, maxiter 40, 2 restarts) is too small for the 15-17 parameter
  theories to reach the warm training NLL reliably, so differences of 10-35 NLL among xsl_NN
  theories are not interpretable here; see the larger-budget rerun below.
- `2_x8_39_4x4` and `3_x8_369_4x4` were put in the freq369 family; 2xx conditions are each
  their own family (no sub-experiment grouping).
- The stored leaderboard files and the docs still show the old wSSE; they were not rewritten.

## 4. Larger-budget cold CV of the top cluster

[../cv_checks_robust/comparison.md](../cv_checks_robust/comparison.md): xsl_14, xsl_13, xsl_12,
xsl_11 and xsl_09, cold, on both splits, popsize 12, maxiter 60, 2 restarts (the robust-refit
budget per fold), about 61 CPU-hours in all.

| held-out NLL | warm, random (stored) | cold, random | cold, grouped |
|---|---|---|---|
| xsl_14 | 16880.2 | 16908.1 | 17000.7 |
| xsl_13 | 16886.6 | 16902.4 | 16993.8 |
| xsl_12 | 16905.5 | 16906.6 | 16978.4 |
| xsl_11 | 16913.6 | 16916.2 | 17007.9 |
| xsl_09 | 16921.7 | 16927.8 | 17002.5 |

- The bigger budget closes half of xsl_14's training shortfall (+31.6 -> +15.6 over the warm
  fits) and its held-out NLL improves only 6.5 (16914.6 -> 16908.1). The cold - warm held-out
  gap is now +27.9 for xsl_14 and +15.8 for xsl_13 but only +1 to +6 for xsl_09/11/12. So the
  warm start flattered the two newest theories most: their 6.5 and 25-41 NLL leads over
  xsl_12/11/09 on the stored leaderboard are mostly the warm start.
- Cold, on random folds, the five span 25 NLL and xsl_13 (-5.7) and xsl_12 (-1.4) are ahead of
  xsl_14. On grouped folds they span 30 NLL and xsl_12 is first (22 ahead of xsl_14), mainly on
  the filt family; xsl_09 is now 1.9 *behind* xsl_14, where the smaller budget had it 25 ahead,
  even though its training NLL barely moved (67414 -> 67411). Fold-level optimiser noise of
  ~20 NLL is the same size as the gaps.
- Reading: the leaderboard cannot rank xsl_09 through xsl_14 against each other; their
  differences are within warm-start bias, optimiser noise and split choice. The debate's
  later revisions (xsl_13, xsl_14) did not produce a measurable held-out gain over xsl_12.
  Their lead over the incumbents (>= 205 NLL against kachergis, >= 239 against inc_fazly) is not
  in question.
