# Do the leaderboard order or the gaps to the incumbents change?

Tables are in [comparison.md](comparison.md). Four theories were refitted (xsl_14, xsl_09,
kachergis, inc_fazly); the other 15 only have the wSSE recomputation.

**Short answer.** The generated theories' lead over the incumbents holds in every variant
(at least 205 NLL). The order inside the top cluster does not: xsl_14 vs xsl_09 is within
optimiser and split noise, and reverses under grouped folds. The order of the two incumbents
also depends on the variant.

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
- xsl_09 now beats xsl_14 by 25.2 (it is better on filt and freq369; the two are equal on
  temporal contiguity and on the single-condition families, where xsl_14 leads by 5). xsl_14's
  extra parameters over xsl_09 do not generalise across families at this fitting budget.
- Gaps to xsl_14: kachergis 253 -> 205, fazly 239 -> 411. kachergis is clearly ahead of fazly
  (206 NLL), the reverse of the cold random-fold order.

## Caveats

- One fold seed and one grouped split; the 12-condition filt fold dominates the grouped score.
- The cold budget (popsize 10, maxiter 40, 2 restarts) is too small for the 15-17 parameter
  theories to reach the warm training NLL reliably, so differences of 10-35 NLL among xsl_NN
  theories are not interpretable here. A cold run of xsl_14 (and xsl_13/12/11) with the
  robust-refit budget would settle whether xsl_14 > xsl_09 survives.
- `2_x8_39_4x4` and `3_x8_369_4x4` were put in the freq369 family; 2xx conditions are each
  their own family (no sub-experiment grouping).
- The stored leaderboard files and the docs still show the old wSSE; they were not rewritten.
