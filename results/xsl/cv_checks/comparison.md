# XSL scoring checks

Source leaderboard: `results/xsl/debate_xsl_12_vs_inc_fazly_gated`. Written by `scripts/xsl/cv_checks.py`; see `note.md` for the reading.

## 1. Weighted SSE: median-person vs population-mean prediction

Recomputed from the stored full-fit and fold parameters (no refitting). "before" compares the accuracy of a median-ability person with the human mean; "after" compares the population-mean accuracy (`population_accuracy`) with it. The "before" columns and the held-out NLL were also recomputed from the stored parameters and match the stored values. NLL is unaffected. Rows are in leaderboard (held-out NLL) order; (n) is the rank on that column.

| label | k | ability_sd | held-out NLL | full wSSE before | full wSSE after | held-out wSSE before | held-out wSSE after |
|---|---|---|---|---|---|---|---|
| xsl_14 | 17 | 1.436 | 16880.2 | 0.3696 (1) | 0.3382 (1) | 0.4002 (1) | 0.3630 (1) |
| xsl_13 | 18 | 1.441 | 16886.6 | 0.3767 (2) | 0.3456 (2) | 0.4110 (2) | 0.3727 (2) |
| xsl_12 | 16 | 1.439 | 16905.5 | 0.3866 (6) | 0.3507 (4) | 0.4364 (6) | 0.3884 (3) |
| xsl_11 | 16 | 1.434 | 16913.6 | 0.3807 (5) | 0.3484 (3) | 0.4306 (3) | 0.3898 (4) |
| xsl_09 | 15 | 1.437 | 16921.7 | 0.3793 (4) | 0.3528 (6) | 0.4333 (5) | 0.3941 (5) |
| xsl_10 | 15 | 1.437 | 16922.0 | 0.3793 (3) | 0.3527 (5) | 0.4331 (4) | 0.3941 (6) |
| xsl_04 | 9 | 1.465 | 16991.3 | 0.4424 (9) | 0.4138 (11) | 0.4936 (8) | 0.4504 (9) |
| xsl_05 | 10 | 1.481 | 16995.9 | 0.4483 (10) | 0.4111 (8) | 0.4968 (9) | 0.4491 (8) |
| xsl_06 | 11 | 1.475 | 16998.6 | 0.4512 (11) | 0.4137 (10) | 0.4933 (7) | 0.4461 (7) |
| xsl_03 | 7 | 1.492 | 17050.6 | 0.5310 (13) | 0.4666 (12) | 0.5456 (13) | 0.4834 (12) |
| xsl_08 | 13 | 1.464 | 17061.6 | 0.4285 (8) | 0.4136 (9) | 0.5091 (11) | 0.4638 (10) |
| xsl_07 | 11 | 1.461 | 17065.7 | 0.4224 (7) | 0.4008 (7) | 0.5168 (12) | 0.4712 (11) |
| kachergis | 5 | 1.559 | 17135.1 | 0.6372 (14) | 0.5268 (14) | 0.6886 (14) | 0.5642 (15) |
| inc_fazly | 4 | 1.446 | 17153.6 | 0.4807 (12) | 0.4882 (13) | 0.5039 (10) | 0.5046 (13) |
| inc_uncfam_attention | 5 | 1.562 | 17170.3 | 0.6470 (15) | 0.5311 (15) | 0.6964 (15) | 0.5641 (14) |
| inc_uncfam_gamma | 6 | 1.595 | 17177.2 | 0.6659 (16) | 0.5561 (16) | 0.7072 (17) | 0.5872 (16) |
| inc_tilles | 5 | 1.548 | 17357.2 | 0.6785 (17) | 0.6307 (17) | 0.7003 (16) | 0.6507 (17) |
| inc_bayesian_decay | 5 | 1.552 | 17657.3 | 0.8370 (18) | 0.6904 (18) | 0.8615 (18) | 0.7143 (18) |
| baseline | 2 | 2.874 | 19568.7 | 1.3383 (19) | 1.1486 (19) | 1.3742 (19) | 1.1850 (19) |

## 2. Cold-start CV on the leaderboard's folds

Same 5 folds (`cv_folds(47, 5, seed=0)`), each fold fitted from scratch (popsize 10, maxiter 40, 2 restarts, no `x0`). "warm" is the stored leaderboard value, every fold warm-started from the full-corpus fit. Train NLL is summed over the 5 folds' training sets (each condition counted 4 times): a cold train NLL above the warm one means the cold optimiser stopped short of the warm solution, so its held-out NLL also reflects optimiser error, not only the removal of leakage.

| label | k | held-out NLL warm (stored) | held-out NLL cold | cold - warm | train NLL warm | train NLL cold | cold - warm | held-out wSSE cold | max fold-parameter shift, cold vs warm (% of range) |
|---|---|---|---|---|---|---|---|---|---|
| xsl_14 | 17 | 16880.2 | 16914.6 | +34.4 | 67370.3 | 67401.9 | +31.6 | 0.3807 | 83 |
| xsl_09 | 15 | 16921.7 | 16929.9 | +8.2 | 67425.9 | 67426.0 | +0.1 | 0.3989 | 79 |
| kachergis | 5 | 17135.1 | 17167.9 | +32.8 | 68388.3 | 68384.0 | -4.3 | 0.5663 | 62 |
| inc_fazly | 4 | 17153.6 | 17153.8 | +0.2 | 68508.8 | 68508.7 | -0.1 | 0.5052 | 0 |

Gap to `xsl_14` in held-out NLL (positive: worse than `xsl_14`):

| label | warm (stored) | cold |
|---|---|---|
| xsl_09 | +41.6 | +15.3 |
| kachergis | +255.0 | +253.3 |
| inc_fazly | +273.5 | +239.3 |

## 3. Grouped CV: experiment families kept whole

Folds from `cv_folds(47, 5, seed=0, groups=[experiment_family(c.label) ...])`, fitted cold with the same budget, so compare with the cold column of table 2. Held-out conditions per fold:

- fold 0: filt0E_3L, filt3E_3L, filt6E_3L, filt9E_3L, filt0E_6L, filt3E_6L, filt6E_6L, filt9E_6L, filt0E_9L, filt3E_9L, filt6E_9L, filt9E_9L
- fold 1: 206, 213, 216, 2_x8_39_4x4, 3_x8_369_4x4, freq369-3x3loCD, freq369-3x3hiCD, freq369_36mx, freq369_39mx
- fold 2: 202, 212, 221, 222, 1_max_temp_spat_cont_orig, 4_no_spat_orig_max_tc, 1olap3tr_contr, temp_cont_1olap2tr_contr, temp_cont_2olap2tr_contr
- fold 3: 201, 203, 204, 207, 214, 217, 218, 219, 223
- fold 4: 205, 211, 215, 220, 224, 225, 301, cont_div6-12

| label | k | held-out NLL, random folds (cold) | held-out NLL, grouped folds (cold) | grouped - random | held-out wSSE grouped | held-out NLL per grouped fold (0-4) |
|---|---|---|---|---|---|---|
| xsl_14 | 17 | 16914.6 | 17008.5 | +93.9 | 0.4209 | 4226.6, 4071.9, 2946.2, 2630.0, 3133.7 |
| xsl_09 | 15 | 16929.9 | 16983.3 | +53.4 | 0.4032 | 4212.0, 4055.0, 2953.4, 2633.2, 3129.7 |
| kachergis | 5 | 17167.9 | 17213.7 | +45.8 | 0.5748 | 4294.7, 4086.6, 3017.5, 2648.7, 3166.2 |
| inc_fazly | 4 | 17153.8 | 17419.6 | +265.7 | 0.5617 | 4570.7, 4036.9, 2985.1, 2666.6, 3160.3 |

Gap to `xsl_14` in held-out NLL:

| label | random folds (cold) | grouped folds (cold) |
|---|---|---|
| xsl_09 | +15.3 | -25.2 |
| kachergis | +253.3 | +205.2 |
| inc_fazly | +239.3 | +411.1 |

Held-out NLL under grouped folds, summed by family:

| family | n conditions | xsl_14 | xsl_09 | kachergis | inc_fazly |
|---|---|---|---|---|---|
| single-condition families | 24 | 8002.1 | 8007.5 | 8138.8 | 8114.5 |
| filt | 12 | 4226.6 | 4212.0 | 4294.7 | 4570.7 |
| freq369 | 6 | 3209.7 | 3193.9 | 3202.9 | 3147.8 |
| temporal_contiguity | 5 | 1570.1 | 1569.9 | 1577.3 | 1586.6 |
