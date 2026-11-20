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

Same 5 folds (`cv_folds(47, 5, seed=0)`), each fold fitted from scratch (popsize 12, maxiter 60, 2 restarts, no `x0`). "warm" is the stored leaderboard value, every fold warm-started from the full-corpus fit. Train NLL is summed over the 5 folds' training sets (each condition counted 4 times): a cold train NLL above the warm one means the cold optimiser stopped short of the warm solution, so its held-out NLL also reflects optimiser error, not only the removal of leakage.

| label | k | held-out NLL warm (stored) | held-out NLL cold | cold - warm | train NLL warm | train NLL cold | cold - warm | held-out wSSE cold | max fold-parameter shift, cold vs warm (% of range) |
|---|---|---|---|---|---|---|---|---|---|
| xsl_14 | 17 | 16880.2 | 16908.1 | +27.9 | 67370.3 | 67386.0 | +15.6 | 0.3831 | 50 |
| xsl_13 | 18 | 16886.6 | 16902.4 | +15.8 | 67384.4 | 67392.8 | +8.4 | 0.3789 | 39 |
| xsl_12 | 16 | 16905.5 | 16906.6 | +1.1 | 67392.3 | 67391.6 | -0.7 | 0.3893 | 29 |
| xsl_11 | 16 | 16913.6 | 16916.2 | +2.6 | 67420.8 | 67426.7 | +5.9 | 0.3910 | 50 |
| xsl_09 | 15 | 16921.7 | 16927.8 | +6.1 | 67425.9 | 67433.6 | +7.6 | 0.3996 | 69 |

Gap to `xsl_14` in held-out NLL (positive: worse than `xsl_14`):

| label | warm (stored) | cold |
|---|---|---|
| xsl_13 | +6.5 | -5.7 |
| xsl_12 | +25.4 | -1.4 |
| xsl_11 | +33.4 | +8.1 |
| xsl_09 | +41.6 | +19.8 |

## 3. Grouped CV: experiment families kept whole

Folds from `cv_folds(47, 5, seed=0, groups=[experiment_family(c.label) ...])`, fitted cold with the same budget, so compare with the cold column of table 2. Held-out conditions per fold:

- fold 0: filt0E_3L, filt3E_3L, filt6E_3L, filt9E_3L, filt0E_6L, filt3E_6L, filt6E_6L, filt9E_6L, filt0E_9L, filt3E_9L, filt6E_9L, filt9E_9L
- fold 1: 206, 213, 216, 2_x8_39_4x4, 3_x8_369_4x4, freq369-3x3loCD, freq369-3x3hiCD, freq369_36mx, freq369_39mx
- fold 2: 202, 212, 221, 222, 1_max_temp_spat_cont_orig, 4_no_spat_orig_max_tc, 1olap3tr_contr, temp_cont_1olap2tr_contr, temp_cont_2olap2tr_contr
- fold 3: 201, 203, 204, 207, 214, 217, 218, 219, 223
- fold 4: 205, 211, 215, 220, 224, 225, 301, cont_div6-12

| label | k | held-out NLL, random folds (cold) | held-out NLL, grouped folds (cold) | grouped - random | held-out wSSE grouped | held-out NLL per grouped fold (0-4) |
|---|---|---|---|---|---|---|
| xsl_14 | 17 | 16908.1 | 17000.7 | +92.6 | 0.4172 | 4224.9, 4065.5, 2949.9, 2628.1, 3132.3 |
| xsl_13 | 18 | 16902.4 | 16993.8 | +91.4 | 0.4140 | 4222.1, 4062.4, 2950.3, 2628.9, 3130.2 |
| xsl_12 | 16 | 16906.6 | 16978.4 | +71.7 | 0.4097 | 4205.7, 4056.8, 2949.4, 2630.5, 3136.0 |
| xsl_11 | 16 | 16916.2 | 17007.9 | +91.7 | 0.4065 | 4210.0, 4083.5, 2952.1, 2631.9, 3130.4 |
| xsl_09 | 15 | 16927.8 | 17002.5 | +74.7 | 0.4100 | 4211.4, 4075.0, 2953.6, 2633.5, 3129.0 |

Gap to `xsl_14` in held-out NLL:

| label | random folds (cold) | grouped folds (cold) |
|---|---|---|
| xsl_13 | -5.7 | -6.8 |
| xsl_12 | -1.4 | -22.3 |
| xsl_11 | +8.1 | +7.2 |
| xsl_09 | +19.8 | +1.9 |

Held-out NLL under grouped folds, summed by family:

| family | n conditions | xsl_14 | xsl_13 | xsl_12 | xsl_11 | xsl_09 |
|---|---|---|---|---|---|---|
| single-condition families | 24 | 8000.1 | 8000.2 | 8007.1 | 8007.2 | 8007.4 |
| filt | 12 | 4224.9 | 4222.1 | 4205.7 | 4210.0 | 4211.4 |
| freq369 | 6 | 3204.9 | 3202.3 | 3195.9 | 3221.1 | 3213.5 |
| temporal_contiguity | 5 | 1570.7 | 1569.3 | 1569.7 | 1569.6 | 1570.2 |
