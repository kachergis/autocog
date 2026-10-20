# Prospective design queue

3 accepted design(s); 3 proposals tried.

## ambiguity_at_fixed_word_frequency_78ac7a8725

Frozen 2026-10-20T05:53:00.856573+00:00, sha256 `78ac7a8725bab632…`, commit `e49ec54c`.

**Rationale.** Both arms give every word the same number of exposures (reps per word). They differ only in how many words share a trial. Arm A has 1 word with several objects per trial, so no words compete within a trial. Arm B has 2 or 3 words per trial with their referents plus fillers. Kachergis learns by uncertainty- and familiarity-weighted attention, with almost no decay (C=0.996). Its fit predicts high accuracy for single-word, many-object trials (corpus 1x4: 0.51, 1x3: 0.56), where each word's row narrows by repeated cross-trial intersection. xsl_14 adds load-limited forgetting scaled by objects per trial, a throttle on returns for words with ambiguous rows (kappa<0, so repetition buys little while a word is still uncommitted), and commitment-gated exclusion. Its fit predicts low accuracy for 1x4 (0.29), and its predictions for multi-word trials are comparable or higher (3x3 frequency designs about 0.45-0.5 versus Kachergis about 0.38-0.41). The contrast A minus B therefore has opposite signs under the two theories. It reads as the effect of within-trial word competition versus pure referential ambiguity, with word frequency held equal. The decision is made on a difference of arms, so the between-subject ability offset largely cancels.

**Prediction.** Contrast = mean accuracy in the 1-word-per-trial arm A minus the multi-word arm B at equal word frequency. Kachergis predicts a positive contrast of roughly +0.10 to +0.20: arm A about 0.5 or higher, arm B about 0.35-0.45, because it learns well from repeated single-word ambiguity and is hurt by dense multi-word trials. xsl_14 predicts a zero or negative contrast of roughly -0.05 to -0.20: arm A stays near 0.3 because ambiguous uncommitted words get throttled returns and objects per trial drive forgetting, while arm B benefits from competition between words and commitment-gated exclusion. The decision threshold is the midpoint, near 0. These magnitudes are estimated from corpus fits, not simulated.

**Design.** arm `A`: 18 words × 18 objects, 108 trials, 18 test items; arm `B`: 18 words × 18 objects, 36 trials, 18 test items (knobs {'noA': 4, 'noB': 3, 'nw': 3, 'reps': 6}, seed 1); 50 people per arm.

**Decision rule.** Contrast above -0.0288 favours `kachergis`. Predicted: xsl_14 -0.2045, kachergis +0.1470; power 0.999 (P(correct | xsl_14) 1.000, P(correct | kachergis) 0.999; 2000 simulated experiments each).

```python
def contrast(acc, groups):
    return float(acc['A'].mean() - acc['B'].mean())
```

| theory | predicted contrast | mean accuracy per arm |
|---|---|---|
| inc_fazly | -0.1678 | A 0.324, B 0.491 |
| inc_uncfam_gamma | +0.1636 | A 0.574, B 0.411 |
| kachergis | +0.1470 | A 0.547, B 0.400 |
| xsl_03 | -0.2496 | A 0.266, B 0.516 |
| xsl_07 | -0.1346 | A 0.372, B 0.507 |
| xsl_08 | -0.1975 | A 0.292, B 0.489 |
| xsl_09 | -0.1565 | A 0.347, B 0.503 |
| xsl_04 | -0.1906 | A 0.309, B 0.500 |
| xsl_05 | -0.2099 | A 0.281, B 0.491 |
| xsl_06 | -0.2155 | A 0.276, B 0.491 |
| baseline | -0.0302 | A 0.261, B 0.292 |
| inc_bayesian_decay | -0.0266 | A 0.364, B 0.390 |
| inc_tilles | -0.2888 | A 0.218, B 0.506 |
| inc_uncfam_attention | +0.1507 | A 0.550, B 0.400 |
| xsl_10 | -0.1567 | A 0.347, B 0.503 |
| xsl_11 | -0.1853 | A 0.322, B 0.507 |
| xsl_12 | -0.2001 | A 0.315, B 0.515 |
| xsl_13 | -0.1963 | A 0.318, B 0.515 |
| xsl_14 | -0.2045 | A 0.305, B 0.509 |

## massed_vs_spaced_same_frequency_1c3649987a

Frozen 2026-10-20T05:53:59.280795+00:00, sha256 `1c3649987a702297…`, commit `e49ec54c`.

**Rationale.** This targets the temporal-contiguity mechanism. xsl_14 has a fading trace (prev, rho=0.23) that multiplies the gain of any word-object pair repeated on the immediately preceding trial by 1 + r, which is about 3.1 with r=2.1. It also lets that gain feed commitment. Kachergis has no trial-to-trial trace. Its only temporal term is near-uniform decay (C=0.996), so for equal exposure counts the order of repetitions should matter little. In one arm, 9 words appear in unbroken runs of L consecutive trials (massed) and 9 words appear with gaps of at least 2 trials (spaced). Both groups get exactly 6 exposures, trials are 3x3, and every word has a distinct consistent referent. Frequency, trial size, set size and test format are therefore identical, and only the temporal arrangement of repetitions differs. Because the contrast is a within-arm item contrast, the person ability offset cancels. The knob L (run length 2 or 3) tunes how much of the massed group's exposure falls on trace-boosted repeats.

**Prediction.** xsl_14 predicts a positive contrast (massed minus spaced), roughly +0.05 to +0.15. Each repeat on the next trial gets the (1 + r*prev) gain boost, which pushes massed pairs toward commitment faster, and commitment then amplifies later gains. Kachergis predicts a contrast near 0, or slightly negative, because it has no trace of the previous trial and almost no decay, so equal exposure counts give nearly equal associations. The decision midpoint is therefore a small positive difference, and a clearly positive observed contrast supports xsl_14.

**Design.** arm `a`: 18 words × 18 objects, 36 trials, 18 test items (knobs {'L': 2}, seed 100002); 50 people per arm.

**Decision rule.** Contrast above +0.0089 favours `xsl_14`. Predicted: xsl_14 +0.0401, kachergis -0.0223; power 0.841 (P(correct | xsl_14) 0.855, P(correct | kachergis) 0.841; 2000 simulated experiments each).

```python
def contrast(acc, groups):
    a = acc['a']
    g = groups['a']
    return a[g == 'massed'].mean() - a[g == 'spaced'].mean()
```

| theory | predicted contrast | mean accuracy per arm |
|---|---|---|
| inc_fazly | +0.0120 | a 0.495 |
| inc_uncfam_gamma | -0.0025 | a 0.412 |
| kachergis | -0.0223 | a 0.408 |
| xsl_03 | +0.0004 | a 0.517 |
| xsl_07 | +0.0310 | a 0.522 |
| xsl_08 | +0.0529 | a 0.523 |
| xsl_09 | +0.0230 | a 0.548 |
| xsl_04 | -0.0070 | a 0.498 |
| xsl_05 | +0.0003 | a 0.494 |
| xsl_06 | -0.0047 | a 0.491 |
| baseline | +0.0000 | a 0.292 |
| inc_bayesian_decay | -0.0029 | a 0.389 |
| inc_tilles | -0.0154 | a 0.534 |
| inc_uncfam_attention | -0.0216 | a 0.405 |
| xsl_10 | +0.0231 | a 0.548 |
| xsl_11 | +0.0236 | a 0.549 |
| xsl_12 | +0.0289 | a 0.557 |
| xsl_13 | +0.0311 | a 0.569 |
| xsl_14 | +0.0401 | a 0.569 |

## object_load_within_arm_small_vs_large_trials_5f5afbffdc

Frozen 2026-10-20T05:56:01.287766+00:00, sha256 `5f5afbffdc71d72c…`, commit `e49ec54c`.

**Rationale.** This targets how each theory handles the number of objects shown per trial, while word frequency, word competition and referential structure are held equal. In xsl_14, every exposure multiplies the whole row of each present word and the whole column of each present object by (1 - deff). Here deff grows roughly like (objects per trial)^1.07, so the model is load-limited. Encoding also falls with objects per trial. xsl_14 therefore predicts a steep accuracy loss for words trained in trials with many objects. Kachergis adds a fixed attention budget X per trial, shared over pairs, and decays everything only by C=0.996. Its distractor cost comes only from dilution of the budget and shows up gradually (corpus 2x2 to 2x4 drops about 0.10, against about 0.28 for xsl_14). The design is one arm with two interleaved item types. The 'S' words are trained in 2-word x 2-object trials using only S objects. The 'L' words are trained in 2-word x k-object trials, with the k-2 extra objects drawn from other L objects. Both types get the same number of exposures, the same words per trial, the same set size (12 words and objects each) and the same test format (12-AFC within the item's own group, so chance is equal). The contrast is a within-arm item difference, so person ability largely cancels. A larger effect under xsl_14 than under Kachergis reads as load-limited forgetting and encoding versus a fixed attention budget. The knobs set how heavy the L context is and how many repetitions there are.

**Prediction.** The contrast is accuracy on S words (2x2 trials) minus accuracy on L words (2xk trials, k=4 to 8), in the same arm. Both theories predict a positive value, because more objects per trial is harder. xsl_14 predicts a large gap, roughly +0.4 to +0.55. Its object-count-scaled row and column decay (deff about 0.2 to 0.35 per exposure at k=5 to 8) and reduced encoding push L words near floor, while S words stay high through the trace, commitment and pair-count gains. Kachergis predicts a clearly smaller gap, roughly +0.15 to +0.3. Its learning mass per trial is fixed, its decay is nearly zero, and L words can still be resolved by cross-trial intersection because the correct object appears on every exposure while each distractor appears only on a fraction. The decision threshold is the midpoint, about +0.3 to +0.4. Larger k and more repetitions should widen the gap between the theories, and the search picks the setting with the best separation.

**Design.** arm `x`: 24 words × 24 objects, 72 trials, 24 test items (knobs {'k': 6, 'reps': 6}, seed 200000); 50 people per arm.

**Decision rule.** Contrast above +0.3888 favours `xsl_14`. Predicted: xsl_14 +0.4788, kachergis +0.2988; power 0.998 (P(correct | xsl_14) 0.998, P(correct | kachergis) 0.999; 2000 simulated experiments each).

```python
def contrast(acc, groups):
    a = acc['x']
    g = groups['x']
    return float(a[g == 'S'].mean() - a[g == 'L'].mean())
```

| theory | predicted contrast | mean accuracy per arm |
|---|---|---|
| inc_fazly | +0.3321 | x 0.450 |
| inc_uncfam_gamma | +0.2607 | x 0.458 |
| kachergis | +0.2988 | x 0.457 |
| xsl_03 | +0.4539 | x 0.447 |
| xsl_07 | +0.4443 | x 0.466 |
| xsl_08 | +0.4045 | x 0.434 |
| xsl_09 | +0.4550 | x 0.472 |
| xsl_04 | +0.4081 | x 0.445 |
| xsl_05 | +0.4027 | x 0.429 |
| xsl_06 | +0.4022 | x 0.426 |
| baseline | +0.1195 | x 0.295 |
| inc_bayesian_decay | +0.1703 | x 0.395 |
| inc_tilles | +0.3033 | x 0.421 |
| inc_uncfam_attention | +0.2833 | x 0.455 |
| xsl_10 | +0.4554 | x 0.472 |
| xsl_11 | +0.4716 | x 0.470 |
| xsl_12 | +0.4624 | x 0.468 |
| xsl_13 | +0.4544 | x 0.469 |
| xsl_14 | +0.4788 | x 0.460 |

