# Prospective design queue

3 accepted design(s); 4 proposals tried.

## sparse_vs_dense_ambiguity_matched_exposure_a61ff1e790

Frozen 2026-10-20T04:39:15.357955+00:00, sha256 `a61ff1e7907c9022…`, commit `767303cb`.

**Rationale.** The design targets how each theory handles within-trial ambiguity when total exposure is held constant. In xsl_14, forgetting and interference scale with the number of objects per trial (deff grows with (no/4)^p and with the number of word-object pairs per trial). Its uncommitted-word returns are throttled when a word's row is spread over many objects, and committed pairs gain from clean repetition (kpos). So xsl_14 should do much better with sparse, clean repetition (1 word with a few objects per trial) than with dense trials (many words and many objects). inc_fazly has no capacity or interference cost. It simply accumulates alignments, and when several words are present they compete for each object (explaining away). That helps dense trials, whereas a single word per trial gets no cross-word competition and learns only from raw co-occurrence counts. Both arms use the same words, the same objects, the same number of exposures per word (6) and the same test (all objects as candidates). The only differences are the number of words and objects per trial and hence the number of trials. The contrast is mean accuracy in the sparse arm minus mean accuracy in the dense arm. It is positive and large for xsl_14, and about zero or negative for inc_fazly. The existing fits already point this way: 1x3 with 108 trials gives xsl_14 0.50 and fazly 0.38, while 4x4 with 27 trials gives xsl_14 0.34 and fazly 0.44.

**Prediction.** The contrast is mean accuracy in the sparse arm minus mean accuracy in the dense arm, with 6 exposures per word in both. xsl_14 predicts a clearly positive value, roughly +0.15 to +0.2 (sparse about 0.5, dense about 0.3-0.35). The reasons are that clean sparse repetition keeps returns near-linear and commits pairs, while dense trials cause heavy forgetting, interference and ambiguity-throttled learning. inc_fazly predicts a value near zero or negative, roughly -0.05 to -0.1 (sparse about 0.38, dense about 0.43). It has no load cost, and its cross-word competition for objects helps dense trials, whereas a lone word per trial is learned only from co-occurrence counts. The midpoint is about +0.05, so the decision rule is: contrast above it picks xsl_14, below it picks inc_fazly.

**Design.** arm `sparse`: 18 words × 18 objects, 108 trials, 18 test items; arm `dense`: 18 words × 18 objects, 27 trials, 18 test items (knobs {'dense_k': 4, 'n': 18, 'sparse_o': 2}, seed 1); 50 people per arm.

**Decision rule.** Contrast above +0.1450 favours `xsl_14`. Predicted: xsl_14 +0.2474, inc_fazly +0.0427; power 0.964 (P(correct | xsl_14) 0.964, P(correct | inc_fazly) 0.966; 2000 simulated experiments each).

```python
def contrast(acc, groups):
    return float(np.mean(acc['sparse']) - np.mean(acc['dense']))
```

| theory | predicted contrast | mean accuracy per arm |
|---|---|---|
| inc_fazly | +0.0427 | sparse 0.468, dense 0.425 |
| inc_uncfam_gamma | +0.3297 | sparse 0.645, dense 0.315 |
| kachergis | +0.3310 | sparse 0.632, dense 0.301 |
| xsl_03 | +0.1293 | sparse 0.520, dense 0.390 |
| xsl_07 | +0.2133 | sparse 0.564, dense 0.351 |
| xsl_08 | +0.1487 | sparse 0.536, dense 0.387 |
| xsl_09 | +0.2541 | sparse 0.600, dense 0.346 |
| xsl_04 | +0.1703 | sparse 0.538, dense 0.368 |
| xsl_05 | +0.1483 | sparse 0.515, dense 0.367 |
| xsl_06 | +0.1391 | sparse 0.509, dense 0.370 |
| baseline | +0.0832 | sparse 0.345, dense 0.261 |
| inc_bayesian_decay | +0.0596 | sparse 0.422, dense 0.362 |
| inc_tilles | -0.2045 | sparse 0.305, dense 0.509 |
| inc_uncfam_attention | +0.3440 | sparse 0.643, dense 0.299 |
| xsl_10 | +0.2542 | sparse 0.600, dense 0.346 |
| xsl_11 | +0.2767 | sparse 0.621, dense 0.344 |
| xsl_12 | +0.2645 | sparse 0.604, dense 0.340 |
| xsl_13 | +0.2527 | sparse 0.592, dense 0.339 |
| xsl_14 | +0.2474 | sparse 0.589, dense 0.341 |

## retroactive_object_reappearance_decay_99f807f88b

Frozen 2026-10-20T04:40:27.089876+00:00, sha256 `99f807f88bb2272d…`, commit `767303cb`.

**Rationale.** The design targets one mechanism difference: xsl_14 has retrieval-induced interference and decay, and inc_fazly has none. In xsl_14, every time an object appears in a trial, M[:, objs] is multiplied by (1 - deff). That erodes every word's association to that object, including pairs learned earlier. Carrier words that co-occur with the object also add to its column mass, which lowers the old word's column-share term at test. In inc_fazly, a word's meaning probabilities are updated only when that word is heard, so later trials that do not contain the word cannot change its row. In block 1, 18 word-object pairs are trained identically (2x2 trials, interleaved). Half of them (the 'foil' set) then have their objects reappear many times in block 2 as distractors beside new carrier words. The other half ('ctrl') are never touched again. Foil and ctrl items are assigned at random and trained identically, so the contrast is a within-arm, within-person item contrast. Person ability cancels, which gives high power. The only difference between the item types is whether the object reappears later with other words. xsl_14 predicts that foil items are forgotten and ctrl items are better. inc_fazly predicts no difference, because the foil words' rows are untouched.

**Prediction.** xsl_14 predicts a clearly positive contrast, roughly +0.15 to +0.30 (control minus foil), growing with k. Each reappearance of an object multiplies its column by about 0.85 (deff is roughly 0.15 for a 1x4 trial), so after 6-12 reappearances the old association is cut to about 0.4-0.15 of its value. Carrier words also add mass to that object's column, which lowers its share. inc_fazly predicts a contrast near 0 (within about ±0.03, i.e. noise only), because a word's meaning probabilities change only on trials where that word is heard. Foil and control words are trained identically, so the predicted accuracy difference is zero. The midpoint decision is therefore about +0.08 to +0.15.

**Design.** arm `x`: 27 words × 27 objects, 63 trials, 18 test items (knobs {'k': 9, 'reps': 4}, seed 100000); 50 people per arm.

**Decision rule.** Contrast above +0.0778 favours `xsl_14`. Predicted: xsl_14 +0.1555, inc_fazly +0.0000; power 0.993 (P(correct | xsl_14) 0.993, P(correct | inc_fazly) 0.997; 2000 simulated experiments each).

```python
def contrast(acc, groups):
    g = groups['x']
    a = acc['x']
    return a[g == 'ctrl'].mean() - a[g == 'foil'].mean()
```

| theory | predicted contrast | mean accuracy per arm |
|---|---|---|
| inc_fazly | +0.0000 | x 0.488 |
| inc_uncfam_gamma | +0.0001 | x 0.450 |
| kachergis | -0.0004 | x 0.457 |
| xsl_03 | +0.2550 | x 0.476 |
| xsl_07 | +0.2414 | x 0.454 |
| xsl_08 | +0.2510 | x 0.407 |
| xsl_09 | +0.1330 | x 0.534 |
| xsl_04 | +0.1949 | x 0.452 |
| xsl_05 | +0.1950 | x 0.430 |
| xsl_06 | +0.1990 | x 0.426 |
| baseline | +0.0000 | x 0.345 |
| inc_bayesian_decay | +0.0193 | x 0.294 |
| inc_tilles | +0.3290 | x 0.247 |
| inc_uncfam_attention | +0.0007 | x 0.460 |
| xsl_10 | +0.1337 | x 0.534 |
| xsl_11 | +0.1749 | x 0.510 |
| xsl_12 | +0.2041 | x 0.516 |
| xsl_13 | +0.1582 | x 0.543 |
| xsl_14 | +0.1555 | x 0.548 |

## synonym_second_label_column_competition_80a8ec0485

Frozen 2026-10-20T04:42:39.542359+00:00, sha256 `80a8ec048577be93…`, commit `767303cb`.

**Rationale.** The design targets the test-time column competition and retroactive interference in xsl_14, which inc_fazly lacks. In xsl_14 the final score is M * share^b, where share is the pair's fraction of the object's total association mass across all words. Every appearance of an object also multiplies its whole column by (1-deff), so old associations erode. inc_fazly instead reads each word's own row of probabilities, normalised within the word. It only learns competitively among words present in the same trial.

Phase 1 teaches A words to 'old' objects with unambiguous 1x1 trials, so xsl_14 builds a very strong association to each old object. Phase 2 trains two item types in the same ambiguous trials, with equal repetitions and positions. S words are second labels (synonyms) for old objects, so the object's column is dominated by the A word. U words are labels for brand-new objects. Because the A words never appear in phase 2, there is no in-trial competition for either type, so inc_fazly should treat S and U identically. xsl_14 should score S items lower because the old objects' column mass shrinks the S words' share. The contrast is a within-arm, within-person item difference (U minus S), so ability cancels. The only difference between the item types is whether the target object already carries another word's association.

**Prediction.** xsl_14 predicts a clearly positive contrast, roughly +0.08 to +0.20 (U better than S). Old objects carry large association mass from the A words, so the S words' share of those columns is small. Phase-2 appearances of those objects also decay the old column. The S words therefore score low at test, while U words on fresh columns keep a share near 1. inc_fazly predicts a contrast near 0 (within about ±0.02). Its word rows are normalised per word, A words are absent in phase 2, and S and U words receive identical alignments and repetitions, so the old object's history does not affect S words.

**Design.** arm `x`: 27 words × 18 objects, 42 trials, 18 test items (knobs {'a_reps': 2, 'k': 3, 'n': 9, 'p_reps': 4}, seed 201001); 50 people per arm.

**Decision rule.** Contrast above +0.0897 favours `xsl_14`. Predicted: xsl_14 +0.1862, inc_fazly -0.0068; power 0.999 (P(correct | xsl_14) 0.999, P(correct | inc_fazly) 1.000; 2000 simulated experiments each).

```python
def contrast(acc, groups):
    g = groups['x']
    a = acc['x']
    return a[g == 'U'].mean() - a[g == 'S'].mean()
```

| theory | predicted contrast | mean accuracy per arm |
|---|---|---|
| inc_fazly | -0.0068 | x 0.420 |
| inc_uncfam_gamma | +0.2154 | x 0.311 |
| kachergis | +0.2644 | x 0.282 |
| xsl_03 | +0.0004 | x 0.404 |
| xsl_07 | +0.1463 | x 0.334 |
| xsl_08 | +0.0234 | x 0.387 |
| xsl_09 | +0.1286 | x 0.317 |
| xsl_04 | +0.0062 | x 0.390 |
| xsl_05 | -0.0012 | x 0.378 |
| xsl_06 | -0.0018 | x 0.378 |
| baseline | +0.0000 | x 0.292 |
| inc_bayesian_decay | +0.0029 | x 0.277 |
| inc_tilles | +0.0617 | x 0.394 |
| inc_uncfam_attention | +0.2826 | x 0.274 |
| xsl_10 | +0.1286 | x 0.317 |
| xsl_11 | +0.1367 | x 0.316 |
| xsl_12 | +0.1056 | x 0.341 |
| xsl_13 | +0.1506 | x 0.319 |
| xsl_14 | +0.1862 | x 0.300 |

