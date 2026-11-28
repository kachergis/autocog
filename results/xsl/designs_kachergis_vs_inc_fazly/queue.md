# Prospective design queue

3 accepted design(s); 3 proposals tried.

## me_vs_cooccurrence_1word_trios_483e797294

Frozen 2026-10-20T05:43:19.188651+00:00, sha256 `483e7972944d7a32…`, commit `767303cb`.

**Rationale.** The theories differ in where competition lies. Kachergis spreads attention across the objects within a trial, weighting each by its uncertainty (column entropy) and familiarity. Objects that already have a strong partner are therefore down-weighted, which gives a strong mutual-exclusivity bias. Fazly normalises alignments across the words competing for an object. With one word per trial there is no competition, so it reduces to co-occurrence counting. It cannot use the fact that an object already belongs to another word. Design: 6 known words are first taught with their objects in 1x1 trials. Then each of 6 'ME' words is always shown alone with the same 3 objects: one novel target and 2 known objects (the referents of other, already learned words). Each of 6 'control' words is always shown with 3 novel objects. Within each trio the objects co-occur identically with the word on every trial. Cross-trial statistics therefore cannot identify the target, and only the known/novel status of the objects can. Kachergis should pick the novel target in ME trios, because known objects have low entropy and get little attention. Fazly gives equal association to all three objects, so it sits at 1/3 on both item types. The control items are matched on trio structure and number of exposures, and they remove any generic offset. The contrast is a within-arm item-type difference.

**Prediction.** Kachergis: the novel target wins the attention competition in ME trios, so ME accuracy (3-AFC) is well above chance, roughly 0.7-0.95. Control accuracy is about 1/3 because the three novel objects are symmetric. Contrast about +0.4 to +0.6. Inc_fazly: with a single word per trial, every co-present object gets nearly the same alignment, so the target and the two known distractors are tied. ME and control accuracy both sit near 1/3, and the contrast is about 0 (within about ±0.05). A threshold near +0.2 to +0.25 separates the theories.

**Design.** arm `main`: 18 words × 30 objects, 66 trials, 18 test items (knobs {'r1': 3, 'r2': 4}, seed 0); 50 people per arm.

**Decision rule.** Contrast above +0.2379 favours `kachergis`. Predicted: kachergis +0.4758, inc_fazly +0.0000; power 1.000 (P(correct | kachergis) 1.000, P(correct | inc_fazly) 1.000; 2000 simulated experiments each).

```python
def contrast(acc, groups):
    a = acc['main']
    g = groups['main']
    return a[g == 'me'].mean() - a[g == 'ctrl'].mean()
```

| theory | predicted contrast | mean accuracy per arm |
|---|---|---|
| inc_fazly | +0.0000 | main 0.496 |
| inc_uncfam_gamma | +0.4737 | main 0.667 |
| kachergis | +0.4758 | main 0.675 |
| xsl_03 | +0.1144 | main 0.517 |
| xsl_07 | +0.2993 | main 0.566 |
| xsl_08 | +0.1022 | main 0.511 |
| xsl_09 | +0.1809 | main 0.573 |
| xsl_04 | +0.0887 | main 0.502 |
| xsl_05 | +0.0877 | main 0.496 |
| xsl_06 | +0.0897 | main 0.495 |
| baseline | +0.0000 | main 0.495 |
| inc_bayesian_decay | +0.0399 | main 0.415 |
| inc_tilles | +0.3472 | main 0.481 |
| inc_uncfam_attention | +0.5022 | main 0.692 |
| xsl_10 | +0.1809 | main 0.573 |
| xsl_11 | +0.2013 | main 0.564 |
| xsl_12 | +0.1838 | main 0.564 |
| xsl_13 | +0.2209 | main 0.584 |
| xsl_14 | +0.2558 | main 0.601 |

## hub_vs_private_referent_single_word_b136841628

Frozen 2026-10-20T05:46:32.844396+00:00, sha256 `b13684162811658e…`, commit `767303cb`.

**Rationale.** The design targets how each theory handles object-side uncertainty when trials contain one word. Kachergis weights attention by object entropy (B about 5.3), so it favours objects that co-occur with many words ('hub' objects) and neglects objects tied to one word only ('private' objects). Fazly normalises alignments only across the words in a trial. With one word per trial there is no competition, so it is pure within-word co-occurrence counting and cannot use how many other words an object appears with. Each word has a fixed trio of objects (referent plus k companions) that co-occur with it on every trial, so the within-word counts are exactly tied. In group 'hubref', the referent is a hub object (it also appears as a companion in other words' trials) and the companions are private to that word. In group 'privref', the referent is private and the companions are hubs. Fazly is therefore at about 1/k+1 on both groups. Kachergis should be well above that on 'hubref' and below it on 'privref'. The contrast is a within-arm item-type difference with identical trial structure, exposure counts and candidate sets, so only the hub/private role of the referent changes. This is the opposite manipulation to a known-object ME design: it uses no pre-training and tests attraction to high-entropy objects rather than avoidance of low-entropy ones.

**Prediction.** Kachergis: strongly positive, about +0.4 to +0.6. Its entropy-driven attention shifts association mass to hub objects, so hub referents are picked far above the 1/(k+1) tie level (about 0.6-0.8) and private referents with hub companions fall below it (about 0.05-0.2). Inc_fazly: about 0 (within a few hundredths either way). Single-word trials give no competition and each word's trio is count-tied, so both groups sit near 1/(k+1) (0.33 for k=2, 0.25 for k=3). The midpoint decision threshold is about +0.2 to +0.3. The within-arm, matched-structure contrast makes ability differences largely cancel.

**Design.** arm `x`: 12 words × 24 objects, 60 trials, 12 test items (knobs {'k': 2, 'reps': 5}, seed 100000); 50 people per arm.

**Decision rule.** Contrast above -0.0716 favours `inc_fazly`. Predicted: kachergis -0.1432, inc_fazly +0.0000; power 0.978 (P(correct | kachergis) 0.978, P(correct | inc_fazly) 0.988; 2000 simulated experiments each).

```python
def contrast(acc, groups):
    a = acc['x']
    g = groups['x']
    return a[g == 'hubref'].mean() - a[g == 'privref'].mean()
```

| theory | predicted contrast | mean accuracy per arm |
|---|---|---|
| inc_fazly | +0.0000 | x 0.331 |
| inc_uncfam_gamma | -0.1505 | x 0.284 |
| kachergis | -0.1432 | x 0.267 |
| xsl_03 | -0.3564 | x 0.341 |
| xsl_07 | -0.3591 | x 0.310 |
| xsl_08 | -0.2013 | x 0.300 |
| xsl_09 | -0.1879 | x 0.288 |
| xsl_04 | -0.2247 | x 0.305 |
| xsl_05 | -0.2475 | x 0.300 |
| xsl_06 | -0.2542 | x 0.299 |
| baseline | +0.0000 | x 0.287 |
| inc_bayesian_decay | -0.0720 | x 0.249 |
| inc_tilles | -0.3761 | x 0.253 |
| inc_uncfam_attention | -0.1367 | x 0.275 |
| xsl_10 | -0.1883 | x 0.288 |
| xsl_11 | -0.2544 | x 0.301 |
| xsl_12 | -0.2675 | x 0.299 |
| xsl_13 | -0.2412 | x 0.286 |
| xsl_14 | -0.2274 | x 0.281 |

## order_matched_counts_novelty_capture_71c651e9ab

Frozen 2026-10-20T05:49:09.140263+00:00, sha256 `71c651e9abec4e65…`, commit `767303cb`.

**Rationale.** Both theories reach the same word-referent counts here, so any difference comes from how they use order. Each word has two objects: a referent R and a decoy D. R is always seen with the word more often than D (rS+k versus 1+k), and these counts are identical in both item groups. Only the order differs. 'Early' words have R seeded alone first, then k ambiguous [R,D] trials, then D alone once. 'Late' words have D seeded alone once, then k ambiguous trials, then R alone rS times. All trials have one word. In that case inc_fazly's alignment of an object is normalised only across the words in the trial, so it is essentially order-free co-occurrence counting. Its small first-exposure penalty is the same in both groups, so it predicts no difference. Kachergis weights attention by exp(B*entropy) of the object, with B about 5.3. In an ambiguous trial the object that is still unfamiliar (high column entropy) takes almost all of the attention, even when its pair is weaker. In 'early' words, the ambiguous trials therefore go to the decoy D and erode the early advantage of R. In 'late' words, they go to R, the newcomer, and add to R's alone trials. Kachergis therefore predicts a large recency/novelty advantage for late-introduced referents at matched counts. Fazly predicts about zero. The contrast is a within-arm item-type difference. It differs only in presentation order and in which of the two objects is correct, and the objects are randomly assigned to groups.

**Prediction.** The contrast is mean accuracy on 'late' items minus mean accuracy on 'early' items. Kachergis predicts a clearly positive value, roughly +0.2 to +0.5, growing with k. In ambiguous trials its entropy-weighted attention goes to the less familiar object. That object is the decoy for 'early' words, so their accuracy is lowered and may fall below chance between the two objects when k is large. For 'late' words it is the referent, so their accuracy is raised. inc_fazly predicts about 0, within roughly ±0.03. With one word per trial it is nearly order-free counting, and R and D counts are the same in both groups, so its accuracy is about equal for early and late items. Only a small, group-symmetric first-exposure penalty remains.

**Design.** arm `x`: 12 words × 24 objects, 72 trials, 12 test items (knobs {'k': 2, 'rS': 3}, seed 200000); 50 people per arm.

**Decision rule.** Contrast above +0.0789 favours `kachergis`. Predicted: kachergis +0.1651, inc_fazly -0.0073; power 0.988 (P(correct | kachergis) 0.988, P(correct | inc_fazly) 0.994; 2000 simulated experiments each).

```python
def contrast(acc, groups):
    a = acc['x']
    g = groups['x']
    return a[g == 'late'].mean() - a[g == 'early'].mean()
```

| theory | predicted contrast | mean accuracy per arm |
|---|---|---|
| inc_fazly | -0.0073 | x 0.459 |
| inc_uncfam_gamma | +0.0132 | x 0.465 |
| kachergis | +0.1651 | x 0.475 |
| xsl_03 | +0.0308 | x 0.480 |
| xsl_07 | -0.0092 | x 0.450 |
| xsl_08 | -0.0480 | x 0.457 |
| xsl_09 | -0.1945 | x 0.440 |
| xsl_04 | -0.0131 | x 0.445 |
| xsl_05 | -0.0268 | x 0.443 |
| xsl_06 | -0.0258 | x 0.440 |
| baseline | +0.0000 | x 0.379 |
| inc_bayesian_decay | +0.0000 | x 0.308 |
| inc_tilles | +0.0187 | x 0.357 |
| inc_uncfam_attention | +0.1416 | x 0.475 |
| xsl_10 | -0.1945 | x 0.440 |
| xsl_11 | -0.1541 | x 0.443 |
| xsl_12 | -0.2284 | x 0.469 |
| xsl_13 | -0.1924 | x 0.456 |
| xsl_14 | -0.1807 | x 0.453 |

