# Offline AutoCog for cross-situational word learning: mode-B results

Status: **draft, 2026-10-16.** Nine generator rounds, then a re-evaluation of
every theory on corrected data, against the xsl-bakeoff incumbents: first
under an item-accuracy objective (with robust fits and a guessing-rate
sensitivity check), then under a per-subject random-ability likelihood that
models individual differences directly.
Design: [`xsl_offline_proposal.md`](xsl_offline_proposal.md).

| run | what | directory |
|---|---|---|
| generation | 9 rounds of LLM theory generation (multinomial objective, XSLmodels `2999e8b`) | [`results/xsl/mode_b/`](../results/xsl/mode_b/) |
| re-score | all 16 theories (seeds, 5 incumbents, `xsl_01`–`09`) refit and cross-validated, binomial objective, `e70c10c` | [`results/xsl/binomial_e70c10c/`](../results/xsl/binomial_e70c10c/) |
| robust | top 10 refit with warm-started multi-start fits | [`results/xsl/binomial_e70c10c_robust/`](../results/xsl/binomial_e70c10c_robust/) |
| sensitivity | top 7 refit with the guessing rate capped at .10 | [`results/xsl/binomial_e70c10c_lapse10/`](../results/xsl/binomial_e70c10c_lapse10/) |
| random ability | top 10 refit on per-subject responses with random abilities (robust fits) | [`results/xsl/subject_e70c10c_robust/`](../results/xsl/subject_e70c10c_robust/) |

## Summary

- An LLM theory generator (Claude Sonnet 5.5 via OpenRouter) proposed,
  revised and code-instantiated theories of cross-situational word learning,
  each fitted to 47 adult conditions from the XSLmodels corpus (31,266 human
  test choices). No new data were collected.
- **The best generated theory, `xsl_09` (commitment-gated claim exclusion with
  redistributed credit), is first under every objective and treatment
  tried.** Under the per-subject random-ability likelihood — the final
  objective — it predicts held-out conditions 213 NLL better than the best
  xsl-bakeoff model on that criterion, kachergis (held-out NLL 16,922 vs
  17,135; fazly 17,154), with the lowest held-out weighted SSE of all (0.433;
  best incumbent fazly 0.504). Under the item-count binomial it led fazly by 312, and beat
  it on 33 of 47 conditions. It is over-parameterised: several parameters
  fit at their bounds.
- **Under the random-ability likelihood every generated theory out-predicts
  every incumbent.** The interference + test-time claim competition family
  (`xsl_04`–`06`, 8–10 parameters) comes second to fourth; the larger
  `xsl_07`/`xsl_08` overfit (held-out minus full-fit gaps > 100).
- **Individual differences, not guessing.** With people's abilities modelled
  (spread 1.4–1.6 logits), every theory in every cross-validation fold puts
  the guessing rate at its floor (.001). The high fitted guessing rates
  (.15–.46) under the earlier objectives were absorbing differences between
  people.
- **Fazly is a strong incumbent on item accuracy** — with three parameters
  it was second under the binomial objective and has the best held-out
  weighted SSE of the incumbents — but it ranks only ninth under the
  random-ability likelihood, which also scores how each person's responses
  hang together. The two criteria disagree for the incumbents.
- The objective now scores **item-level accuracy** (binomial). Under the
  earlier multinomial objective 71–80% of the likelihood was about *which
  wrong object* people chose, and fits traded accuracy for it. Under the
  binomial objective the incumbents rank exactly as in the xsl-bakeoff SSE
  fits.
- **Fitting reliability matters as much as the theories.** Single-start
  fits left several 8–10-parameter theories hundreds of NLL from their
  optimum; warm-started multi-start fits fixed most of it. Likelihood
  surfaces have separate optima, so held-out differences of ~100 can reflect
  which optimum a fold lands in.
- **Under the item-count binomial the guessing rate was a modelling choice,
  not a nuisance**: capping it at .10 improved held-out prediction for fazly,
  `xsl_07` and `xsl_08` and worsened it for the others. The random-ability
  likelihood resolves this (see above).
- LLM cost from logged tokens ≈ $3.8 for the nine rounds; the re-evaluation
  used no LLM calls. The OpenRouter key reports more spend than the logged
  tokens explain (see *Cost*).

## Setup

| | |
|---|---|
| Data | `xsl_datasets`, 47 adult conditions, exported from XSLmodels `e70c10c` (the pin of the xsl-bakeoff v05 fits; includes PR #31's fix of the 211/214 4AFC alternatives) to `data/xsl/`. Holdouts (`benitez2020`, `kachergis_initial_accuracy`, `suanda2014`, `koehne2013`) sealed, unused. |
| Theory format | Prose claim + `learn(params, trials, n_words, n_objects) → association matrix`; power-Luce test choice (`ch_dec`, if declared, on n-AFC and m-AFC tests alike). Indices are shuffled before `learn` sees them, so the corpus's "word i ↔ object i" convention cannot be exploited. |
| Per-subject data | Each condition's 0/1 test responses per person, exported from XSLmodels `data-raw` (1,609 people, no missing responses; every condition's per-item accuracy and `n_subj` reproduced exactly) to `data/xsl/subject_responses.json`. |
| Objectives | **Random ability** (final): person s has an ability offset u_s ~ N(0, `ability_sd`²) on the log-odds of the theory's predicted accuracy; each person's likelihood over their items is integrated over u_s (exact flat tails + composite Gauss–Legendre; within 1.4e-7 per person of a dense grid). **Binomial**: each item's correct count out of n under the predicted accuracy. **Multinomial**: over all objects (diagnostic). |
| Guessing | A guessing rate `lapse` (uniform choice at test) is fitted for every theory and counted as a parameter; under the random-ability objective `ability_sd` is fitted and counted too. Binomial sensitivity check with `lapse ≤ .10`. |
| Fitting | Differential evolution. Re-score: one start (popsize 10, ≤ 40 generations). Robust: 3 starts (popsize 12, ≤ 60 generations), the first warm-started from the previous fit; each CV fold 2 starts warm-started from the full-corpus fit. |
| Score | Held-out NLL from 5-fold cross-validation over conditions (same split for every theory). |
| Incumbents | NumPy ports of the xsl-bakeoff v05 panel — fazly, uncfam_gamma, uncfam_attention, tilles, Bayesian_decay — plus kachergis (uncfam) and co-occurrence baseline; each matches XSLmodels on all 47 conditions at its v05 fit. |

## Results

### Random-ability objective (final)

Ranked by held-out NLL under the per-subject random-ability likelihood (5-fold
CV over conditions; robust fits). k counts free parameters including `lapse`
and `ability_sd`. "Gap" is held-out minus full-corpus NLL (both summed over
the 47 conditions): large gaps mean overfitting or folds landing in different
optima. NLLs here are on a per-person scale and not comparable with the
binomial tables.

| rank | theory | k | held-out NLL | held-out wSSE | gap | ability_sd | lapse | rank under binomial |
|---|---|---|---|---|---|---|---|---|
| 1 | `xsl_09` commitment-gated claim exclusion | 15 | **16,921.7** | **0.433** | +59 | 1.44 | .001 | 1 |
| 2 | `xsl_04` interference + test-time claim competition | 9 | 16,991.3 | 0.494 | +44 | 1.47 | .001 | 8 |
| 3 | `xsl_05` + repetition boost | 10 | 16,995.9 | 0.497 | +34 | 1.48 | .001 | 6 |
| 4 | `xsl_06` + consolidation protection | 11 | 16,998.6 | 0.493 | +33 | 1.48 | .001 | 9 |
| 5 | `xsl_03` load-scaled absent-alternative interference | 7 | 17,050.6 | 0.546 | +38 | 1.49 | .001 | 5 |
| 6 | `xsl_08` decaying-trace contiguity interference | 13 | 17,061.6 | 0.509 | +115 | 1.46 | .001 | 3 |
| 7 | `xsl_07` soft-sharpness interference + contiguity | 11 | 17,065.7 | 0.517 | +130 | 1.46 | .001 | 4 |
| 8 | kachergis | 5 | 17,135.1 | 0.689 | +34 | 1.56 | .001 | 10 |
| 9 | fazly | 4 | 17,153.6 | 0.504 | +24 | 1.45 | .001 | 2 |
| 10 | uncfam_gamma | 6 | 17,177.2 | 0.707 | +102 | 1.59 | .001 | 7 |

- **No guessing is needed.** Every theory, and every cross-validation fold of
  every theory (one `xsl_03` fold: .008), puts `lapse` at its .001 floor.
  The between-person spread is stable at 1.4–1.6 logits — people differ
  roughly ±1.5 on the log-odds of their accuracy. `lapse` could be fixed to
  zero.
- **Parameters change meaning once individual differences are modelled.**
  Fazly's fit (`lambda` .147, `beta` 21.8) returns almost exactly to the
  xsl-bakeoff SSE fit (`beta` 21.4); uncfam_gamma's familiarity exponent
  flips from .69 (dampened) to 1.44 (amplified); kachergis settles at B ≈ 5
  between the two optima it alternated between under the binomial;
  `xsl_07`'s sharpening exponent rises above 1 (1.10) and its claim
  competition hits its bound; `xsl_04`'s claim competition, zero under the
  binomial, is active again (`b` .26).
- **The two criteria disagree for the incumbents.** Kachergis is the best
  incumbent by held-out NLL but the worst by held-out wSSE (0.689); fazly the
  reverse. The random-ability likelihood scores how each person's responses
  hang together, not just average item accuracy. `xsl_09` is best on both.
- **Remaining fit instability.** uncfam_gamma's large gap comes from one fold
  landing in its other optimum (B 12.8, γ .74 instead of B ≈ 2.5, γ ≈ 1.5).
  `xsl_05` contains `xsl_04` yet fitted 15 NLL worse on the full corpus (its
  repetition boost fitted to ~0 but it stayed near `Mmax` 2.2 rather than
  `xsl_04`'s 0.3).

### Binomial objective (item counts)

Ranked by held-out NLL with the guessing rate fitted
freely. k counts free parameters including `lapse`. "Fit" says whether the
numbers come from the robust multi-start refit or the single-start re-score.

| rank | theory | k | held-out NLL | held-out wSSE | lapse | held-out NLL, lapse ≤ .10 | fit |
|---|---|---|---|---|---|---|---|
| 1 | `xsl_09` commitment-gated claim exclusion | 14 | **3,192.8** | **0.400** | .23 | **3,288.5** | robust |
| 2 | fazly (Fazly et al. 2010) | 3 | 3,504.5 | 0.492 | .15 | 3,486.8 | robust |
| 3 | `xsl_08` decaying-trace contiguity interference | 12 | 3,519.2 | 0.486 | .20 | 3,392.6 | robust |
| 4 | `xsl_07` soft-sharpness interference + contiguity | 10 | 3,534.1 | 0.494 | .20 | 3,396.3 | robust |
| 5 | `xsl_03` load-scaled absent-alternative interference | 6 | 3,551.4 | 0.499 | .31 | 3,571.8 | robust |
| 6 | `xsl_05` interference + repetition boost | 9 | 3,566.9 | 0.501 | .29 | — | robust |
| 7 | uncfam_gamma | 5 | 3,593.4 | 0.505 | .23 | 3,685.0 | robust |
| 8 | `xsl_04` interference + test-time claim competition | 8 | 3,607.0 | 0.514 | .29 | — | robust |
| 9 | `xsl_02` two-stage verify-and-consolidate | 6 | 3,652.4 | 0.536 | .23 | — | single-start |
| 10 | `xsl_01` interference-limited mutual responsibility | 5 | 3,652.6 | 0.526 | .34 | — | single-start |
| 11 | `xsl_06` + consolidation protection | 10 | 3,678.1 | 0.517 | .31 | — | robust (lower bound, see below) |
| 12 | kachergis (Kachergis, Yu & Shiffrin 2012) | 4 | 3,916.8 | 0.604 | .28 | 4,018.1 | robust |
| 13 | tilles (Tilles & Fontanari 2012) | 4 | 3,931.8 | 0.600 | .30 | — | single-start |
| 14 | uncfam_attention | 4 | 3,987.2 | 0.622 | .29 | — | single-start |
| 15 | Bayesian_decay | 4 | 4,037.8 | 0.647 | .38 | — | single-start |
| 16 | co-occurrence baseline | 1 | 6,716.9 | 1.353 | .28 | — | single-start |

### Why the objective changed

For n-AFC tests the multinomial likelihood splits exactly into an accuracy
term (binomial: correct or not) and a which-wrong-object term. In the
generation runs the latter was 71–80% of the NLL (kachergis 77%, `xsl_09`
80%), so fits traded item-level accuracy for confusion structure: kachergis
fitted that way predicted .41 accuracy on the hardest conditions, where
people score .16. The binomial objective scores accuracy directly, weighting
each item by its sample size (which SSE does not).

Under it the incumbents' full-corpus ranking is exactly the xsl-bakeoff v05
SSE ranking (fazly < uncfam_gamma < kachergis < uncfam_attention < tilles <
Bayesian_decay); held out, only tilles and uncfam_attention swap. Under the
multinomial objective Bayesian_decay had out-fitted kachergis — its edge was
in which wrong objects people choose, not accuracy.

The objective also changes which generated mechanisms survive (see *How the
theories were found*): test-time claim competition fits to zero in `xsl_04`,
`xsl_05` and `xsl_06`, and is active only alongside soft sharpening (`xsl_07`
onward).

### Fitting reliability

| theory | k | full NLL, single-start → robust | held-out NLL, single-start → robust |
|---|---|---|---|
| fazly | 3 | 3,404.2 → 3,404.2 | 3,504.6 → 3,504.5 |
| uncfam_gamma | 5 | 3,454.7 → 3,454.7 | 3,629.7 → 3,593.4 |
| kachergis | 4 | 3,620.7 → 3,620.7 | 3,807.8 → 3,916.8 |
| `xsl_03` | 6 | 3,319.2 → 3,319.2 | 3,551.5 → 3,551.4 |
| `xsl_07` | 10 | 3,099.8 → 3,095.4 | 3,523.3 → 3,534.1 |
| `xsl_08` | 12 | 3,090.4 → 3,084.4 | 3,537.4 → 3,519.2 |
| `xsl_09` | 14 | 2,966.2 → 2,927.5 † | 3,321.1 → 3,192.8 † |
| `xsl_04` | 8 | 3,322.9 → 3,238.9 | 3,907.2 → 3,607.0 |
| `xsl_05` | 9 | 3,623.1 → 3,202.2 | 3,796.1 → 3,566.9 |
| `xsl_06` | 10 | 3,243.7 → 3,243.7 | 3,881.8 → 3,678.1 |

† with `s` widened to [0.01, 4] and `kc` to [0, 40].

- Single-start fits were fine for the incumbents and `xsl_03` but missed the
  optimum of several larger theories by up to 421 NLL. `xsl_05` contains
  `xsl_04` as a special case yet had fitted 300 NLL worse; after the refit it
  fits better, as it must, and its repetition boost is clearly nonzero.
- **Separate optima.** A warm-started refit can never fit its training data
  worse, and none did. But better training fits need not generalise better:
  in one kachergis fold the refit moved from a low-uncertainty-weight optimum
  (B 1.4, guessing .31) to a high one (B 8.2) — 41 NLL better on training,
  109 worse held out. For kachergis the uncertainty weight and the guessing
  rate trade off: diffuse predictions with frequent guessing, or sharp ones
  with less. In uncfam_gamma the refit escaped a poor optimum in one fold
  (held out −37).
- `xsl_06` contains `xsl_05` but its three starts never reached `xsl_05`'s
  low-`Mmax` region, so its score is a lower bound on what it can do.

### Guessing rate

| theory | free lapse | full NLL cost of lapse ≤ .10 | held-out NLL, free → capped |
|---|---|---|---|
| fazly | .15 | +7 | 3,504.5 → 3,486.8 |
| `xsl_07` | .20 | +18 | 3,534.1 → 3,396.3 |
| `xsl_08` | .20 | +19 | 3,519.2 → 3,392.6 |
| `xsl_09` | .23 | +40 (`lam`, `r` still at bounds) | 3,192.8 → 3,288.5 |
| uncfam_gamma | .23 | +84 | 3,593.4 → 3,685.0 |
| `xsl_03` | .31 | +108 (`ch_dec` → its floor) | 3,551.4 → 3,571.8 |
| kachergis | .28 | +163 (B 1.3 → 11.2) | 3,916.8 → 4,018.1 |

Theories that fit well with little guessing (fazly, `xsl_07`, `xsl_08`)
predict held-out conditions *better* with the cap: a freely fitted guessing
rate overfits the training conditions. Theories that lean on guessing
(uncfam_gamma, kachergis, `xsl_03`) compensate in other ways — kachergis
jumps to its high-B optimum, `xsl_03` flattens its choice rule to the floor —
and predict worse; so does `xsl_09` (+96 held out), though it stays first
(3,288.5 vs 3,392.6 for `xsl_08` and 3,486.8 for fazly). The ranking between
fazly and `xsl_07`/`xsl_08` flips with the cap, so under item counts the
treatment of guessing changes conclusions. The random-ability likelihood
settles it: once people's abilities are modelled, no theory needs any
guessing at all (see above).

### Where `xsl_09` beats fazly

Held-out NLL, `xsl_09` minus fazly (robust fits, free guessing rate;
negative = `xsl_09` better). Total −311.7; `xsl_09` wins on 33 of 47.

| condition (corpus label) | words × objects | per trial | trials | test | human acc | Δ held-out NLL |
|---|---|---|---|---|---|---|
| 215 ("1x3") | 18 × 18 | 1 × 3 | 108 | 18-AFC | .74 | −97.5 |
| 206 ("3x3 +1w/o") | 18 × 18 | 3 × 3 | 54 | 18-AFC | .17 | −36.7 |
| 220 ("3x4 9/12") | 18 × 18 | 3 × 4 | 54 | 18-AFC | .69 | −29.9 |
| 223 ("4x4") | 18 × 18 | 4 × 4 | 27 | 18-AFC | .31 | −27.4 |
| 216 ("2x3") | 18 × 18 | 2 × 3 | 54 | 18-AFC | .58 | −26.7 |
| 201 ("3x4") | 18 × 18 | 3 × 4 | 36 | 18-AFC | .19 | −24.4 |
| … 27 further conditions improve, 8 get worse | | | | | | |
| cont_div6-12 | 18 × 18 | 3 × 3 | 36 | 18-AFC | .43 | +15.5 |
| filt6E_9L | 12 × 12 | 2 × 2 | 72 | 11-AFC | .59 | +16.2 |
| 3_x8_369_4x4 | 18 × 18 | 4 × 4 | 27 | 18-AFC | .39 | +20.7 |
| 2_x8_39_4x4 | 18 × 18 | 4 × 4 | 27 | 18-AFC | .41 | +21.7 |
| 301 ("2x2 5x 24AFC") | 24 × 24 | 2 × 2 | 60 | 24-AFC | .51 | +38.3 |
| filt0E_3L | 12 × 12 | 2 × 2 | 18 | 11-AFC | .37 | +42.6 |

Condition 215 — a 1×3 design with the same layout as 222 but much higher
human accuracy (.74 vs .39), and the worst condition for every theory under
the multinomial objective — is now `xsl_09`'s largest gain. Its losses are
concentrated in short filtering designs, 301, and the frequency-manipulated
4×4 conditions.

## How the theories were found

The nine generator rounds ran under the multinomial objective on the
`2999e8b` export, so their numbers below are not comparable with the tables
above; they record what the loop did.

### Trajectory

| round | accept gate | theory added | held-out NLL (multinomial) | Δ vs previous best | LLM cost |
|---|---|---|---|---|---|
| 1 | BIC | `xsl_01` | 16,146.9 | −127.9 (vs kachergis) | $0.24 |
| 2 | BIC | `xsl_02` | 16,104.9 | −42.0 | $0.28 |
| 3 | BIC | `xsl_03` | 16,128.6 | +23.7 | $0.30 |
| 4 | BIC | `xsl_04` | 15,879.4 | −225.5 | $0.30 |
| 5 | BIC | `xsl_05` | 16,040.9 | +161.5 | $1.57 † |
| 6 | BIC | `xsl_06` | 16,095.6 | +216.2 | $0.29 |
| 7 | CV | `xsl_07` | 15,529.5 | −349.9 | $0.25 |
| 8 | CV | `xsl_08` | 15,496.1 | −33.4 | $0.27 |
| 9 | CV | `xsl_09` | 15,417.3 | −78.8 | $0.30 |

† includes one runaway reply (128k output tokens of invented tool-call
markup, ≈ $1.30); output is now capped at 32k tokens per call.

**Accept gate.** Under BIC, rounds 5 and 6 produced theories with better BIC
than `xsl_04` but worse held-out NLL: the group likelihood treats all 31k
choices as independent, so the log(N) penalty per parameter is small relative
to the NLL differences at stake. The gate switched to held-out NLL from round
7. (In hindsight, part of rounds 5–6's poor held-out scores was also the
single-start fits; see *Fitting reliability*.)

### Mechanisms

All top generated theories descend from `xsl_01`. What each step added, and
whether it survives under the accuracy objective:

1. **Mutual-responsibility credit (`xsl_01`).** A word's credit is split over
   the co-present objects and each object's over the co-present words, in
   proportion to sharpened current strength; a pair gains the product of the
   two shares — soft mutual exclusivity and explaining-away during learning.
2. **Interference instead of global forgetting (`xsl_01` onward).** Only links
   of the words and objects on the current trial decay; later, only their
   links to *absent* partners, at a rate growing with trial load.
3. **Test-time claim competition (`xsl_04`).** At test an object strongly
   claimed by other words is a poorer referent. The largest step under the
   multinomial objective; under the accuracy objective it fits to zero in
   `xsl_04`–`06` and is active only alongside soft sharpening.
4. **Fitted, soft sharpening (`xsl_07`).** The exponent on current strength in
   credit assignment became free (fitted ≈ 0.7 < 1).
5. **Temporal contiguity (`xsl_07`, `xsl_08`).** Pairs repeated from the
   previous trial get amplified credit; `xsl_08` makes it a fading trace. Under
   the accuracy objective the graded trace adds little over the one-trial
   boost (held out −15 free, −4 capped). A separate repetition boost in
   `xsl_05` fits clearly nonzero once fitted robustly.
6. **Commitment with redistributed credit (`xsl_09`).** A word forms a graded
   working hypothesis once its strongest object dominates; the committed pair
   is strengthened and its object withdrawn from other co-present words, whose
   credit is redistributed. The strongest theory under both objectives.

## Caveats

- **Objective choice changes the incumbents' order** (fazly 2nd under the
  binomial, 9th under random ability; kachergis 10th → 8th) and which
  generated mechanisms look active; `xsl_09`'s first place is the one result
  stable across all of them.
- Under random ability, `xsl_09` still fits `lam`, `Mmax` and `xi` at their
  upper bounds (`kc` now inside its widened range, at 25).
- **`xsl_09` is over-parameterised.** `lam`, `Mmax`, `r`, `kc` and `xi` sit at
  their upper bounds in the full fit, and `kc`, `Mmax` and `r` in every fold
  too (`kc` 37.7–40 after its range was widened from 8 to 40, `Mmax` ≈ 30,
  `r` 2.96–3): directions along which the fit keeps improving. It needs reparameterising
  (or those parameters fixed) before its parameter estimates mean anything;
  its predictions are what has been validated.
- **Guessing treatment changes the ranking** between fazly and `xsl_07` /
  `xsl_08` (see *Guessing rate*); `xsl_09` is first either way.
- **Separate optima** make held-out NLL differences of ~100 fragile, and some
  fits (marked single-start, and `xsl_06`) are not robust.
- **Adaptive overfitting of the CV split.** The generator saw per-condition
  held-out results on a fixed split every round, so the split itself may be
  overfit. The sealed holdouts are the guard and have not been used.
- **Unpenalised degrees of freedom.** Learners carry hard-coded constants the
  LLM chose after seeing the data.
- **Group-level likelihood.** It treats all choices as independent, ignoring
  between-subject variability; a beta-binomial model with a fitted dispersion
  would be more honest and would partly replace the guessing rate.
- **Index exploitation ruled out.** Every theory gives identical NLL with
  indices shuffled, and the loop now shuffles by construction.
- **Single lineage, single LLM.** Every top theory refines one family; only
  Sonnet 5.5 has been tried.

## Cost

Per-call token counts are logged in every prompt file; at OpenRouter's list
price for Sonnet 5.5 ($2 / $10 per million input / output tokens) the nine
rounds come to ≈ $3.8, of which ≈ $1.30 is the single runaway reply. The
OpenRouter key, however, reported $32.39 spent; usage over the last UTC day
matched the logged tokens and no hidden resamples or retries appear in the
run logs, so ≈ $28 of earlier spend is unaccounted for — other use of the key
or billing not visible in per-call usage. OpenRouter's Activity page will
settle it; treat ≈ $0.30 per round as a lower bound. The re-evaluation
(re-score, robust refits, sensitivity) used no LLM calls.

## Next steps

1. **Adopt the random-ability likelihood as the default objective** for
   further generation, with `lapse` fixed at 0 (it fits to its floor
   everywhere) — one fewer parameter per theory. Report held-out weighted SSE
   alongside, since the two criteria can disagree.
2. **Reparameterise `xsl_09`** (or fix its bounded parameters) and refit.
3. Robustly refit the remaining single-start theories (tilles,
   uncfam_attention, Bayesian_decay, `xsl_01`, `xsl_02`) and give `xsl_06` a
   start in `xsl_05`'s region.
4. Evaluate the top theories once on the sealed holdouts (adults and children
   reported separately).
5. Resume generation under the random-ability objective with the incumbents
   and `xsl_09` on the leaderboard, using robust (warm-started) fits at the
   accept gate where affordable.
6. Phase 2: seed a debate with structurally different theories (associative
   vs hypothesis-testing). Phase 3: use `xsl_09` vs fazly to propose
   discriminating designs (e.g. 215- vs 222-style orderings).

## Reproducing

```bash
# generation (multinomial objective, as run): seeds + 9 rounds, resumable
uv run python main_xsl_offline.py --objective multinomial --llm_provider openrouter \
    --llm_model anthropic/claude-sonnet-5.5 --n_rounds 9 --max_iters 3 --out results/xsl/mode_b

# re-score every theory + incumbents under the binomial objective
uv run python main_xsl_offline.py --n_rounds 0 --incumbents \
    --import_from results/xsl/mode_b/leaderboard.json --out results/xsl/binomial_e70c10c

# robust multi-start refits, then the lapse <= .10 sensitivity check
uv run python scripts/xsl/robust_refit.py --source results/xsl/binomial_e70c10c \
    --out results/xsl/binomial_e70c10c_robust --labels inc_fazly inc_uncfam_gamma kachergis \
    xsl_03 xsl_07 xsl_08 xsl_09 xsl_04 xsl_05 xsl_06 \
    --bounds "xsl_09:s=[0.01, 4]" "xsl_09:kc=[0, 40]"
uv run python scripts/xsl/robust_refit.py --source results/xsl/binomial_e70c10c_robust \
    --out results/xsl/binomial_e70c10c_lapse10 --lapse_max 0.1 --restarts 2 \
    --labels inc_fazly inc_uncfam_gamma kachergis xsl_03 xsl_07 xsl_08 xsl_09

# per-subject responses (XSLmodels checkout at the manifest commit), then the
# random-ability refits, warm-started from the binomial robust fits
Rscript scripts/xsl/export_subject_responses.R --xslmodels ../XSLmodels
uv run python scripts/xsl/robust_refit.py --source results/xsl/binomial_e70c10c_robust \
    --source_objective binomial --objective subject --out results/xsl/subject_e70c10c_robust \
    --labels inc_fazly inc_uncfam_gamma kachergis xsl_03 xsl_07 xsl_08 xsl_09 xsl_04 xsl_05 xsl_06
```

Every generator and critic exchange is in `results/xsl/mode_b/round_NN/prompts/`;
each theory's code is in `results/xsl/<run>/theories/<label>.yaml`.
