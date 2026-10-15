# Offline AutoCog for cross-situational word learning: first mode-B results

Status: **draft, 2026-10-14.** Nine generator rounds complete.
Run directory: [`results/xsl/mode_b/`](../results/xsl/mode_b/). Design:
[`xsl_offline_proposal.md`](xsl_offline_proposal.md).

## Summary

- An LLM theory generator (Claude Sonnet 5.5 via OpenRouter) proposed,
  revised and code-instantiated theories of cross-situational word learning,
  each fitted to 47 adult conditions from the XSLmodels corpus (31,266 human
  test choices). No new data were collected.
- After 9 rounds the best theory, **`xsl_09` (commitment-gated claim
  exclusion with redistributed credit), predicts held-out conditions 858 NLL
  better than the Kachergis, Yu & Shiffrin (2012) uncertainty/familiarity
  model** (5-fold cross-validation over conditions: 15,417 vs 16,275), and
  beats it on 36 of 47 conditions.
- The improvements came from a single lineage: competitive "mutual
  responsibility" credit assignment → interference-based (not global)
  forgetting → test-time mutual exclusivity ("claim competition") →
  fitted sharpening and a fading trace of recent co-occurrence (temporal
  contiguity) → graded per-word "commitment" that excludes the committed
  object for other words — a hypothesis-testing-like element grafted onto
  the associative learner.
- LLM cost from logged tokens ≈ $3.8 for 9 rounds (≈ $0.28 per round, plus
  one $1.30 runaway reply that is now capped). The OpenRouter key reports
  $32.39 spent this week; the gap is unexplained (see *Cost*).
- Main caveats: the only established comparison on this objective so far is
  the Kachergis model (the other xsl-bakeoff models have not yet been refit
  by likelihood); the cross-validation split is fixed and reused, so it can
  itself be overfit over many rounds; the sealed holdout sets have not been
  touched; the data export predates the XSLmodels 211/214 fix (`e70c10c`).

## Setup

| | |
|---|---|
| Data | `xsl_datasets`, 47 adult conditions, exported from XSLmodels `2999e8b` (`data/xsl/`). Holdouts (`benitez2020`, `kachergis_initial_accuracy`, `suanda2014`, `koehne2013`) sealed, unused. |
| Theory format | Prose claim + `learn(params, trials, n_words, n_objects) → association matrix`; Luce test choice (optionally power-Luce via `ch_dec`). |
| Objective | Group negative log-likelihood of the human choice counts: multinomial over all objects for n-AFC tests, binomial correct/incorrect for m-AFC tests. A guessing rate `lapse` is fitted for every theory and counted as a parameter. |
| Fitting | Differential evolution (popsize 10, ≤ 40 generations, polished). |
| Score | Held-out NLL from 5-fold cross-validation over conditions (same split for every theory). BIC and n_subj-weighted SSE reported for reference. |
| Loop | Per round: up to 3 propose → fit → cross-validate → critique iterations; a candidate is accepted only if it beats the round's running best; the round's best joins the leaderboard. |
| LLM | `anthropic/claude-sonnet-5.5` via OpenRouter, reasoning effort `high`. |
| Seeds | `baseline` (co-occurrence counting), `kachergis` (uncfam, entropy variant; NumPy port matches XSLmodels to ≤ 2e-13). |

Rounds 1–6 used BIC on the full-corpus fit as the accept gate; rounds 7–9
use held-out NLL (see *Accept gate* below). Every theory on the leaderboard
has been cross-validated with the same folds, so the ranking below is
consistent across both regimes.

## Results

### Leaderboard

Ranked by held-out NLL (lower is better). k counts free parameters including
`lapse`. wSSE is the n_subj-weighted SSE on per-item accuracy, with guessing
mixed in.

| rank | theory | k | held-out NLL | held-out wSSE | BIC (full fit) | full-fit wSSE | lapse |
|---|---|---|---|---|---|---|---|
| 1 | `xsl_09` commitment-gated claim exclusion, redistributed credit | 14 | **15,417.3** | **0.409** | **30,506.3** | 0.345 | .24 |
| 2 | `xsl_08` decaying-trace contiguity interference | 12 | 15,496.1 | 0.433 | 30,691.9 | 0.377 | .20 |
| 3 | `xsl_07` soft-sharpness interference + contiguity boost | 10 | 15,529.5 | 0.439 | 30,721.2 | 0.379 | .19 |
| 4 | `xsl_04` interference + test-time claim competition | 8 | 15,879.4 | 0.511 | 31,165.9 | 0.428 | .28 |
| 5 | `xsl_05` + consecutive-repetition boost | 9 | 16,040.9 | 0.547 | 31,077.1 | 0.418 | .26 |
| 6 | `xsl_06` + consolidation protection | 10 | 16,095.6 | 0.545 | 31,093.9 | 0.418 | .26 |
| 7 | `xsl_02` two-stage verify-and-consolidate | 6 | 16,104.9 | 0.538 | 32,002.0 | 0.498 | .34 |
| 8 | `xsl_03` load-scaled absent-alternative interference | 6 | 16,128.6 | 0.563 | 31,863.2 | 0.510 | .45 |
| 9 | `xsl_01` interference-limited mutual responsibility | 5 | 16,146.9 | 0.590 | 32,045.4 | 0.553 | .46 |
| 10 | `kachergis` (Kachergis, Yu & Shiffrin 2012) | 4 | 16,274.8 | 0.598 | 32,391.7 | 0.567 | .36 |
| 11 | `baseline` co-occurrence counting | 1 | 19,786.1 | 1.381 | 39,486.6 | 1.366 | .39 |

For orientation only: the xsl-bakeoff v05 group fits (by SSE, not
cross-validated, without a guessing rate) reach weighted SSE 0.477 (fazly),
0.512 (uncfam_gamma) and 0.558 (kachergis). These are not directly
comparable with the table — different objective and no held-out data — but
the top theories' *held-out* wSSE (0.41–0.44) is below every in-sample
bakeoff SSE.

### Trajectory

| round | accept gate | theory added | held-out NLL | Δ vs previous best | LLM cost |
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

### Accept gate: BIC admitted overfitting

Under the BIC gate, rounds 5 and 6 produced theories with *better* BIC than
`xsl_04` (31,077 and 31,094 vs 31,166) but *worse* held-out NLL (16,041 and
16,096 vs 15,879). The group likelihood treats all 31k choices as
independent, so its log(N) penalty per parameter is small relative to the
NLL differences at stake, and BIC did not penalise added mechanisms that
only fit the training conditions. After switching the gate to held-out NLL,
the next round gained 350 held out; within round 8 a 13-parameter candidate
was rejected for generalising worse than a 12-parameter one.

## What the theories found

All generated theories that reached the top of the leaderboard descend from
`xsl_01`. The lineage, with the mechanism each step added:

1. **Mutual-responsibility credit (`xsl_01`).** On each trial a word's credit
   is split over the co-present objects and each object's over the
   co-present words, in proportion to sharpened current strength; a pair
   gains the product of the two shares. This yields soft mutual exclusivity
   and explaining-away during learning.
2. **Interference instead of global forgetting (`xsl_01` onward).** Only
   links of the words and objects on the current trial decay; in later
   versions only their links to *absent* partners decay ("retrieval-induced
   interference"), at a rate that grows with trial load (words × objects).
3. **Test-time claim competition (`xsl_04`).** At test, an object strongly
   associated with other words is a poorer referent for the probed word: the
   association is multiplied by the probed word's share of the object's total
   association, raised to a fitted power. This was the largest single step
   (−226 held out over the previous best, −268 over `xsl_01`).
4. **Fitted, soft sharpening (`xsl_07`).** The exponent on current strength
   in the credit assignment became a free parameter (fitted ≈ 0.74 < 1: soft
   rather than winner-take-all competition).
5. **Temporal contiguity (`xsl_07`, `xsl_08`).** Pairs that co-occurred on the
   previous trial get amplified credit (`xsl_07`); `xsl_08` replaces the
   one-trial flag with a geometrically fading trace (≈ 37% of it carried
   over per trial), so overlap two or three trials back still helps, more
   weakly.
6. **Commitment with redistributed credit (`xsl_09`).** A word forms a graded
   "working hypothesis" once its strongest object holds a dominant share of
   its association; the committed pair is strengthened and its object is
   withdrawn as a candidate for other co-present words, whose credit is
   redistributed to their remaining objects (explaining-away). Sparse or
   frequent designs snowball into exclusive mappings; dense ones stay
   diffuse. With commitment in place, the fitted test-time claim exponent
   `b` went to 0: exclusion during learning replaced exclusion at test.

### Where the gains come from

Held-out NLL, `xsl_08` minus `kachergis` (negative = `xsl_08` better), by
condition. (`xsl_09` shows the same pattern: −236.5 on 221, −62.1 on 222,
+55.4 on 215; it gains most over `xsl_08` on 206, 220 and 221 and loses most
on 301.)

| condition (corpus label) | words × objects | per trial | trials | test | human acc | Δ held-out NLL |
|---|---|---|---|---|---|---|
| 221 ("1x4") | 18 × 18 | 1 × 4 | 108 | 18-AFC | .19 | −212.5 |
| 206 ("3x3 +1w/o") | 18 × 18 | 3 × 3 | 54 | 18-AFC | .17 | −66.9 |
| 222 ("1x3") | 18 × 18 | 1 × 3 | 108 | 18-AFC | .39 | −60.5 |
| 207 ("4x4 +2w/o") | 18 × 18 | 4 × 4 | 54 | 18-AFC | .10 | −59.2 |
| 205 ("2x4") | 18 × 18 | 2 × 4 | 54 | 18-AFC | .30 | −54.3 |
| freq369-3x3hiCD | 18 × 18 | 3 × 3 | 36 | 18-AFC | .56 | −41.2 |
| … 27 further conditions improve, 14 get worse | | | | | | |
| filt0E_3L | 12 × 12 | 2 × 2 | 18 | 11-AFC | .37 | +17.8 |
| 301 ("2x2 5x 24AFC") | 24 × 24 | 2 × 2 | 60 | 24-AFC | .51 | +19.5 |
| 215 ("1x3") | 18 × 18 | 1 × 3 | 108 | 18-AFC | .74 | +68.7 |

`xsl_08` wins on 33 of 47 conditions (total −778.8). A quarter of the gain is
one condition (221), and the largest remaining loss is 215 — a 1×3 design
with the same layout as 222 but much higher human accuracy (.74 vs .39),
presumably because of its trial order. The contiguity trace was introduced
to capture order effects of this kind, but it does not yet separate 215
from 222.

## Caveats

- **Comparison set.** On the likelihood objective, only `kachergis` and
  `baseline` are established references. The stronger xsl-bakeoff models
  (fazly, uncfam_gamma, uncfam_attention, tilles, Bayesian_decay) must be
  refit by NLL with the same guessing rate and cross-validated before
  claiming `xsl_08` beats the field.
- **Adaptive overfitting of the CV split.** The generator sees per-condition
  held-out results on a fixed split every round, so the split itself can be
  overfit as rounds accumulate. The sealed holdout sets are the guard; they
  have not been used.
- **Weak identification and bounds.** Several parameters vary widely across
  folds (e.g. `xsl_08`: interference-load exponent q 0.00–0.38, contiguity
  boost r 0.73–1.65; `xsl_09`: Mmax 11–29, kc 2.7–6.2). `xsl_09` fits two
  parameters at their bounds: `b` = 0 (test-time claim competition switched
  off) and the sharpening exponent `s` at its lower bound 0.3 in every fold,
  so its bounds should be widened before it is taken as final. The learners also carry hard-coded constants chosen by the
  LLM after seeing the data (e.g. `c`, `g`, `eps` in `xsl_08`), which are
  unpenalised degrees of freedom.
- **Guessing.** Fitted guessing rates are high (0.19–0.46); the best theories
  pair sharp associations with ~20% uniform guessing.
- **Data version.** The export is from XSLmodels `2999e8b`. PR #31
  (`e70c10c`) corrected the 4AFC alternatives of conditions 211 and 214 and
  Bayesian_decay's test-condition choice rule; all theories need re-scoring
  after re-export. (211 and 214 are scored binomially on the target, which
  limits the effect.)
- **Index exploitation ruled out.** In the corpus word *i*'s referent is
  object *i*; every theory gives identical NLL with indices shuffled, and the
  loop now shuffles indices by construction.
- **Single lineage, single LLM.** Every top theory refines one family. By
  round 9 that family had absorbed a hypothesis-testing-like commitment
  step, but the loop has not been seeded with a structurally different
  account to compete against, and only one model (Sonnet 5.5) has been tried.

## Cost

Per-call token counts are logged in every prompt file; at OpenRouter's list
price for Sonnet 5.5 ($2 / $10 per million input / output tokens) the nine
rounds come to ≈ $3.8, of which ≈ $1.30 is the single runaway reply. The
OpenRouter key, however, reports $32.39 spent (all of it this week). Usage
since 00:00 UTC ($0.65) matches the logged tokens for rounds 7–9, and no
hidden resamples or retries appear in the run logs, so ≈ $28 of earlier
spend is unaccounted for — either other use of the key or billing not
visible in per-call usage. OpenRouter's Activity page (per-request cost)
will settle it; until then treat ≈ $0.30 per round as a lower bound.

## Next steps

1. Re-export `data/xsl/` at XSLmodels `e70c10c` once the v05 refit finishes,
   and re-score every theory.
2. Refit the incumbent xsl-bakeoff models by NLL (+ `lapse`) and
   cross-validate them on the same folds, so the leaderboard compares against
   the field, not just `kachergis`.
3. Evaluate the top theories once on the sealed holdouts (adults and children
   reported separately).
4. Lower `xsl_09`'s bound on `s` (fitted at its floor, 0.3) and test
   dropping `b` (fitted at 0), then refit and cross-validate.
5. Phase 2: seed a debate with structurally different theories
   (associative vs hypothesis-testing) to escape the single lineage.
6. Phase 3: use the top theories vs the incumbents to propose discriminating new
   designs (e.g. separating 215- vs 222-style orderings).

## Reproducing

```bash
# seeds + 9 rounds (resumable), then (re)compute CV for every theory
uv run python main_xsl_offline.py --llm_provider openrouter \
    --llm_model anthropic/claude-sonnet-5.5 --n_rounds 9 --max_iters 3 \
    --out results/xsl/mode_b
uv run python main_xsl_offline.py --n_rounds 0 --cv --out results/xsl/mode_b
```

Every generator and critic exchange is in
`results/xsl/mode_b/round_NN/prompts/`; each theory's code is in
`results/xsl/mode_b/theories/<label>.yaml`.
