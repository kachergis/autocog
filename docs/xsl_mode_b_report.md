# Offline AutoCog for cross-situational word learning: progress report

Status: **draft, 2026-10-19.** It covers everything done so far: Phase 0, the
data bridge; Phase 1, theory generation in mode B (the full corpus visible);
re-evaluation under three objectives; Phase 2, a debate between the best
generated theory and the best incumbent; and a first Phase 3 run of
prospective designs that discriminate them. No new data were collected. The
sealed holdouts and mode A (sequential unsealing) have not been used.
Design: [`xsl_offline_proposal.md`](xsl_offline_proposal.md). Branch
`xsl-offline-phase0`.

## Summary

- **What the system does.** An LLM theory generator (Claude Sonnet 5.5 via
  OpenRouter) writes theories of cross-situational word learning. Each theory
  is a prose claim plus a `learn()` function. The loop fits every theory to
  47 adult conditions from the XSLmodels corpus (1,609 people, 31,266 test
  choices) and scores it by cross-validated held-out NLL. It then critiques
  the theory and revises it. The incumbents are the xsl-bakeoff v05 models,
  ported to NumPy and checked against R.
- **Generated theories beat every incumbent.** Under the final objective, a
  per-subject random-ability likelihood, the best theory is `xsl_14`
  (17 parameters). It predicts held-out conditions 255 NLL better than the
  best incumbent, kachergis (16,880.2 vs 17,135.1), and 273 better than fazly
  (17,153.6). It also has the lowest held-out weighted SSE (0.400; fazly
  0.504). All twelve generated theories in the table rank above all six
  incumbents.
- **One lineage.** Every leading theory descends from `xsl_01`. The lineage
  adds mutual-responsibility credit and interference instead of global
  forgetting (`xsl_01`), test-time claim competition (`xsl_04`), contiguity
  (`xsl_07`/`08`), and commitment with redistributed credit (`xsl_09`). It
  then adds exposure scaling (`xsl_12`) and the two debate revisions
  (`xsl_13`, `xsl_14`). From `xsl_09` on, all variants are within about 40
  held-out NLL of each other.
- **Individual differences, not guessing.** Once each person's ability is
  modelled (spread 1.4–1.6 logits), every theory puts the guessing rate at
  its floor, in every cross-validation fold. The high guessing rates fitted
  under earlier objectives (.14–.46) were absorbing differences between
  people.
- **The objective matters.** The first objective was a multinomial over all
  objects. 71–80% of its likelihood was about which wrong object people
  chose, and fits traded item accuracy for it. The binomial item-accuracy
  objective reproduces the xsl-bakeoff ranking of the incumbents exactly.
  The random-ability objective changes it again: fazly is 2nd of 16 under
  the binomial but 14th of 19 under random ability, below every generated
  theory.
- **Phase 2 found a persistent failure.** In six debate rounds, `xsl_12`/`14`
  won every query on load and ambiguity. Examples are the cost of an extra
  object per trial and the floor in high-ambiguity designs, where fazly is up
  to 16 SE off. Fazly won every query on repetition. In the filtering
  experiments people gain about .17 from late repetitions, whatever their
  number of early exposures. The commitment-gated family predicts about .04
  once early exposures are present. Two revisions were accepted on overall
  fit (`xsl_13`, `xsl_14`) but did not fix it. Under a stricter gate that
  required closing the lost query, none of 9 candidates passed. The
  candidates that moved the query most cost 31–34 held-out NLL. Within this
  family, fitting the repetition effect costs fit elsewhere.
- **Phase 3 queued three new experiments** for `xsl_14` vs fazly, each at
  power ≥ .96 with 50 people per arm. The strongest is a retroactive
  interference test. 18 word–object pairs are learned; then half of the
  objects reappear with new words. Every theory that updates only the words
  heard (fazly, kachergis, uncfam) predicts no cost to the old pairs. Every
  interference theory (`xsl_03`–`14`, tilles) predicts a cost of .13–.33. A
  synonym test splits fazly from xsl_14 and kachergis. A load test is largely
  answered by the existing corpus already. Three more designs, for
  kachergis vs fazly (power ≥ .98), all use one word per trial: fazly's
  competition acts only between the words on a trial, so it reduces to
  counting there. One of them, a trial-order design, also splits kachergis
  from the generated theories, which predict the opposite sign.
- **Process lessons.**
  - Single-start fits left larger theories up to 421 NLL from their optimum,
    so fitting needs restarts and warm starts.
  - BIC on the group likelihood admitted overfitting, so the accept gate
    uses cross-validated NLL.
  - The generator copied fitted values into "fixed" constants. This leaked
    held-out data, so such constants are now refitted.
  - Fitting can switch off a mechanism the debate asked for, so a debate
    revision gate has to test the query, not just the NLL.
- **Cost:** about $8.1 of LLM calls in logged tokens across 12 generator
  rounds, 6 debate rounds and 7 design proposals. All re-evaluation was local computation.

## Steps taken

| date | step | commits |
|---|---|---|
| 10-14 | **Phase 0, data bridge.** Rscript export of `xsl_datasets`, the sealed holdouts, the v05 incumbent fits and R reference predictions → `data/xsl/`. NumPy ports of baseline and kachergis match R to ≤ 2.2e-13 on all 47 conditions. An Rscript bridge reproduces v05 `ModelPerf`. | `ba13e1a7` |
| 10-14 | **Phase 1, generator loop.** `XSLTheory` (prose + `learn`), fitting (differential evolution with a fitted guessing rate), 5-fold CV over conditions, prompts, a critic, and an accept gate. | `0e2b189f` |
| 10-14 | Rounds 1–6 under a BIC gate (multinomial objective, XSLmodels `2999e8b`). Guards added: shuffled indices, a no-tool-call rule, and a 32k output cap after one runaway reply. | `1f19199d`–`fd6bfefc` |
| 10-14 | **Accept gate → held-out NLL** (BIC admitted `xsl_05`/`06`, which generalised worse), then rounds 7–9. | `e8343260`, `2e7f9a91` |
| 10-14 | Re-export at XSLmodels `e70c10c` (the v05 pin, including PR #31's fix of the 211/214 test alternatives). `ch_dec` applied on m-AFC tests. NumPy ports of the five-model v05 panel. | `cfff5512`, `ea9f4f55` |
| 10-14/15 | **Binomial item-accuracy objective**, and a re-score of all 16 theories under it. | `709fbea9`, `86fa6286` |
| 10-15 | **Robust fitting**: multi-start, warm starts and parallel populations. Top 10 refit; guessing rate ≤ .10 sensitivity check. | `d6aadb05`–`01c40b7e` |
| 10-16 | **Per-subject data and the random-ability likelihood**: logit-normal person offsets with a fitted SD, integrated per person. Top 10 refit robustly. | `e3b8433c`, `714c49d3` |
| 10-17 | Random ability made the default objective. Rounds 10–12 with the full incumbent field on the leaderboard. A labelling bug was found and fixed. **Leakage through pinned constants** was found: `xsl_10`–`12` were refit honestly, and the loop now refits pinned copies. | `e5ab5867`–`e015aee6` |
| 10-17/18 | **Phase 2, corpus-query debate** (`xsl_12` vs fazly), rounds 1–3: `xsl_13` and `xsl_14` accepted. | `40360e68`, `f91110d4` |
| 10-18/19 | **Query gate** (`--gate nll_and_query`) and gated rounds 4–6: nothing accepted. | `0b35b8a6`, `db8214f3` |
| 10-19 | **Phase 3, prospective designs** (`xsl_14` vs fazly): an LLM-written design generator, a power simulation at N = 50 and a frozen queue. 3 designs accepted from 4 proposals. | `767303cb` |
| 10-19 | Phase 3 for kachergis vs fazly: 3 designs accepted from 3 proposals. | |

| run | what | directory |
|---|---|---|
| generation | 9 rounds of LLM theory generation (multinomial objective, `2999e8b`) | [`results/xsl/mode_b/`](../results/xsl/mode_b/) |
| re-score | all 16 theories refit and cross-validated, binomial objective, `e70c10c` | [`results/xsl/binomial_e70c10c/`](../results/xsl/binomial_e70c10c/) |
| robust | top 10 refit with warm-started multi-start fits | [`results/xsl/binomial_e70c10c_robust/`](../results/xsl/binomial_e70c10c_robust/) |
| sensitivity | top 7 refit with the guessing rate capped at .10 | [`results/xsl/binomial_e70c10c_lapse10/`](../results/xsl/binomial_e70c10c_lapse10/) |
| random ability | top 10 refit on per-subject responses (robust fits) | [`results/xsl/subject_e70c10c_robust/`](../results/xsl/subject_e70c10c_robust/) |
| generation 2 | rounds 10–12 (`xsl_10`–`12`) under random ability | [`results/xsl/mode_b_subject/`](../results/xsl/mode_b_subject/) |
| honest re-evaluation | `xsl_10`–`12` with their hard-coded parameters freed | [`results/xsl/mode_b_subject_unpinned/`](../results/xsl/mode_b_subject_unpinned/) |
| debate | Phase 2, rounds 1–3 (`xsl_13`, `xsl_14`) | [`results/xsl/debate_xsl_12_vs_inc_fazly/`](../results/xsl/debate_xsl_12_vs_inc_fazly/) |
| gated debate | rounds 4–6 under the query gate; holds the current full leaderboard | [`results/xsl/debate_xsl_12_vs_inc_fazly_gated/`](../results/xsl/debate_xsl_12_vs_inc_fazly_gated/) |
| designs | Phase 3: queue of accepted designs with frozen predictions, `xsl_14` vs fazly | [`results/xsl/designs_xsl_14_vs_inc_fazly/`](../results/xsl/designs_xsl_14_vs_inc_fazly/) |
| designs 2 | the same for kachergis vs fazly | [`results/xsl/designs_kachergis_vs_inc_fazly/`](../results/xsl/designs_kachergis_vs_inc_fazly/) |

## Setup

| | |
|---|---|
| Data | `xsl_datasets`: 47 adult conditions exported from XSLmodels `e70c10c` to `data/xsl/`. The holdouts (`benitez2020`, `kachergis_initial_accuracy`, `suanda2014`, `koehne2013`) are sealed and unused. |
| Per-subject data | Each person's 0/1 test responses per condition, exported from XSLmodels `data-raw` (1,609 people, no missing responses). Every condition's per-item accuracy and `n_subj` are reproduced exactly. |
| Theory format | A prose claim plus `learn(params, trials, n_words, n_objects) → association matrix`, with a power-Luce test choice (`ch_dec`, if declared). Indices are shuffled before `learn` sees them, so the corpus's "word i ↔ object i" convention cannot be exploited. |
| Objective (final) | **Random ability**: person s has an offset u_s ~ N(0, `ability_sd`²) on the log-odds of the theory's predicted accuracy. Each person's likelihood is integrated over u_s, using exact flat tails plus composite Gauss–Legendre panels (within 1.4e-7 per person of a dense grid). `ability_sd` and the guessing rate `lapse` are fitted and counted in k. |
| Earlier objectives | **Multinomial** over all objects (generation rounds 1–9). **Binomial**: each item's correct count out of n. See *Why the objective changed*. |
| Fitting | Differential evolution. Robust fits use 3 starts, the first warm-started from the previous fit; each CV fold uses 2 starts warm-started from the full-corpus fit. New candidates warm-start from the running best and the leaderboard. |
| Score | Held-out NLL from 5-fold cross-validation over conditions, with the same split for every theory. Weighted SSE of held-out item accuracy is reported alongside. |
| Incumbents | NumPy ports of the xsl-bakeoff v05 panel (fazly, uncfam_gamma, uncfam_attention, tilles, Bayesian_decay) plus kachergis and a co-occurrence baseline. Each matches XSLmodels on all 47 conditions at its v05 fit. Hypothesis-testing models (propose-but-verify, pursuit) are not included: too slow to fit. |

## Current leaderboard

All theories under the random-ability likelihood, ranked by held-out NLL
(5-fold CV over conditions). k counts free parameters, including `lapse` and
`ability_sd`. "Gap" is held-out minus full-corpus NLL: a large gap means
overfitting, or folds landing in different optima.

| rank | theory | k | held-out NLL | held-out wSSE | gap | at a bound |
|---|---|---|---|---|---|---|
| 1 | `xsl_14` competition-scaled returns + pair-count commitment gain | 17 | **16,880.2** | **0.400** | +34 | `lam`, `xi` |
| 2 | `xsl_13` commitment-protected, consistency-scaled exclusion | 18 | 16,886.6 | 0.411 | +36 | `lam`, `xi`, `kappa`, `pg` |
| 3 | `xsl_12` commitment-gated exclusion + exposure scaling | 16 | 16,905.5 | 0.436 | +51 | `lam`, `xi` |
| 4 | `xsl_11` + count-based familiarity at test | 16 | 16,913.6 | 0.431 | +53 | `lam`, `Mmax`, `xi` |
| 5 | `xsl_09` commitment-gated claim exclusion | 15 | 16,921.7 | 0.433 | +59 | `lam`, `Mmax`, `xi` |
| 6 | `xsl_10` (= `xsl_09` once its pinned values are freed) | 15 | 16,922.0 | 0.433 | +59 | `lam`, `Mmax`, `xi` |
| 7 | `xsl_04` interference + test-time claim competition | 9 | 16,991.3 | 0.494 | +44 | `Mmax` |
| 8 | `xsl_05` + repetition boost | 10 | 16,995.9 | 0.497 | +34 | |
| 9 | `xsl_06` + consolidation protection | 11 | 16,998.6 | 0.493 | +33 | |
| 10 | `xsl_03` load-scaled absent-alternative interference | 7 | 17,050.6 | 0.546 | +38 | `q`, `ch_dec` |
| 11 | `xsl_08` decaying-trace contiguity interference | 13 | 17,061.6 | 0.509 | +115 | `q` |
| 12 | `xsl_07` soft-sharpness interference + contiguity | 11 | 17,065.7 | 0.517 | +130 | `b` |
| 13 | kachergis (Kachergis, Yu & Shiffrin 2012) | 5 | 17,135.1 | 0.689 | +34 | |
| 14 | fazly (Fazly, Alishahi & Stevenson 2010) | 4 | 17,153.6 | 0.504 | +24 | |
| 15 | uncfam_attention ‡ | 5 | 17,170.3 | 0.696 | +70 | |
| 16 | uncfam_gamma | 6 | 17,177.2 | 0.707 | +102 | |
| 17 | tilles (Tilles & Fontanari 2012) ‡ | 5 | 17,357.2 | 0.700 | +44 | |
| 18 | Bayesian_decay ‡ | 5 | 17,657.3 | 0.861 | +54 | `ch_dec` |
| 19 | co-occurrence baseline ‡ | 2 | 19,568.7 | 1.374 | +133 | |

‡ Standard fit, not a robust multi-start refit. Every theory except the
baseline fits `lapse` to its .001 floor (baseline: .35), so it is not listed
under "at a bound". `ability_sd` is 1.43–1.59 for every theory.

- **Two criteria, one winner.** Among incumbents, kachergis is best by
  held-out NLL but nearly worst by held-out wSSE, and fazly is the reverse.
  The random-ability likelihood scores how each person's responses hang
  together, not just mean item accuracy. The generated theories from
  `xsl_09` on lead on both criteria.
- **Over-parameterised.** The commitment-gated family fits `lam` and `xi` at
  their upper bounds in every version, and `Mmax` too through `xsl_11`.
  Its predictions are validated, but its parameter estimates are not
  interpretable until it is reparameterised.
- **Parameters change meaning with the objective.** Under random ability,
  fazly's fit (`lambda` .147, `beta` 21.8) returns almost exactly to the
  xsl-bakeoff SSE fit. uncfam_gamma's familiarity exponent flips from .69 to
  1.44. kachergis settles at B ≈ 5, between the two optima it alternated
  between under the binomial.

## How the theories evolved

All leading theories descend from `xsl_01`. What each step added:

1. **Mutual-responsibility credit (`xsl_01`).** A word's credit is split over
   the objects present, and each object's credit over the words present, in
   proportion to sharpened current strength. A pair gains the product of the
   two shares: soft mutual exclusivity and explaining-away during learning.
2. **Interference instead of global forgetting (`xsl_01` onward).** Only the
   links of words and objects on the current trial decay; later versions
   decay only their links to *absent* partners, at a rate that grows with
   trial load.
3. **Test-time claim competition (`xsl_04`).** At test, an object strongly
   claimed by other words is a poorer referent. This was the largest step
   under the multinomial objective. Under the accuracy objective it fits to
   zero in `xsl_04`–`06` and is active only alongside soft sharpening.
4. **Soft sharpening and temporal contiguity (`xsl_07`, `xsl_08`).** The
   sharpening exponent is fitted (≈ 0.7). Pairs repeated from the previous
   trial get amplified credit; in `xsl_08` this is a fading trace.
5. **Commitment with redistributed credit (`xsl_09`).** A word forms a graded
   working hypothesis once its strongest object dominates. The committed
   pair is strengthened, its object is withdrawn from the other words on the
   trial, and their credit is redistributed. This was the decisive step:
   about 70 held-out NLL ahead of the next family.
6. **Exposure scaling (`xsl_12`).** Encoding efficiency scales with a word's
   accumulated exposure. `kappa` fits at −0.41, meaning diminishing returns.
   `xsl_11`'s alternative, a weak count-based familiarity trace at test,
   helps less.
7. **Debate revisions (`xsl_13`, `xsl_14`).** `xsl_13` adds increasing returns
   for committed words and shields committed links from erosion; its `kpos`
   fits to .04, effectively off. `xsl_14` scales the diminishing returns by
   how ambiguous a word's association row is (inverse Simpson index), and
   makes the committed gain grow with the pair's own co-occurrence count
   (`kpos` .20, `kappa` −0.90).

Generation rounds 1–9 ran under the multinomial objective and rounds 10–12
under random ability. Appendix B has the round-by-round trajectory.

## Findings

### Why the objective changed

For n-AFC tests the multinomial likelihood splits exactly into two terms: an
accuracy term (binomial: correct or not) and a which-wrong-object term. In
the generation runs the latter made up 71–80% of the NLL, so fits traded
item-level accuracy for confusion structure. For example, kachergis fitted
this way predicted .41 accuracy on the hardest conditions, where people
score .16. The binomial objective scores accuracy directly, weighting each
item by its sample size (SSE does not). Under it, the incumbents' full-corpus
ranking is exactly the xsl-bakeoff v05 SSE ranking.

The binomial treats all choices as independent, though. Its fitted guessing
rates (.14–.38) were large, and capping them changed the ranking (Appendix
A). The random-ability likelihood models between-person variability
directly. With it, no theory needs any guessing (`lapse` at its .001 floor
in every fold of every theory, one `xsl_03` fold at .008). People differ by
roughly ±1.5 on the log-odds of their accuracy.

### Fitting reliability

- Single-start differential evolution was fine for the incumbents but missed
  the optimum of several 8–14-parameter theories by up to 421 NLL. `xsl_05`
  contains `xsl_04` as a special case, yet it had fitted 300 NLL worse.
  Multi-start, warm-started refits fixed most of this, and the loop now
  warm-starts every candidate.
- **Separate optima.** Better training fits need not generalise better. In
  one kachergis fold a refit moved to a different optimum: 41 NLL better on
  training, 109 worse held out. Held-out differences of about 100 can
  therefore reflect which optimum a fold lands in. Differences among the
  commitment-gated variants (up to about 40) are smaller than that.

### Leakage through pinned constants

`xsl_10`–`12` were presented as leaner versions of `xsl_09`. In fact they
declared six of its parameters as fixed numbers (`lam` 6, `Mmax` 30, `rho`
.15, `th` .79, `kc` 25, `xi` 1), which are exactly `xsl_09`'s full-corpus
fitted values, visible on the leaderboard. Because those values were fitted
on all 47 conditions, every CV fold inherited information from its own
held-out conditions, and the stated k understated the tuned parameters.

Refit with those six freed, `xsl_10` converges to exactly `xsl_09`. `xsl_12`
loses about 8 NLL of its gain but keeps a real 16-NLL improvement. The loop
now refits any constant that a candidate pins under a name a leaderboard
theory fits, and the prompt forbids hard-coding fitted values.

### Phase 2: debate between `xsl_12` and fazly

The two sides were `xsl_12`, the best generated theory, and fazly, the
incumbent closest to people's item accuracy and structurally different. Each
round runs three steps
([`main_xsl_debate.py`](../main_xsl_debate.py), [`src/xsl/debate.py`](../src/xsl/debate.py)):

1. **Queries.** Each side's advocate proposes a **corpus query**: a set of
   conditions plus a Python statistic of their per-item accuracies on which
   its theory should match people better. Predictions are population-mean
   accuracies under each theory's random-ability fit. A query counts only if
   the two predictions differ by at least 2 SE of the human value
   (bootstrapped over subjects). The theory closer to people wins it.
2. **Arbitration.** An arbiter reads both outcomes, the ledger of earlier
   queries and both theories' held-out NLL. It picks a theory to revise,
   either keeping its central claim (`new_model`) or replacing it
   (`new_theory`).
3. **Revision.** The generator revises that theory with up to 3 candidates,
   using the target itself as the bar. In rounds 1–3 a revision had to lower
   held-out NLL. In rounds 4–6 it had to pass the **query gate**: held-out NLL
   within 10 of the target's, *and* error on each query the target lost
   reduced by at least 1 SE.

Every query was separable on the first attempt, and each side won its own
query in every round. In all six rounds the arbiter revised the generated
theory (`new_model`), because fazly's held-out NLL is far worse.

| round | proposer | query | human ± SE | generated theory off by (SE) | fazly off by (SE) |
|---|---|---|---|---|---|
| 1 | `xsl_12` | drop from 1w×3o (222) to 1w×4o (221) | .201 ± .050 | −0.2 | −3.0 |
| 1 | fazly | mean of filt3E/6E/9E_9L | .589 ± .022 | −4.4 | +0.2 |
| 2 | `xsl_13` | mean of 216, 219 | .563 ± .043 | −0.2 | −2.5 |
| 2 | fazly | filt6E_6L, filt9E_6L, filt9E_9L, 2_x8_39_4x4, 3_x8_369_4x4 | .494 ± .021 | −3.4 | −0.1 |
| 3 | `xsl_13` | 207 (high ambiguity, near floor) | .103 ± .010 | +4.9 | +16.1 |
| 3 | fazly | late-repetition gain, filt\*_9L − filt\*_3L | .173 ± .026 | −5.1 | −1.1 |
| 4 | `xsl_14` | 225 minus 207 | .684 ± .055 | −3.7 | −6.9 |
| 4 | fazly | late-repetition gain at 6–9 early exposures | .177 ± .038 | −3.6 | −1.4 |
| 5 | `xsl_14` | 223 | .312 ± .029 | +0.9 | +4.2 |
| 5 | fazly | 2_x8_39_4x4, 3_x8_369_4x4, temp_cont_2olap2tr_contr, 218 | .409 ± .027 | −1.9 | +0.4 |
| 6 | `xsl_14` | 216 minus 223 | .267 ± .083 | −0.5 | −3.0 |
| 6 | fazly | filt6E_9L, filt9E_9L, 218, 2_x8_39_4x4, 3_x8_369_4x4 | .482 ± .023 | −3.1 | −0.1 |

Signed errors (prediction minus human) are for the generated theory in the
debate at the time: `xsl_12` in round 1, `xsl_13` in rounds 2–3, `xsl_14`
in rounds 4–6. Several statistics are differences between conditions.

The generated family wins on **load and ambiguity**: the cost of an extra
object per trial (fazly predicts a quarter of it), the 216/219 orderings,
and the floor in 207 (fazly is 16 SE too high). Fazly wins on **repetition**:
filtering designs and the mixed-frequency 4×4 conditions.

**Revisions.**

| round | gate | candidates | outcome |
|---|---|---|---|
| 1 | lower NLL | 1 | `xsl_13` accepted (16,905.5 → 16,886.6); its repetition gain `kpos` fits to .04 |
| 2 | lower NLL | 3 (16,889.9–16,898.9) | none beat `xsl_13` |
| 3 | lower NLL | 3 | `xsl_14` accepted (→ 16,880.2); `kpos` .20, `kappa` −0.90 |
| 4 | query | 3 (16,882.7–16,904.9) | none passed: lost-query error 3.6 → 3.6 SE for all three |
| 5 | query | 3 (16,886.0–16,911.5) | none passed: best query move 1.9 → 1.4 SE, at +31 NLL |
| 6 | query | 3 (16,889.7–16,914.7) | none passed: best query move 3.1 → 2.5 SE, at +34 NLL |

The accepted revisions improved overall prediction (−25 held-out NLL; wSSE
0.436 → 0.400) but barely moved the queries fazly won (−4.4 → −4.1 SE). In
round 4, every gated candidate predicted the same late-repetition gain as
`xsl_14` once fitted (.041–.043, against people's .177). The mechanisms
were in the code, and the fit switched them off. In rounds 5–6 the
candidates that moved fazly's queries most paid 31–34 held-out NLL for it,
including one that added a Fazly-style alignment channel. The candidates
that stayed within the NLL tolerance moved the queries by 0.2 SE or less.

**The filtering experiments.** These 12 conditions (338 people, 26% of all
test choices) cross early exposures (0, 3, 6, 9) with late exposures
(3, 6, 9). Mean accuracy:

| | human 3L / 6L / 9L | `xsl_14` 3L / 6L / 9L | fazly 3L / 6L / 9L |
|---|---|---|---|
| 0E | .37 / .49 / .59 | .50 / .55 / .58 | .51 / .63 / .69 |
| 3E | .45 / .48 / .58 | .46 / .48 / .51 | .47 / .57 / .62 |
| 6E | .42 / .55 / .59 | .45 / .47 / .50 | .46 / .54 / .59 |
| 9E | .41 / .51 / .60 | .45 / .47 / .50 | .45 / .53 / .57 |

People gain about .17 from 3 to 9 late exposures at every early level, and
early exposures cost nothing. **Both theories predict that early exposures
hurt**, in different ways. Fazly keeps the late gain but is too high with no
early exposure. `xsl_14` keeps the late gain only with no early exposure,
and is nearly flat (+.05) once there are any. The debate queries averaged
over early levels, so they saw only `xsl_14`'s problem.

**What the debate showed about the loop.**

- **Queries work.** Bootstrapped human SEs plus the 2-SE rule gave
  unambiguous, interpretable wins, and the advocates found them at once.
- **NLL-gated revision does not steer.** Fitting to the whole corpus is free
  to switch off the mechanism the debate asked for. The query gate makes the
  revision answer the query, and it shows the price: in this family, closing
  the repetition gap costs overall fit. Either the family's core (load-limited
  competitive encoding with commitment-gated exclusion) suppresses late
  gains after early exposure, or the corpus pulls in two directions. A model
  with both mechanisms would tell these apart.
- **The arbiter never chose fazly.** It always revised the side with better
  held-out NLL. A debate between theories closer in overall fit would test
  the arbiter more.

### Phase 3: prospective designs (`xsl_14` vs fazly)

The aim is a new experiment on which the two theories make clearly
different, pre-registrable predictions
([`main_xsl_design.py`](../main_xsl_design.py), [`src/xsl/design.py`](../src/xsl/design.py)).

1. **Proposal.** A designer (Sonnet 5.5) sees both theories, the corpus
   comparison and the debate ledger. It writes a design generator,
   `make_design(knobs, rng)`, with a few knobs, plus a pre-stated
   **contrast**: a statistic of per-item test accuracy.
2. **Validation.** Each design is checked against the runnable design space.
   Per arm: ≤ 36 words and objects, ≤ 108 trials, every word heard ≥ 2
   times, and ≤ 36-AFC tests with an explicit correct object that co-occurred
   with the word. There are at most 4 between-subject arms.
3. **Power search.** At every knob setting (3 random orders each),
   experiments with **50 people per arm** are simulated from each theory's
   random-ability fit. Each simulated experiment draws its parameters from
   the full fit or one of the five CV fold fits. The decision rule is
   pre-registered: the observed contrast favours whichever theory's
   prediction is on its side of the midpoint. Power is the smaller of the
   two rates of picking the theory that generated the data. The best setting
   is re-estimated with 2,000 fresh simulations per theory, so power is not
   inflated by selection.
4. **Acceptance.** At power ≥ .8 the design is frozen into the queue, hashed
   and timestamped, with every leaderboard theory's per-item predictions.
   Otherwise the designer sees the result and revises.

Four proposals produced three accepted designs. The rejected one was a
massed-vs-spaced repetition contrast: the two theories' predictions differed
by only .015, for power .70. Each search took 1–6 s.

| design | structure | contrast | `xsl_14` | fazly | power |
|---|---|---|---|---|---|
| **retroactive object reappearance** | 1 arm, 27 words × 27 objects, 63 trials. Block 1: 18 pairs in 2×2 trials, 4 times each. Block 2: 9 of the 18 old objects reappear as foils, 1 new word × 4 objects per trial, 27 trials. Test: the 18 old words, 18-AFC | control − foil-object pairs | +.156 | .000 | .993 |
| **synonym vs new label** | 1 arm, 27 words × 18 objects (w > o), 42 trials. Phase 1: 9 words learned 1×1. Phase 2 in 3×3 trials: 9 second labels for those objects and 9 labels for new objects, 4 times each. Test: the 18 phase-2 words, 18-AFC | new-object labels − second labels | +.186 | −.007 | .999 |
| **sparse vs dense, matched exposure** | 2 arms × 18 words × 18 objects, 6 exposures per word: 1 word × 2 objects (108 trials) vs 4×4 (27 trials) | sparse − dense mean accuracy | +.247 | +.043 | .964 |

**What every theory predicts.** The queue freezes all 19 leaderboard
theories' predictions, which shows what each design actually discriminates:

| contrast | predicts ≈ 0 | predicts a clear effect |
|---|---|---|
| retroactive reappearance | fazly, kachergis, uncfam_gamma, uncfam_attention, baseline (all .000); Bayesian_decay .02 | every interference theory: `xsl_03`–`14` +.13 to +.26, tilles +.33 |
| synonym vs new label | fazly −.01, `xsl_03`–`06` and `xsl_08` (≤ .03), Bayesian_decay, baseline; tilles +.06 | `xsl_09`–`14` +.11 to +.19, `xsl_07` +.15, kachergis and uncfam +.22 to +.28 |
| sparse vs dense | fazly .04, Bayesian_decay .06, baseline .08 | generated theories +.13 to +.28, kachergis and uncfam +.33 to +.34; tilles **−.20** |

- **Retroactive reappearance is the most valuable design.** It splits the
  field along one mechanism: whether learning a new word can erode an old
  word's link to the same object. Theories that update only the words heard
  on a trial predict exactly no effect, whatever their parameters. Every
  interference theory predicts a cost. The corpus has no condition that
  isolates this.
- **Synonym vs new label** is a mutual-exclusivity test with late-introduced
  second labels. It separates fazly (and the `xsl_04`–`06` subfamily) from
  `xsl_14` and the uncfam models. Published work on late-introduced and
  one-to-many mappings (including the sealed koehne2013 holdout) may already
  bear on it, so check before running it.
- **Sparse vs dense is largely answered already.** Its dense arm is close to
  corpus condition 223 (4×4, 27 trials, human .31; `xsl_14` predicts .34,
  fazly .43). Its sparse arm sits next to 222 (1 word × 3 objects, human
  .39). The search optimises discriminability, not novelty relative to the
  corpus, so a redundancy check is the obvious next filter.

#### Kachergis vs fazly

The same pipeline, run on the two best incumbents by held-out NLL. The
designer saw no debate ledger, since there was no debate for this pair.
All three proposals were accepted on their first attempt.

| design | structure | contrast | kachergis | fazly | power |
|---|---|---|---|---|---|
| **mutual exclusivity vs co-occurrence** | 1 arm, 18 words × 30 objects, 66 trials. Phase 1: 6 words learned 1×1, 3 times each. Phase 2: 1 word × 3 objects per trial, 4 times each. 6 "ME" words always appear with their novel target and 2 known objects; 6 control words with 3 novel objects. Test: 3-AFC over each word's trio | ME − control | +.476 | .000 | 1.000 |
| **hub vs private referent** | 1 arm, 12 words × 24 objects, 60 trials of 1 word × 3 objects, 5 per word; each word always appears with the same trio. Half the words' referents also appear in other words' trials (hubs); half are private, with hub companions. Test: 24-AFC | hub referent − private referent | −.143 | .000 | .978 |
| **trial order at matched counts** | 1 arm, 12 words × 24 objects, 72 trials, 1 word per trial. Each word has a referent R (on 5 of its 6 trials: 3 alone, 2 ambiguous) and a decoy D (on 3: 1 alone, 2 ambiguous), with the same counts in both groups. "Early" words see R alone first and D alone last; "late" words the reverse; 2 ambiguous [R, D] trials in between. Test: 24-AFC | late − early | +.165 | −.007 | .988 |

What every theory predicts:

| contrast | ≈ 0 | positive | negative |
|---|---|---|---|
| ME vs co-occurrence | fazly, baseline (.000); Bayesian_decay .04 | kachergis and uncfam +.47 to +.50, tilles +.35, `xsl_07` +.30, `xsl_09`–`14` +.18 to +.26, `xsl_03`–`06`, `xsl_08` +.09 to +.11 | |
| hub vs private | fazly, baseline (.000) | | every other theory: kachergis and uncfam −.14 to −.15, Bayesian_decay −.07, generated −.19 to −.36, tilles −.38 |
| trial order | fazly, baseline, Bayesian_decay, uncfam_gamma, tilles, `xsl_03`–`08` (all within ±.05) | kachergis +.17, uncfam_attention +.14 | `xsl_09`–`14` −.15 to −.23 |

- **One lever.** All three designs put a single word on each trial. Fazly
  normalises alignments only across the words on a trial, so with one word
  it reduces to co-occurrence counting. It predicts exactly the baseline's
  zero on any contrast that counting cannot see. This is a real mechanistic
  difference: fazly has no object-side competition. But the designer found
  one lever and used it three times, although the queue was in its prompt.
  The ME design's outcome is also close to foregone, given how robust mutual
  exclusivity is in people.
- **The rationale can be wrong while the design is fine.** For the hub
  design, the designer argued that kachergis's entropy-weighted attention
  would favour hub referents (+.4 to +.6). The computed prediction is the
  opposite (−.14): hub referents lose because other words also claim them.
  The design was accepted on the computed predictions, which the decision
  rule uses. The rationale shipped with a queued design must therefore be
  checked against those predictions before it is pre-registered.
- **The trial-order design is a three-way test.** The question is whether
  the first unambiguous exposure or the last one wins when counts are equal.
  Kachergis and uncfam_attention predict that late words are learned better
  (familiarity-weighted attention lets the decoy capture the ambiguous
  trials for early words). The commitment-gated family predicts the reverse
  (early commitment protects the referent). Fazly and the other counting-like
  models predict no difference. Of the six designs so far, it is the only
  one on which `xsl_14` and kachergis predict opposite signs.
- Cost ≈ $0.54 (3 LLM calls).

**Caveats (both runs).**
- Power assumes one of the two theories generated the data. In the filtering
  conditions people matched neither, so the pre-registration should add a
  third outcome: the human value outside both theories' predictive intervals.
- Several designs use structures absent from the corpus: 1×1 trials, more
  words than objects, and block structure. There the predictions extrapolate
  the fitted parameters.
- Within-arm item contrasts get high power from 50 people. Test order and
  item counterbalancing beyond random assignment are not modelled.
- The designer's mechanistic rationales are LLM claims. The predictions are
  computed, and the all-theories table is the check on them; for the
  retroactive design, for instance, fazly's prediction of exactly zero
  confirms its stated reason.
- Cost ≈ $0.37 (4 LLM calls, 40k input and 29k output tokens).

## Caveats

- **One lineage, one LLM.** Every leading theory refines `xsl_01`, and only
  Sonnet 5.5 has been used. The structure of the debate's failure suggests
  the lineage has hit a limit, not that the problem is unsolvable.
- **Over-parameterisation.** The commitment-gated family has parameters at
  their bounds (see the leaderboard). Its predictions are validated; its
  parameter values are not.
- **Adaptive overfitting of the CV split.** The generator saw per-condition
  held-out results on a fixed split every round, so the split itself may be
  overfit. The sealed holdouts are the guard and have not been used.
- **Unpenalised degrees of freedom.** Learners carry hard-coded constants
  the LLM chose after seeing the data. Constants that copy fitted values are
  now refitted, but other choices are not counted in k.
- **Fit instability.** Separate optima make differences of about 100 NLL
  fragile. Three incumbents and the baseline (‡) have standard, not robust,
  fits under random ability.
- **Incumbent panel.** Hypothesis-testing models are absent, and the
  debate's opponent was a three-parameter model.
- **Index exploitation ruled out.** Every theory gives identical NLL with
  indices shuffled, and the loop shuffles by construction.

## Cost

LLM cost from logged tokens at OpenRouter list price for Sonnet 5.5 ($2 / $10
per million input / output tokens):

| stage | LLM cost |
|---|---|
| generation rounds 1–9 | ≈ $3.8 (≈ $1.30 of it one runaway reply, now capped) |
| generation rounds 10–12 | ≈ $0.85 |
| debate rounds 1–3 | ≈ $1.04 |
| gated debate rounds 4–6 | ≈ $1.48 |
| Phase 3 designs (7 proposals) | ≈ $0.91 |
| re-scores, robust refits, sensitivity checks | none (local fitting) |
| **total** | **≈ $8.1** |

Compute dominates wall time: each debate round took about 4 h, almost all
of it fitting and cross-validating candidates. Early in the project the
OpenRouter key reported $32.39 spent, about $28 more than the logged tokens
explain. No hidden retries appear in the run logs; OpenRouter's Activity
page would settle it. The key ran out of credits once, after debate round 4;
the run resumed from its saved state.

## Next steps

1. **A structurally different theory for repetition.** The gated debate
   shows that patching the commitment-gated family does not produce "late
   repetitions help, early exposures don't hurt". Seed generation from fazly,
   adding load-dependent encoding, or ask for a hybrid with both mechanisms to
   measure the trade-off directly.
2. **Phase 3, next.**
   - Add a corpus-redundancy check: reject designs whose contrast the
     existing conditions already pin down.
   - Add a "neither theory" outcome to the decision rule.
   - Export the retroactive-reappearance and trial-order designs for a
     human run.
   - Push the designer toward diversity: queued designs are in its prompt,
     but it reused the one-word-per-trial lever three times.
   - Target the open filtering interaction (early × late exposure), where
     both theories are wrong.
   - Run `xsl_14` vs kachergis: they agree in sign on four of the six
     contrasts so far.
3. **Reparameterise the commitment-gated family** (`lam` and `xi` at their
   bounds), and fix `lapse` at 0 under random ability: one fewer parameter
   for every theory.
4. **Robustly refit** the theories that only have standard fits (‡).
5. **Evaluate once on the sealed holdouts**, reporting adults and children
   separately.
6. Try a second generator LLM, and a debate between two closer theories
   (e.g. `xsl_14` vs `xsl_04`).
7. Mode A, sequential unsealing.

## Appendix A: binomial-objective results

Item-count binomial with a freely fitted guessing rate (robust fits unless
marked), and the same with the guessing rate capped at .10. NLLs are on a
different scale from the random-ability tables.

| rank | theory | k | held-out NLL | held-out wSSE | lapse | held-out NLL, lapse ≤ .10 | fit |
|---|---|---|---|---|---|---|---|
| 1 | `xsl_09` | 14 | **3,192.8** | **0.400** | .23 | **3,288.5** | robust |
| 2 | fazly | 3 | 3,504.5 | 0.492 | .15 | 3,486.8 | robust |
| 3 | `xsl_08` | 12 | 3,519.2 | 0.486 | .20 | 3,392.6 | robust |
| 4 | `xsl_07` | 10 | 3,534.1 | 0.494 | .20 | 3,396.3 | robust |
| 5 | `xsl_03` | 6 | 3,551.4 | 0.499 | .31 | 3,571.8 | robust |
| 6 | `xsl_05` | 9 | 3,566.9 | 0.501 | .29 | — | robust |
| 7 | uncfam_gamma | 5 | 3,593.4 | 0.505 | .23 | 3,685.0 | robust |
| 8 | `xsl_04` | 8 | 3,607.0 | 0.514 | .29 | — | robust |
| 9 | `xsl_02` | 6 | 3,652.4 | 0.536 | .23 | — | single-start |
| 10 | `xsl_01` | 5 | 3,652.6 | 0.526 | .34 | — | single-start |
| 11 | `xsl_06` | 10 | 3,678.1 | 0.517 | .31 | — | robust (lower bound) |
| 12 | kachergis | 4 | 3,916.8 | 0.604 | .28 | 4,018.1 | robust |
| 13 | tilles | 4 | 3,931.8 | 0.600 | .30 | — | single-start |
| 14 | uncfam_attention | 4 | 3,987.2 | 0.622 | .29 | — | single-start |
| 15 | Bayesian_decay | 4 | 4,037.8 | 0.647 | .38 | — | single-start |
| 16 | co-occurrence baseline | 1 | 6,716.9 | 1.353 | .28 | — | single-start |

**The guessing cap changes conclusions under the binomial.** Fazly, `xsl_07`
and `xsl_08` fit well with little guessing, and they predict held-out
conditions *better* with the cap. uncfam_gamma, kachergis and `xsl_03` lean
on guessing, compensate in other ways when capped, and predict worse.
`xsl_09` also does worse (+96) but stays first. The ranking between fazly
and `xsl_07`/`08` flips with the cap. The random-ability likelihood resolved
this.

**Where `xsl_09` beat fazly** (binomial, held-out NLL difference; negative
means `xsl_09` is better): −311.7 in total, winning 33 of 47 conditions. The
largest gains were 215 (1×3, −97.5), 206, 220, 223, 216 and 201. The losses
were concentrated in short filtering designs (filt0E_3L +42.6), 301
(+38.3) and the frequency-manipulated 4×4 conditions (+20–22). This is the
same repetition weakness the debate later isolated.

## Appendix B: generation trajectory, rounds 1–9

Multinomial objective on the `2999e8b` export (not comparable with the
tables above).

| round | accept gate | theory | held-out NLL (multinomial) | Δ vs previous best | LLM cost |
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

† Includes one runaway reply: 128k output tokens of invented tool-call
markup.

Under BIC, rounds 5 and 6 produced theories with better BIC than `xsl_04`
but worse held-out NLL. The group likelihood treats all 31k choices as
independent, so the log(N) penalty per parameter is small next to the NLL
differences at stake. Part of their poor held-out scores was also
single-start fitting.

Rounds 10–12 under random ability (as generated → honest, after freeing
pinned values): `xsl_10` 16,913.8 → 16,922.0; `xsl_11` 16,914.5 → 16,913.6;
`xsl_12` 16,897.6 → 16,905.5. A labelling bug in the first attempt (a second
`xsl_08`) was caught after one round. Labels now continue from the highest
number, and duplicate labels cannot be saved.

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

# generation under the random-ability objective (the default), seeded with those
# fits plus the remaining incumbents
cp results/xsl/subject_e70c10c_robust/leaderboard.json results/xsl/mode_b_subject/
uv run python main_xsl_offline.py --n_rounds 0 --incumbents --out results/xsl/mode_b_subject
uv run python main_xsl_offline.py --llm_provider openrouter --llm_model anthropic/claude-sonnet-5.5 \
    --n_rounds 3 --max_iters 3 --out results/xsl/mode_b_subject

# Phase 2 debate, from the honest leaderboard (resumable; --rounds adds rounds)
uv run python main_xsl_debate.py --a xsl_12 --b inc_fazly --rounds 3 \
    --llm_provider openrouter --llm_model anthropic/claude-sonnet-5.5

# gated continuation: copy that run's state, then 3 more rounds
cp -R results/xsl/debate_xsl_12_vs_inc_fazly results/xsl/debate_xsl_12_vs_inc_fazly_gated
uv run python main_xsl_debate.py --a xsl_12 --b inc_fazly --gate nll_and_query --rounds 3 \
    --out results/xsl/debate_xsl_12_vs_inc_fazly_gated \
    --llm_provider openrouter --llm_model anthropic/claude-sonnet-5.5
```

```bash
# Phase 3 prospective designs (resumable; --n_designs adds design slots)
uv run python main_xsl_design.py --a xsl_14 --b inc_fazly --n_designs 3 --max_iters 4 \
    --llm_provider openrouter --llm_model anthropic/claude-sonnet-5.5
uv run python main_xsl_design.py --a kachergis --b inc_fazly --debate results/xsl/designs_kachergis_vs_inc_fazly \
    --n_designs 3 --max_iters 4 --llm_provider openrouter --llm_model anthropic/claude-sonnet-5.5
```

Every LLM exchange is logged under `results/xsl/<run>/round_NN/prompts/`
(debate revisions under `round_NN/revision/prompts/`), and each theory's
code is in `results/xsl/<run>/theories/<label>.yaml`.
