# Offline AutoCog for cross-situational word learning (XSL)

Status: proposal, Phase 0 in progress (2026-10-14).

A variant of AutoCog that collects **no new data**. It iterates on theories of
cross-situational word learning and tests them against two evidence streams:

1. **Retrospective** — the pooled human data in the
   [XSLmodels](https://github.com/kachergis/XSLmodels) package (47 conditions,
   ~1,600 adults), and
2. **Prospective** — new experimental designs proposed by the loop, compared
   against frozen predictions from the previous best-fitting models (the
   xsl-bakeoff "incumbents").

## 1. What carries over from AutoCog

AutoCog runs an adversarial debate: two LLM agents each own a theory (prose +
executable model), propose an experiment and an LLM-written metric where their
theory should win, collect data, and an Arbiter routes the losing slot to the
Improver (`new_model`) or TheoryGenerator (`new_theory`), each of which runs a
critique loop judged on `aggregate_loss` over the whole evidence pool.

Reused unchanged: `src/llm.py`, the Promptable / exec-source pattern,
`Observations` / `Round` persistence and backfill, Arbiter, Improver,
TheoryGenerator, the Feedback critique loop, leaderboard plumbing, prompt logs.

Three things do **not** carry over:

- **Theories are never fitted in AutoCog** — parameters are sampled from prior
  ranges per simulated subject, so every score is prior-predictive. XSL models
  are conventionally group-fitted; this variant fits.
- **AutoCog theories are trial-by-trial choice models** (`predict` / `policy`).
  XSL training trials have no response; a learner maps a training sequence to a
  word × object association matrix, then a choice rule scores the test.
- **"Collect data" becomes "read the corpus"** (stream 1) or "freeze a
  prediction" (stream 2).

## 2. Two evidence streams

| | Stream 1: retrospective corpus | Stream 2: prospective designs |
|---|---|---|
| Replaces "run the experiment" with | a query over existing conditions | a new training order |
| Outcome | human per-item accuracy and choice counts | incumbent predictions at their fitted parameters |
| Scores correctness? | yes | no — a reference, not ground truth |
| Role | fitness: leaderboard and accept gates | novelty and critical experiments; a frozen queue to run later |

Stream 2 rewards **divergence from the incumbents at matched corpus fit**. A
theory that matches the incumbents on the corpus *and* on every new design is a
notational variant; one that diverges on new designs but fits the corpus worse
is dominated; one that fits equally well and diverges defines a critical
experiment. Treating incumbent predictions as truth would pull every theory
back toward the incumbents, so they are never used as a loss target.

Once a queued design is run on humans, it joins the corpus as a new condition
and its frozen predictions are scored — the prospective stream turns into the
retrospective one.

## 3. Decisions

| Question | Decision |
|---|---|
| Fit level | **Group.** Multinomial log-likelihood on each condition's `response_matrix` for the leaderboard and accept gates; `n_subj`-weighted SSE (the bakeoff objective) reported alongside. Significance judgements use condition-level CV, not the group likelihood (which treats choices as independent and is overconfident). |
| Guessing | **A guessing rate `lapse` is fitted for every theory** (mixture with uniform choice at test, counted in k). With a fixed small lapse the likelihood ranks theories by how they handle zero associations, not by mechanism: at lapse = .001 raw co-occurrence counting scores worse than uniform guessing. |
| m-AFC likelihood | Binomial on correct vs incorrect. Conditions 211 and 214 record choices of objects outside their stored candidate lists (4 and 175 choices), so a multinomial over candidates is not usable there; worth checking in XSLmodels. |
| Accept gate | BIC on the full-corpus fit (one fit ≈ 1.5–3 min). k-fold CV over conditions is run for the leaderboard (`--cv`), not inside the critique loop, because it costs k fits. |
| Incumbents | **Top-k by cross-validated fit in bakeoff v05** — see §4. |
| Human-runnable designs | **Yes.** Designs are constrained to an online-feasible paradigm (§5); planned N = 50. |
| Repo | **This fork, `src/xsl/`.** README §8 (Scope) gets updated; `tests/test_no_category_learning.py` only blocks category-learning imports and needs no change. |
| First debate mode | **Mode B (full corpus)**, then mode A (sequential unsealing). |

## 4. Incumbent panel

v05 cross-validation has not been run yet (the older `cv_*.Rdata` predate the
2026-10-13 freq/CD data fix), so the interim panel uses v05 **group** fits at
XSLmodels `2999e8b` (`xsl-bakeoff/fits/v05/adults/group_fits.Rdata`), to be
replaced by the CV ranking when it exists.

v05 group fits, weighted SSE (lower is better), top of the table:

| Model | SSE | k | Notes |
|---|---|---|---|
| fazly | 0.477 | 2 | |
| fazlyt | 0.477 | 3 | duplicates fazly; group-fit only |
| uncfam_gamma | 0.512 | 4 | γ̂ = 0.60; **nests kachergis** (γ = 1) |
| kachergis | 0.558 | 3 | C at its upper bound (1.0) |
| uncfam_elimination | 0.558 | 4 | ≈ kachergis |
| uncfam_attention | 0.563 | 3 | B at its upper bound (15) |
| tilles | 0.615 | 3 | |
| novelty | 0.622 | 3 | uncfam variant |
| Bayesian_decay | 0.634 | 3 | |

Interim panel: **fazly, uncfam_gamma, uncfam_attention, tilles,
Bayesian_decay.** kachergis is dropped from the panel because uncfam_gamma
contains it as the γ = 1 special case and fits better; Bayesian_decay takes the
fifth slot over novelty for family diversity (novelty is another uncfam
variant). kachergis remains the reference port and a seed theory.

Panel rules:

- The panel is a versioned config: XSLmodels commit, fits snapshot hash, model
  list, parameters.
- De-duplicate by **predictions**, not names: two incumbents that agree on
  every proposed design add nothing to the divergence score.
- Every incumbent is associative; hypothesis-testing models fit much worse
  (pursuit 1.45, trueswell2012 2.03). So the hypothesis-testing family belongs
  in the **seed theories** of the debate (kachergis vs propose-but-verify /
  pursuit), not in the panel.

## 5. Design space for prospective experiments

Designs must be runnable online and comparable with the corpus:

| Knob | Allowed | Corpus |
|---|---|---|
| Words | up to 36 | 12–24 (18 in 34/47) |
| Objects | up to 36 | 12–24 |
| Words vs objects | either asymmetry allowed (w > o or o > w) | mostly w = o |
| Trials | up to 108 | 18–108, median 36 |
| Words × objects per trial | e.g. 1×3 … 4×4, w ≤ o or w > o | 1×3 … 4×4 |
| Test | n-AFC up to 36-AFC; default shows **all** objects | 18-AFC, some 4-AFC |
| Planned N | 50 | median 31 (13–104) |

- The default test shows every object (n-AFC = number of objects). Smaller
  foil sets are used when non-1-to-1 associations are tested.
- Because asymmetric and non-1-to-1 designs are allowed, a design carries an
  **explicit word → correct-object(s) mapping** rather than relying on the
  corpus's "word i ↔ object i" diagonal convention. The Python scorer
  supports explicit mappings for every condition from Phase 0 on.
- Validator: minimum appearances per word and per object, no repeats within a
  trial, index ranges, mapping consistent with the training co-occurrences.
- The LLM writes a **design-generator function** (source code, like metrics
  today) rather than listing up to 108 trials by hand. Optionally the LLM sets
  constraints and the existing xsl-bakeoff annealer
  (`fits/discriminative_design_search_v2.R`) optimises the order.
- Acceptance reuses AutoCog's Welch check at N = 50, with binomial subject noise
  plus parameter uncertainty from CV folds.
- The existing online pipeline (sweetbean / Prolific) has no XSL trial type;
  the queue's export format should target whatever runner is used (e.g. the
  lab's jsPsych timelines). Out of scope for v1.

## 6. Architecture

```text
src/xsl/
  corpus.py        load data/xsl/ JSON into Condition objects
  models.py        NumPy reference ports (baseline, kachergis/uncfam, ...)
  scoring.py       Luce choice (get_perf), m-AFC test scoring (mafc_test), likelihood
  incumbents.py    Rscript bridge: designs JSON -> xsl_run() -> predictions JSON
  theory.py        XSLTheory: learn(params, trials, n_words, n_objects, rng) -> W x O matrix
  fitting.py       scipy differential_evolution; CV over conditions
  experiment.py    CorpusQuery (stream 1), XSLDesign (stream 2)
theories/xsl/      seed YAMLs
data/xsl/          JSON export + manifest (XSLmodels commit, hashes)
scripts/xsl/       export_corpus.R, run_incumbents.R
main_xsl_offline.py
results/xsl/
```

- **XSLTheory** replaces `predict` / `policy` with
  `learn(params, trials, n_words, n_objects, rng) -> association matrix` plus a
  choice rule (Luce by default, overridable). Stochastic theories average over
  `n_sim` runs, as `xsl_run` does.
- **Fitting**: `scipy.optimize.differential_evolution` (mirrors DEoptim). Small
  budget, warm starts and a hard timeout inside the critique loop; full CV only
  at accept gates and for the leaderboard.
- **CorpusQuery** (stream 1): the agent picks a subset of conditions and writes
  a diagnostic contrast metric (e.g. high − low contextual diversity), and must
  state its theory's prediction before values are revealed.
- **XSLDesign** (stream 2): validated generator output; predictions from every
  theory and incumbent are frozen, hashed and timestamped into a
  critical-experiment queue.
- **Incumbent bridge**: a subprocess `Rscript` call against the pinned library
  (`xsl-bakeoff/fits/v05/Rlib`), JSON in and out. Imported lazily so entry
  points import without R.

## 7. Modes

- **B. Full corpus** (build first): every condition visible; the debate, theory
  generation and stream 2 run against the whole corpus.
- **A. Sequential unsealing** (later): debate agents see designs but not
  outcomes; each round unseals a few conditions (47 conditions ≈ 10–20 rounds).
  Gives a data-efficiency readout against the bakeoff fit on everything.

## 8. Risks and guards

- **Overfitting a fixed corpus** across many rounds and metrics: the scorable
  holdouts (`benitez2020`, `kachergis_initial_accuracy`, `suanda2014`,
  `koehne2013`) are sealed — never shown to the LLM, used only for final
  evaluation (`load_holdout` refuses without `unseal=True`); CV over conditions
  at accept gates; a parameter-count penalty; the leaderboard uses one
  pre-declared objective, never the LLM's own metrics. `benitez2020` mixes
  adult and child conditions (4 of its 7 are kids), while the corpus and the
  v05 fits are adults-only — report holdout results for adults and children
  separately.
- **The LLM knows this literature**: measure novelty in prediction space
  (distance from incumbents on stream-2 designs), not by how the prose reads.
- **Compute**: deterministic ports first; Numba / parallelism over conditions
  if needed; hard per-candidate timeouts on LLM code.
- **Port fidelity**: analytical parity tests against R outputs (Phase 0).

## 9. Phases

0. **Data bridge.** Rscript export (pinned to `2999e8b`) of `xsl_datasets`, the
   scorable holdouts, the incumbent fits, and R reference predictions →
   `data/xsl/` with a manifest. Python loader; NumPy ports of `baseline` and
   `kachergis`; parity tests (≤ 1e-8 on all 47 conditions); Rscript incumbent
   bridge with a round-trip test against the stored v05 `ModelPerf`.
   **Status: done (2026-10-14).**
   - `scripts/xsl/export_corpus.R` → `data/xsl/` (`corpus.json`,
     `holdout.json`, `incumbents.json` for the 17 v05 models with plain
     constructors, `reference_predictions.json`, `manifest.json`).
     kachergis_sampling had no v05 group fit yet; strength, rescorla-wagner
     and fazlyt are custom wrappers in `specs.R` and are not exported.
   - `src/xsl/corpus.py`, `models.py` (baseline, uncfam with optional γ),
     `scoring.py` (`get_perf`, `mafc_test`, weighted SSE),
     `incumbents.py` + `scripts/xsl/run_incumbents.R`.
   - Ports match R to ≤ 2.2e-13 on all 47 conditions at three parameter
     settings; the Python weighted SSE reproduces v05 `bestval` exactly
     (kachergis 0.558129387959377, baseline 1.628014147617955). A deterministic
     port costs ~2 ms per condition (~0.1 s for the corpus).
   - The bridge reproduces v05 `ModelPerf` for the panel + kachergis + baseline
     on all 47 conditions (~2 s per call), and matches the Python port on a new
     36-word / 12-object synonym design with an explicit 12-AFC test.
1. **Smallest end-to-end loop.** Improver / TheoryGenerator in mode B against
   the fitted corpus objective, no debate.

   **Status: built, not yet run with an LLM (2026-10-14).** XSL-native rather
   than the decision-making `TheoryGenerator`, whose loop is bound to
   trial-by-trial `predict`/`policy` theories and scalar observations; it keeps
   the same accept-gate contract and reuses the LLM clients and parsing.
   - `src/xsl/theory.py` (`XSLTheory`: prose + `learn`, validated for shape,
     sign, determinism and speed), `fitting.py` (group NLL, fitted `lapse`,
     differential evolution, BIC, k-fold CV), `prompts.py`, `generator.py`,
     `main_xsl_offline.py`; seeds `theories/xsl/{baseline,kachergis}.yaml`.
   - Seed leaderboard (full corpus, N = 31,266 choices):

     | theory | k | BIC | NLL | wSSE | parameters |
     |---|---|---|---|---|---|
     | kachergis | 4 | 32,391.7 | 16,175.2 | 0.567 | X = .241, B = 1.46, C = .973, lapse = .362 |
     | baseline | 1 | 39,486.6 | 19,738.1 | 1.366 | lapse = .388 |

     The likelihood fit puts B near 1.5, where the v05 SSE fit had B = 14.2:
     the likelihood also scores which wrong object people choose.
   - Tests cover closed-form likelihoods, parameter recovery (incl. `lapse`),
     seed YAMLs = reference ports, the accept gate / retries / critic with a
     fake LLM client, and leaderboard persistence.
   **First runs (2026-10-14), anthropic/claude-sonnet-5.5 via OpenRouter,
   6 rounds × max 3 iterations, ≈ $3 total** (incl. one $1.30 runaway reply —
   now capped). Data: XSLmodels 2999e8b, i.e. before PR #31 fixed the 211/214
   test alternatives; re-score after re-exporting at e70c10c.

   | theory | k | BIC (full) | held-out NLL (5-fold) | held-out wSSE |
   |---|---|---|---|---|
   | xsl_04 test-time object-claim competition | 8 | 31,165.9 | **15,879.4** | **0.511** |
   | xsl_05 + consecutive-repetition boost | 9 | **31,077.1** | 16,040.9 | 0.547 |
   | xsl_06 + consolidation protection | 10 | 31,093.9 | 16,095.6 | 0.545 |
   | xsl_02 verify-and-consolidate | 6 | 32,002.0 | 16,104.9 | 0.538 |
   | xsl_03 load-scaled interference | 6 | 31,863.2 | 16,128.6 | 0.563 |
   | xsl_01 interference-limited mutual responsibility | 5 | 32,045.4 | 16,146.9 | 0.590 |
   | kachergis | 4 | 32,391.7 | 16,274.8 | 0.598 |
   | baseline | 1 | 39,486.6 | 19,786.1 | 1.381 |

   Lessons: (1) BIC on the group likelihood admits overfitting — xsl_05/06
   beat xsl_04 on BIC but lose to it held out, so the accept gate (or at
   least the leaderboard the LLM sees) should use CV. (2) The gains are
   uneven: xsl_04 beats kachergis held out in 28 of 47 conditions and loses
   most on the filtering conditions and 301. (3) Several xsl_04 parameters
   are weakly identified (fold-to-fold Mmax 0.3–9, q 0.33–0.96). (4) All
   theories are invariant to index shuffling (no diagonal exploitation); the
   loop now shuffles by construction.

2. **Retrospective debate.** Seeds kachergis vs propose-but-verify / pursuit.

   **Status: built and run (2026-10-18), without hypothesis-testing models**
   (too slow to fit): `xsl_12` vs fazly, 3 rounds. Advocates propose corpus
   queries (conditions + a statistic of per-item accuracy) that count only
   at ≥ 2 bootstrapped human SE between the theories' predictions; an arbiter
   picks the theory to revise; the generator revises it with the target as
   the bar on held-out NLL (`main_xsl_debate.py`, `src/xsl/debate.py`).
   Result: two accepted revisions (`xsl_14`, held-out NLL 16,880.2, new best),
   but the repetition failure fazly exposed (filtering experiments) persists.
   Three more rounds with `--gate nll_and_query` (a revision must close the
   query it lost, within 10 held-out NLL) accepted none of 9 candidates; see
   the mode-B report.
3. **Prospective stream.** Design generator + validator, incumbent service,
   divergence-at-matched-fit acceptance, the queue.

   **Status: built and run (2026-10-19)** for `xsl_14` vs fazly
   (`main_xsl_design.py`, `src/xsl/design.py`). Instead of
   divergence-at-matched-fit, acceptance is the power of a pre-stated
   contrast at N = 50 per arm. Experiments are simulated from each theory's
   random-ability fit, with parameter uncertainty from the CV folds, and
   decided by a pre-registered midpoint rule. The designer writes a design
   generator with knobs, and a local search picks the knob setting. Three of
   four proposals were accepted and queued with all 19 theories' frozen
   predictions. Runs for kachergis vs fazly and `xsl_14` vs kachergis
   accepted 3 of 3 each.
   Results are in the mode-B report.
4. **Evaluation.** Holdout vs the bakeoff winner; ablations (no stream 2; A vs
   B); recovery checks with an incumbent as ground truth. Then mode A.
