# Offline AutoCog for cross-situational word learning: summary

*2026-10-20; revised 2026-11-16 and 2026-11-19 after scoring checks. Full report: [`xsl_mode_b_report.md`](xsl_mode_b_report.md); design:
[`xsl_offline_proposal.md`](xsl_offline_proposal.md); branch `xsl-offline-phase0`.*

## What was done

We built a variant of AutoCog that collects no new data. An LLM (Claude
Sonnet 5.5) proposes theories of cross-situational word learning as prose
plus a `learn()` function. Each theory is fitted to the XSLmodels corpus and
compared with the xsl-bakeoff models. The project then debates the theories
and designs new experiments to tell them apart.

- **Data bridge (Phase 0).**
  - Exported 47 adult conditions from XSLmodels: 1,609 people and 31,266
    test choices, including per-person responses.
  - Ported the v05 incumbents (fazly, kachergis, uncfam_gamma,
    uncfam_attention, tilles, Bayesian_decay) to NumPy. The ports match R to
    within 2e-13.
  - Sealed the holdout sets; they have not been used.
- **Theory generation (Phase 1).**
  - 12 rounds of propose → fit → critique produced 12 theories (`xsl_01`–`12`).
  - Every theory is scored by 5-fold cross-validated held-out NLL over
    conditions.
  - Along the way:
    - the accept gate moved from BIC to cross-validation;
    - fitting became multi-start and warm-started;
    - the objective moved from multinomial, to binomial item accuracy, to a
      per-subject random-ability likelihood (logit-normal person offsets);
    - the loop now refits parameters the LLM had hard-coded at fitted values.
- **Debate (Phase 2).** Six rounds pitted the best generated theory against
  fazly, the incumbent closest to people's item accuracy.
  - Each side proposed a corpus query: a statistic on which it expects to
    match people better. A query counts only if the predictions differ by at
    least 2 bootstrapped standard errors.
  - An arbiter chose which theory to revise. Revisions became `xsl_13` and
    `xsl_14`.
  - The last three rounds used a stricter gate: a revision had to close the
    query its theory lost.
- **Prospective designs (Phase 3).** For each pair of xsl_14, kachergis and
  fazly:
  - the LLM wrote design generators with a pre-stated contrast;
  - a search simulated experiments with 50 people per arm from each fitted
    theory (with parameter uncertainty);
  - designs with power ≥ .8 were frozen into a hashed queue, with all 19
    theories' predictions.

LLM cost was about $8.4 in logged tokens. Everything else ran locally.

## Overall results

1. **Generated theories out-predict every incumbent.**
   - The leaderboard's first theory, `xsl_14` (17 parameters), reaches
     held-out NLL 16,880.2, against 17,135.1 for kachergis and 17,153.6 for
     fazly.
   - It also has the lowest held-out weighted SSE (.363; fazly .505,
     kachergis .564).
   - All 12 generated theories refitted under the final objective
     (`xsl_03`–`14`) rank above all six incumbents.
   - **This lead survived three scoring checks**: cold-started folds, folds
     that hold out whole experiments, and a bootstrap over conditions (95%
     interval −381 to −139 against kachergis). `xsl_14` stays at least 205
     NLL ahead of both incumbents.
   - **The order at the top did not survive.** Refitted from cold starts at
     a larger budget, `xsl_09`–`14` fall within 25–30 NLL of each other, and
     the leader changes with the split (`xsl_13` on random folds, `xsl_12`
     when whole experiments are held out). `xsl_14`'s leaderboard lead came
     mostly from warm-started folds. They are a tied cluster, so the later
     steps (exposure scaling, the debate revisions) are not established
     improvements.
   - They form one lineage built on `xsl_01`: mutual-responsibility credit,
     interference instead of global decay, test-time competition for
     objects, contiguity, commitment with redistributed credit (`xsl_09`, the
     decisive step), and exposure scaling.
2. **Individual differences, not guessing.**
   - Once each person's ability is modelled (SD 1.4–1.6 logits), every
     theory drives the guessing rate to zero.
   - Earlier fits had attributed 14–46% of choices to guessing, which was
     really between-person variability.
   - The choice of objective also reorders the incumbents. Fazly ranks 2nd
     of 16 under item accuracy but 14th of 19 under the random-ability
     likelihood.
3. **The debate found a persistent blind spot.**
   - The generated family wins every query on load and ambiguity, such as
     the cost of an extra object per trial and near-floor high-ambiguity
     designs. Fazly misses these by up to 16 SE.
   - Fazly wins every query on repetition. In the filtering experiments
     people gain about .17 from late repetitions, regardless of earlier
     exposures. `xsl_14` predicts about .04.
   - Two qualifications. The advocates saw the human data when choosing
     queries, so each side winning its own is expected. And fazly's match
     depends on having been fitted to other filtering conditions: with that
     experiment held out whole, fazly predicts it worst of the four theories
     tested.
   - Under the stricter gate, none of 9 revisions fixed this. Those that
     moved the query most cost 31–34 held-out NLL. Within this lineage,
     fitting the repetition effect costs fit elsewhere.
4. **Nine experiments are queued, and three stand out.** All were accepted
   at power ≥ .84, eight at ≥ .96. Three are ready for a human run:
   - **Trial order at matched counts** separates all three theories:
     `xsl_14` −.18, kachergis +.17, fazly .00.
   - **Mutual exclusivity with known objects** also separates all three:
     .26 / .48 / .00.
   - **Retroactive object reappearance** isolates interference. All
     interference theories predict a cost of .13–.33; fazly, kachergis and
     the uncfam models predict exactly zero.
   - Three other designs are largely answered by existing corpus conditions.
     In the clearest case, kachergis is already refuted by condition 221.
5. **Process lessons.**
   - Fitting reliability mattered as much as the theories: single-start
     fits missed optima by up to 421 NLL.
   - BIC admitted overfitting.
   - The LLM leaked held-out information by hard-coding fitted values.
   - Whole-corpus fitting can silently switch off a mechanism the debate
     asked for.
   - The designer's rationales are sometimes wrong in sign, so only computed
     predictions should be trusted.
   - Cross-validation folds were warm-started from the full-corpus fit and
     split sibling conditions across folds. Both flattered some scores, and
     a metric (weighted SSE) was computed against the wrong prediction until
     a review caught it.

## Obvious next steps

0. **Change the score.** Make cold-started CV on folds grouped by
   experiment the default accept gate and leaderboard score, with several
   fold seeds and intervals on NLL differences. On one split, fold-level
   optimiser noise (about 20 NLL) is as large as the gaps in the top
   cluster.
1. **Run the three strongest designs on people.**
   - Export the trial-order, mutual-exclusivity and retroactive-reappearance
     designs to a jsPsych timeline.
   - Pre-register them with the frozen predictions.
   - Add a "neither theory" outcome, since people matched neither theory on
     the filtering data.
   - The results become new corpus conditions and a direct test of the
     queue.
2. **Tighten Phase 3.**
   - Add a corpus-redundancy check that rejects designs the existing data
     already decide.
   - Push the designer toward diverse designs. Three kachergis vs fazly
     designs reused the same one-word-per-trial lever.
   - Target the open filtering interaction (early × late exposure), where
     both leading theories are wrong.
3. **Break out of the single lineage.**
   - Seed generation from fazly with load-dependent encoding, or from a
     hybrid with both mechanisms, to measure the repetition-versus-load
     trade-off directly.
   - Try a second generator LLM.
4. **Evaluate once on the sealed holdouts**, reporting adults and children
   separately. This guards against having overfit the fixed CV split over
   many rounds.
5. **Housekeeping.**
   - Reparameterise the commitment-gated family, where `lam` and `xi` sit at
     their bounds.
   - Fix the guessing rate at 0.
   - Robustly refit the incumbents that only have standard fits.
6. **Later: mode A** (sequential unsealing of the corpus), for a
   data-efficiency comparison with the bakeoff.
