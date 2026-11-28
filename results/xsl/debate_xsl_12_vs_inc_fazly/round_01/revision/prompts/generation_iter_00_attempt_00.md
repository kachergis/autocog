# generation_iter_00_attempt_00

## System Prompt

You are a cognitive scientist and expert Python programmer working on theories of cross-situational word learning. Your job is to propose a NEW theory, with an executable learner, that predicts the human data better (lower held-out NLL) than every theory on the leaderboard.

## THE TASK: CROSS-SITUATIONAL WORD LEARNING
On each training trial a learner hears several novel words and sees several novel objects, with no indication of which word goes with which object. Each word has a consistent referent across trials, so its meaning can only be learned by tracking word-object co-occurrence across situations. After training, learning is tested per word: the learner hears a word and chooses its referent, either among ALL objects (n-AFC) or among a few candidates (m-AFC). The corpus holds 47 experimental conditions from published adult studies, which vary the number of words and objects per trial, word frequency, contextual diversity, temporal contiguity, filtering, and test format.

## WHAT A THEORY MUST PROVIDE
A prose claim about the learning mechanism, plus one Python function:

```python
def learn(params, trials, n_words, n_objects):
    # params: dict of parameter values (floats), keyed by the names you declare
    # trials: list of (words, objects) pairs, in training order; each is a 1-D
    #         numpy int array of 0-based indices (words in 0..n_words-1,
    #         objects in 0..n_objects-1)
    # returns: numpy array, shape (n_words, n_objects), of non-negative
    #          word-object association strengths after training
```

Rules for `learn`:
  * `np` (numpy) is the only module available; do not import anything.
  * It must be deterministic (no randomness) and return finite, non-negative values.
  * It must be fast: one pass over all 47 conditions must take well under 2 seconds,
    because the fitter calls it thousands of times.
  * The correct referent of word i is NOT given to `learn`; it only sees co-occurrences.

Test choices are made by the Luce rule from your matrix: P(choose object j for word i) = m[i, j] / sum_k m[i, k] over the candidate objects. If you declare a parameter named `ch_dec`, the rule becomes m[i, j]**ch_dec / sum_k m[i, k]**ch_dec. Do not implement the choice rule yourself.

## PARAMETERS
`parameters` maps each name to a string interval "[lo, hi]" (fitted to the data) or a bare number (held fixed). Add a parameter only if it captures something that generalises across conditions (see scoring below). Do not hard-code values taken from the fitted parameters shown below: those fits used every condition, including each cross-validation fold's held-out ones, so a parameter you pin is refitted as free (and counted) whenever a leaderboard theory fits a parameter of the same name. Every theory automatically gets one extra fitted parameter, `lapse`: the rate of pure guessing (uniform over the choices) at test. Do not declare `lapse` yourself, and do not model random guessing inside `learn`.

## HOW THEORIES ARE SCORED
Each theory is FITTED to all 47 conditions at once (one parameter set for everyone) by differential evolution, minimising the negative log-likelihood (NLL) of the human test responses: each person's test responses: whether they chose the correct referent on each item. People differ in how much they learn, so every person gets an ability offset on the log-odds of your predicted accuracy, drawn from a normal distribution whose spread (`ability_sd`) is fitted automatically, like `lapse` - do not declare it. The score is about item-level ACCURACY; which wrong object people chose does not count. Model what an average learner does; individual differences are handled by `ability_sd`.

The SCORE is the held-out NLL from 5-fold cross-validation over conditions: the theory is fitted to four fifths of the conditions and must predict the human choices in the remaining fifth, for each fifth in turn (always the same split), and the five held-out NLLs are summed. Lower is better. Extra parameters that only fit the training conditions do not help here and usually hurt. Also shown, for reference only: BIC = 2*NLL + k*ln(N) of the full-corpus fit (k free parameters incl. `lapse`, N = 31266 choices) and the n_subj-weighted SSE on per-word accuracy (the xsl-bakeoff objective).

## ACCEPT GATE
You work in a loop. Each proposal is fitted and cross-validated; it is ACCEPTED only if its held-out NLL is strictly lower than that of the best accepted candidate so far in this loop. `## CURRENT BEST CANDIDATE` always shows the best accepted candidate, never a rejected one, so build on it. A critic comments on each candidate; the `## LOOP HISTORY` block shows which of its suggestions helped (ACCEPTED) and which did not (REJECTED).

Leaderboard theories are for inspiration only: do not resubmit one of them or a cosmetic variant. A new mechanism that fits better is the goal.

## OUTPUT
You have no tools and cannot run code: do not write tool calls, XML tags or `<invoke>` blocks. Reply with the JSON object only — no prose before or after it.
Return one JSON object with these fields:
- name: Short name for the theory (a few words).
- theory: The theoretical claim this theory makes about how people learn words across situations.
- learn: Python source defining def learn(params, trials, n_words, n_objects) -> np.ndarray, returning the final (n_words, n_objects) association matrix. `np` is available.
- parameters: Free parameters as intervals, e.g. {"X": "[0.001, 5]"}; a bare number pins a value.
- rationale: Why this theory should fit human cross-situational learning better than the alternatives.


## User Prompt

## GOAL
The debate arbiter asks you to revise xsl_13 (new_model). Keep its central claim; change its mechanism or implementation.
Arbiter's interpretation: The round splits. xsl_13 wins on 216/219 (2x3 designs, 18-AFC): 0.553 vs human 0.563, while inc_fazly sits 2.5 SE low. inc_fazly wins on the high-exposure filtered 2x2 conditions and the 4x4 mixed-frequency conditions: 0.491 vs human 0.494, while xsl_13 sits 3.4 SE low at 0.424. Earlier rounds show the same pattern. xsl_12 lost on filt*_9L by 4.4 SE, and the lineage's one clear win was on the sparse 221/222 conditions. The recurring xsl failure is under-learning where repetition of consistent pairs should keep paying off: filt6E/9E_6L, filt*_9L (0.47-0.50 vs human 0.51-0.60), 220 (0.48 vs 0.69), and 225 (0.62 vs 0.79). In these designs filler trials and extra exposure do not raise accuracy at all, and xsl's accuracy stays flat or falls as exposure grows. That points to the mechanisms that cap or erode learning with exposure: negative kappa (diminishing returns for uncommitted words), per-trial interference erosion (deff), and the commitment threshold. In 2x2 designs a word is often not yet committed (th=0.76), so it is stuck in the diminishing-returns regime and filler erosion eats its gains. inc_fazly, in contrast, accumulates without decay and so misses the cases where people use exclusion or mutual exclusivity: 215 (0.38 vs 0.74), 225, 220, 212, 2x3. It also over-predicts the hard, high-ambiguity conditions: 201-207 are all 0.1-0.15 too high. Held-out NLL favors xsl_13 (16887 vs 17154) and the per-condition NLL differences are mostly negative, so xsl_13 is the better base. Revise theory 1, keeping its core of competition, exclusion and a contiguity trace, and fix the exposure-dependence of its learning rate and erosion.
Recommendation: Keep xsl_13's central claim and replace the commitment-binary exposure and interference machinery with a cumulative-evidence gate. (1) Remove the negative-kappa diminishing returns, which are a main cause of the flat or declining accuracy with exposure in filt*_6L/9L and the 4x4 mixed-frequency conditions. Set the exposure exponent to a smooth non-negative function of the word's evidence concentration, for example e = kpos*conc, where conc = max_j rs_j (or 1 - normalized entropy of the word's row), instead of the sigmoid commit at th=0.76. Even partially-concentrated words should then get increasing returns. (2) Make interference erosion shielding graded by the same concentration: M[w,o] erodes by deff*(1 - rs_wo/mx)^k, not only for committed words. Also make deff for absent-partner erosion scale down for pairs with a long trace, so fillers cannot wash out consistent pairs in 2x2 designs. (3) Keep the exclusion term (xi) and the share^b test discount, since these produce the wins on 216/219/215/225, but apply the discount only after a word's own max row share exceeds a small threshold, so it does not suppress weak words in the high-ambiguity conditions 201-207, where xsl_13 is still too high. Refit d, kpos, and th (or drop th) after these changes, and check that filt6E_6L, filt9E_9L, 2_x8_39_4x4 and 3_x8_369_4x4 rise to about 0.5 without lowering 216/219.
The revision must reach a lower held-out NLL than the CURRENT BEST CANDIDATE, which is xsl_13 itself.

## CORPUS (47 conditions)
| label | condition | words x objects | trials | words x objects per trial | mean word freq | test | n_subj | human acc |
|---|---|---|---|---|---|---|---|---|
| 201 | 3x4 | 18x18 | 36 | 3.0x4.0 | 6.0 | n-AFC | 25 | 0.191 |
| 202 | 3x4 1/.5 | 18x18 | 36 | 3.0x4.0 | 6.0 | n-AFC | 25 | 0.224 |
| 203 | 3x4 1/.66 | 18x18 | 36 | 3.0x4.0 | 6.0 | n-AFC | 25 | 0.211 |
| 204 | 3x4 +6o | 18x24 | 36 | 3.0x4.0 | 6.0 | n-AFC | 20 | 0.272 |
| 205 | 2x4 | 18x18 | 54 | 2.0x4.0 | 6.0 | n-AFC | 33 | 0.303 |
| 206 | 3x3 +1w/o | 18x18 | 54 | 3.0x3.0 | 9.0 | n-AFC | 39 | 0.167 |
| 207 | 4x4 +2w/o | 18x18 | 54 | 4.0x4.0 | 12.0 | n-AFC | 39 | 0.103 |
| 211 | 1x3 6/18 4AFC | 18x18 | 108 | 1.0x3.0 | 6.0 | 4-AFC | 43 | 0.674 |
| 212 | 2x3 6/9 4AFC | 18x18 | 54 | 2.0x3.0 | 6.0 | 4-AFC | 38 | 0.686 |
| 213 | 2x4 6/12 4AFC | 18x18 | 54 | 2.0x4.0 | 6.0 | 4-AFC | 31 | 0.624 |
| 214 | 3x4 6/8 4AFC | 18x18 | 36 | 3.0x4.0 | 6.0 | 4-AFC | 36 | 0.688 |
| 215 | 1x3 | 18x18 | 108 | 1.0x3.0 | 6.0 | n-AFC | 23 | 0.744 |
| 216 | 2x3 | 18x18 | 54 | 2.0x3.0 | 6.0 | n-AFC | 23 | 0.580 |
| 217 | 2x4 6/12 | 18x18 | 54 | 2.0x4.0 | 6.0 | n-AFC | 14 | 0.325 |
| 218 | 3x4 6/8 | 18x18 | 36 | 3.0x4.0 | 6.0 | n-AFC | 13 | 0.419 |
| 219 | 2x3 6/9 | 18x18 | 54 | 2.0x3.0 | 6.0 | n-AFC | 32 | 0.547 |
| 220 | 3x4 9/12 | 18x18 | 54 | 3.0x4.0 | 9.0 | n-AFC | 33 | 0.685 |
| 221 | 1x4 | 18x18 | 108 | 1.0x4.0 | 6.0 | n-AFC | 40 | 0.190 |
| 222 | 1x3 | 18x18 | 108 | 1.0x3.0 | 6.0 | n-AFC | 40 | 0.392 |
| 223 | 4x4 | 18x18 | 27 | 4.0x4.0 | 6.0 | n-AFC | 77 | 0.312 |
| 224 | 3x3 | 18x18 | 36 | 3.0x3.0 | 6.0 | n-AFC | 36 | 0.426 |
| 225 | 2x2 | 18x18 | 54 | 2.0x2.0 | 6.0 | n-AFC | 19 | 0.787 |
| 301 | 2x2 5x 24AFC | 24x24 | 60 | 2.0x2.0 | 5.0 | n-AFC | 46 | 0.514 |
| filt0E_3L | filt0E_3L | 12x12 | 18 | 2.0x2.0 | 3.0 | 11-AFC | 31 | 0.372 |
| filt3E_3L | filt3E_3L | 12x12 | 27 | 2.0x2.0 | 4.5 | 11-AFC | 30 | 0.449 |
| filt6E_3L | filt6E_3L | 12x12 | 36 | 2.0x2.0 | 6.0 | 11-AFC | 30 | 0.424 |
| filt9E_3L | filt9E_3L | 12x12 | 45 | 2.0x2.0 | 7.5 | 11-AFC | 31 | 0.414 |
| filt0E_6L | filt0E_6L | 12x12 | 36 | 2.0x2.0 | 6.0 | 11-AFC | 23 | 0.487 |
| filt3E_6L | filt3E_6L | 12x12 | 45 | 2.0x2.0 | 7.5 | 11-AFC | 23 | 0.482 |
| filt6E_6L | filt6E_6L | 12x12 | 54 | 2.0x2.0 | 9.0 | 11-AFC | 23 | 0.553 |
| filt9E_6L | filt9E_6L | 12x12 | 63 | 2.0x2.0 | 10.5 | 11-AFC | 23 | 0.513 |
| filt0E_9L | filt0E_9L | 12x12 | 54 | 2.0x2.0 | 9.0 | 11-AFC | 31 | 0.586 |
| filt3E_9L | filt3E_9L | 12x12 | 63 | 2.0x2.0 | 10.5 | 11-AFC | 31 | 0.575 |
| filt6E_9L | filt6E_9L | 12x12 | 72 | 2.0x2.0 | 12.0 | 11-AFC | 31 | 0.590 |
| filt9E_9L | filt9E_9L | 12x12 | 81 | 2.0x2.0 | 13.5 | 11-AFC | 31 | 0.601 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 18x18 | 27 | 4.0x4.0 | 6.0 | n-AFC | 30 | 0.411 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 18x18 | 27 | 4.0x4.0 | 6.0 | n-AFC | 30 | 0.391 |
| freq369-3x3loCD | freq369-3x3loCD | 18x18 | 36 | 3.0x3.0 | 6.0 | n-AFC | 34 | 0.485 |
| freq369-3x3hiCD | freq369-3x3hiCD | 18x18 | 36 | 3.0x3.0 | 6.0 | n-AFC | 104 | 0.555 |
| freq369_36mx | freq369_36mx | 18x18 | 36 | 3.0x3.0 | 6.0 | n-AFC | 66 | 0.454 |
| freq369_39mx | freq369_39mx | 18x18 | 36 | 3.0x3.0 | 6.0 | n-AFC | 40 | 0.622 |
| cont_div6-12 | cont_div6-12 | 18x18 | 36 | 3.0x3.0 | 6.0 | n-AFC | 62 | 0.435 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 18x18 | 27 | 4.0x4.0 | 6.0 | n-AFC | 30 | 0.489 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 18x18 | 27 | 4.0x4.0 | 6.0 | n-AFC | 32 | 0.405 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 18x18 | 27 | 4.0x4.0 | 6.0 | n-AFC | 31 | 0.317 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 6.0 | n-AFC | 31 | 0.280 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 6.0 | n-AFC | 31 | 0.414 |

## LEADERBOARD (ranked by held-out NLL; lower is better)
| rank | label | name | k | held-out NLL (score) | held-out wSSE | BIC (full fit) | fitted parameters (full fit) |
|---|---|---|---|---|---|---|---|
| 1 | xsl_13 | Commitment-protected, consistency-scaled contiguity exclusion | 18 | 16886.6 | 0.4110 | 33886.7 | d=0.1577, lam=6, Mmax=9.58, fl=0.008242, b=0.9266, s=0.2652, r=2.504, ch_dec=0.5007, p=0.408, rho=0.2276, th=0.7594, kc=17.61, xi=1, kappa=-0.6, kpos=0.03843, pg=0, lapse=0.001, ability_sd=1.441 — at bound: lam, xi, kappa, pg, lapse |
| 2 | xsl_12 | Contiguity-boosted commitment-gated exclusion with exposure scaling | 16 | 16905.5 | 0.4364 | 33874.5 | d=0.1809, lam=6, Mmax=9.874, fl=0.01817, b=0.6339, s=0.2016, r=2.493, ch_dec=0.5767, p=0.2859, rho=0.1662, th=0.7441, kc=17.49, xi=1, kappa=-0.4109, lapse=0.001, ability_sd=1.439 — at bound: lam, xi, lapse |
| 3 | xsl_11 | Commitment-gated exclusion with count-based residual familiarity | 16 | 16913.6 | 0.4306 | 33887.1 | d=0.1912, lam=6, Mmax=30, fl=0.005666, b=1.075, s=0.1386, r=2.801, ch_dec=0.4371, p=0.565, rho=0.1436, th=0.7535, kc=17.76, xi=1, g2=0.01529, lapse=0.001, ability_sd=1.434 — at bound: lam, Mmax, xi, lapse |
| 4 | xsl_09 | Commitment-gated claim exclusion with redistributed credit | 15 | 16921.7 | 0.4333 | 33881.7 | d=0.1542, lam=6, Mmax=30, fl=0.008398, b=0.8015, s=0.1686, r=2.472, ch_dec=0.4477, p=1.32, rho=0.1521, th=0.7918, kc=25.05, xi=1, lapse=0.001, ability_sd=1.437 — at bound: lam, Mmax, xi, lapse |
| 5 | xsl_10 | Streamlined commitment-gated claim exclusion | 15 | 16922.0 | 0.4331 | 33881.7 | d=0.1548, lam=6, Mmax=30, fl=0.00844, b=0.8011, s=0.1683, r=2.472, ch_dec=0.448, p=1.321, rho=0.1498, th=0.7916, kc=25, xi=1, lapse=0.001, ability_sd=1.437 — at bound: lam, Mmax, xi, lapse |
| 6 | xsl_04 | Interference learning with test-time object-claim competition | 9 | 16991.3 | 0.4936 | 33988.3 | d=0.2364, q=0.2247, lam=0.2273, Mmax=0.3, fl=0.0001701, b=0.2557, ch_dec=0.4969, lapse=0.001, ability_sd=1.465 — at bound: Mmax, lapse |
| 7 | xsl_05 | Consecutive-repetition boosted interference learning | 10 | 16995.9 | 0.4968 | 34028.2 | d=0.3121, q=0.2477, lam=0.2913, Mmax=2.195, fl=0.0001764, b=0.02524, r=0.0693, ch_dec=0.4433, lapse=0.001, ability_sd=1.481 — at bound: lapse |
| 8 | xsl_06 | Consolidation-protected boosted interference learning | 11 | 16998.6 | 0.4933 | 34045.2 | d=0.3338, q=0.2506, lam=0.3096, Mmax=3.991, fl=0.0001509, b=4.32e-05, r=1.746e-05, h=16.68, ch_dec=0.4271, lapse=0.001, ability_sd=1.475 — at bound: lapse |
| 9 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 7 | 17050.6 | 0.5456 | 34097.1 | d=0.4057, q=0, lam=0.4589, Mmax=2.244, ch_dec=0.3, lapse=0.001, ability_sd=1.492 — at bound: q, ch_dec, lapse |
| 10 | xsl_08 | Decaying-trace contiguity interference learning | 13 | 17061.6 | 0.5091 | 34027.3 | d=0.2425, q=0, lam=0.6531, Mmax=3.301, fl=0.0002465, b=0.5619, s=1.572, r=1.168, ch_dec=0.4308, p=1.438, rho=0.04663, lapse=0.001, ability_sd=1.464 — at bound: q, lapse |
| 11 | xsl_07 | Soft-sharpness interference learning with contiguity boost | 11 | 17065.7 | 0.5168 | 33986.3 | d=0.1009, q=0.4014, lam=0.8151, Mmax=0.5607, fl=0.0001104, b=3, s=1.098, r=0.4271, ch_dec=0.4643, lapse=0.001, ability_sd=1.461 — at bound: b, lapse |
| 12 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 5 | 17135.1 | 0.6886 | 34253.6 | X=0.0999, B=5.305, C=0.996, lapse=0.001, ability_sd=1.559 — at bound: lapse |
| 13 | inc_fazly | Probabilistic incremental alignment (Fazly, Alishahi & Stevenson 2010) | 4 | 17153.6 | 0.5039 | 34301.1 | lambda=0.1469, beta=21.84, lapse=0.001, ability_sd=1.446 — at bound: lapse |
| 14 | inc_uncfam_attention | Uncertainty/familiarity associative learning with trial-level attention (uncfam_attention) | 5 | 17170.3 | 0.6964 | 34252.1 | X=0.0972, B=5.146, C=0.9953, lapse=0.001, ability_sd=1.562 — at bound: lapse |
| 15 | inc_uncfam_gamma | Uncertainty/familiarity associative learning with a familiarity exponent (uncfam_gamma) | 6 | 17177.2 | 0.7072 | 34212.0 | X=0.08456, B=2.391, C=0.9969, gamma=1.437, lapse=0.001, ability_sd=1.595 — at bound: lapse |
| 16 | inc_tilles | Confidence-flux associative learning (Tilles & Fontanari 2012) | 5 | 17357.2 | 0.7003 | 34678.9 | x=0.1546, b=0.2465, alpha_0=0.9716, lapse=0.001, ability_sd=1.548 — at bound: lapse |
| 17 | inc_bayesian_decay | Bayesian learner with decay (Bayesian_decay) | 5 | 17657.3 | 0.8615 | 35258.0 | alpha=1.297, delta=1.756, ch_dec=9.991, lapse=0.001, ability_sd=1.552 — at bound: ch_dec, lapse |
| 18 | baseline | Co-occurrence counting | 2 | 19568.7 | 1.3742 | 38891.5 | lapse=0.3495, ability_sd=2.874 |

## LEADERBOARD THEORIES
### xsl_13: Commitment-protected, consistency-scaled contiguity exclusion
This keeps xsl_12's account: competitive, load-dependent, interference-limited learning, a fading contiguity trace, and commitment-gated exclusion with redistributed credit. It changes how repetition and interference treat strong associations. (1) Encoding efficiency depends on whether a word's evidence is consistent. A word that has not committed gets diminishing returns on exposure (negative kappa), so ambiguous, frequent words do not snowball. A word whose top object dominates its association (commitment) gets increasing returns, with exponent kpos on exposure, so repetition pays off in easy, consistent designs. (2) Retrieval-induced interference on absent partners is gated by commitment. The links of a committed word-object pair are shielded from erosion in proportion to how dominant that link is within the word's association. Fillers, or other words and objects on the trial, therefore cannot erode a well-supported pair. (3) Exclusion still only withdraws claimed objects from other words, never from the committed word's own credit. At test, objects claimed by other words are discounted by share^b, and choice is a power-law Luce rule.

### xsl_12: Contiguity-boosted commitment-gated exclusion with exposure scaling
Learning is competitive and interference-limited. Each pair gains the product of word-side and object-side responsibilities. Retrieval-induced interference erodes links to absent partners, at a rate set by trial load and by the number of objects. A fading trace of recent co-occurrence strongly amplifies repeated pairs, so temporal contiguity and immediate repetition have a large effect. A word commits once one object dominates its association, and that object is then withdrawn from other co-present words, which redistributes the freed credit. Encoding efficiency changes with a word's accumulated exposure, with diminishing returns when kappa is negative, so frequent words do not snowball to certainty. At test, objects claimed by other words are discounted by share^b, and choice is a power-law Luce rule. There is no passive decay, because it was fitted to zero.

### xsl_11: Commitment-gated exclusion with count-based residual familiarity
Learning follows the commitment-gated claim-exclusion account. Credit is the product of word-side and object-side responsibilities. Retrieval-induced interference erodes links to absent partners, and a fading trace amplifies repeated pairs. A word commits once one object dominates its association, and a committed object is withdrawn from competing co-present words, with the freed credit redistributed. The new claim is that people also keep a weak, interference-free familiarity trace of raw word-object co-occurrence counts. This trace is not subject to competition or forgetting. At test it adds to the competitive association, so words that never committed still show graded, frequency-driven partial knowledge. Words that did commit are dominated by their competitive mapping.

### xsl_09: Commitment-gated claim exclusion with redistributed credit
Learning is competitive and interference-limited. A fading trace of recent co-occurrence amplifies credit for repeated pairs. Each word forms a graded working hypothesis once its strongest object holds a dominant share of the word's association. A committed pair is strengthened further, and its object is withdrawn as a candidate for other co-present words. The new claim is that withdrawn credit is redistributed. A word that loses an object to another word's hypothesis hands that responsibility to its remaining co-present objects, as in genuine explaining-away, rather than just losing credit. Sparse or frequent designs therefore snowball into committed, exclusive mappings. Dense, ambiguous designs stay uncommitted and keep diffuse credit.

### xsl_10: Streamlined commitment-gated claim exclusion
Learning is competitive and interference-limited. Each word accumulates credit over co-present objects as the product of word-side and object-side responsibilities. Retrieval-induced interference erodes links to absent partners, and a fading trace of recent co-occurrence amplifies repeated pairs. A word becomes committed once its strongest object holds a dominant share of the word's association. A committed pair gets a strong boost, and its object is withdrawn from competing co-present words. The credit this frees is redistributed to the other co-present objects (explaining-away). Sparse or frequent designs snowball into exclusive mappings, while dense, ambiguous designs stay diffuse. At test, an object claimed by other words is discounted by share^b, and choice is a power-law Luce rule.

### xsl_04: Interference learning with test-time object-claim competition
Learning follows the accepted account. Credit for each pair is the product of the word-side and object-side responsibilities, gain saturates, and retrieval-induced interference prunes links to absent partners at a rate that grows as a power law of trial load. At test, people also use mutual exclusivity. An object already strongly claimed by other words is a poor referent for the word being tested. The effective strength of a candidate is therefore its learned association times the share of that object's total association that belongs to the tested word, raised to a fitted power b. A retrieval-noise floor is then added. This changes which wrong objects are chosen: wrong choices are drawn to objects that no word has claimed, and are less likely to be objects that other words own.

### xsl_05: Consecutive-repetition boosted interference learning
Learning is competitive and interference-limited: each pair gains the product of its word-side and object-side responsibilities (soft mutual exclusivity and explaining away), the gain saturates, and retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words at a rate that is a power law of trial load (words x objects). In addition, learners hold a one-trial working-memory trace of the previous trial's co-occurrences. A word-object pair that is also present on the immediately preceding trial is recognised as a repeat, so it receives amplified credit (1 + r). This explains the benefit of temporal contiguity: when consecutive trials share items, repeated pairs are boosted before interference erodes them, whereas in randomly ordered designs consecutive repeats are rare and the boost is negligible. At test, an object already claimed by other words is a poorer referent: association is multiplied by the tested word's share of that object's total association, raised to a power b, plus a retrieval floor. Choice is a power-law Luce rule.

### xsl_06: Consolidation-protected boosted interference learning
Learning is competitive and interference-limited. Each pair gains the product of its word-side and object-side responsibilities (soft mutual exclusivity and explaining away), and the gain saturates. Retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words, at a rate that is a power law of trial load. A one-trial working-memory trace boosts pairs that repeat on consecutive trials (temporal contiguity). The new mechanism is consolidation: interference is not equally damaging to every link. Weak, tentative associations are fully exposed to it, while associations that have already been built up become progressively protected from it. Protection falls as 1/(1+(M/h)^2), so strength acts as a soft consolidation threshold. Words seen often enough to cross the threshold become durable and snowball, as with 9-repetition words and sparse 1-word trials. Words that never cross it keep being eroded under dense load. At test, an object claimed by other words is a poorer referent, with a share^b competition term plus a retrieval floor, and choice is a power-law Luce rule.

### xsl_03: Load-scaled absent-alternative interference with saturating competitive encoding
Each word splits credit over the co-present objects and each object over the co-present words (sharpened strength plus a prior). A pair gains the product of its two responsibilities, which gives soft mutual exclusivity and explaining away. The gain saturates as a pair's strength approaches a ceiling, and it is mildly gated down when many objects are present. Forgetting is retrieval-induced interference. When a word is heard, its links to absent objects decay, and when an object is seen, its links to absent words decay. Co-present pairs are protected from this decay. The decay rate grows as a power law in trial load, the number of words times the number of objects. Sparse trials (1x3, 2x2) therefore barely erode earlier learning, so accuracy is high and snowballs with repetition, while dense trials (3x4, 4x4) prune strongly. Three, six and nine repetitions give graded rather than all-or-none learning. Test choice is a power-law Luce rule.

### xsl_08: Decaying-trace contiguity interference learning
Learning is competitive and interference-limited. Credit for a pair is the product of word-side and object-side responsibilities (power-sharpened shares), and the gain saturates and is gated by the number of visible objects. Retrieval-induced interference prunes links to absent partners, with a rate set by the number of objects and by words x objects load. The new claim is that the memory of recent co-occurrence is a graded, fading trace rather than a one-trial flag. A pair seen on the previous trial is strongly marked, one seen two or three trials ago is marked more weakly, and the mark decays geometrically. Marked pairs receive amplified credit (1 + r*trace), which is how temporal contiguity helps. This lets designs with overlap every 2 or 3 trials differ from designs with immediate repeats, and from random orderings. At test, objects claimed by other words are discounted (share^b), a retrieval floor is added, and choice is a power-law Luce rule.

### xsl_07: Soft-sharpness interference learning with contiguity boost
Learning is competitive and interference-limited. Each word splits credit over the co-present objects, and each object over the co-present words, in proportion to current strength raised to a fitted exponent s. The exponent is allowed to fall below 1, so competition can be soft as well as winner-take-all. A pair gains the product of the two responsibilities, which gives soft mutual exclusivity and explaining away. The gain saturates and is weakly gated down as more objects are visible. Retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words, at a rate that is a power law of trial load (words x objects). A one-trial memory trace marks pairs that co-occurred on the immediately preceding trial, and such repeated pairs receive amplified credit (1+r). This is how temporal contiguity helps. At test, an object already claimed by other words is a poorer referent: association is multiplied by the tested word's share of that object's total association, raised to b. A small retrieval floor is added, and choice is a power-law Luce rule.

### kachergis: Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012)
Learners distribute a limited amount of attention over the word-object pairs on each trial. Attention is drawn both to familiar pairs (already strongly associated) and to uncertain stimuli (words or objects whose associations are still spread over many partners, measured by entropy). The attended share of a fixed associative weight is added to each pair, and all associations decay slightly on every trial. Familiarity produces a rich-get-richer bias; uncertainty pulls attention toward stimuli that have no strong partner yet, which together implement a soft mutual-exclusivity bias.

### inc_fazly: Probabilistic incremental alignment (Fazly, Alishahi & Stevenson 2010)
On each trial every word aligns to each co-present object in proportion to its current probability of meaning that object (smoothed for never-seen pairs); alignments accumulate as associations, and each word's meaning probabilities are the smoothed, normalised associations.

### inc_uncfam_attention: Uncertainty/familiarity associative learning with trial-level attention (uncfam_attention)
uncfam (Kachergis, Yu & Shiffrin 2012) in which the amount of associative weight distributed on a trial scales with how uncertain that trial's objects are relative to the mean uncertainty of all objects seen so far: learners are more alert on trials with more uncertain items.

### inc_uncfam_gamma: Uncertainty/familiarity associative learning with a familiarity exponent (uncfam_gamma)
Kachergis, Yu & Shiffrin (2012) uncfam with familiarity raised to a free power gamma: attention to a pair is assoc^gamma * exp(B * H(word)) * exp(B * H(object)). gamma = 1 is uncfam; gamma < 1 dampens the rich-get-richer bias.

### inc_tilles: Confidence-flux associative learning (Tilles & Fontanari 2012)
Each word holds a probability distribution over objects. On every trial, confidence flows into the word's links to the objects present from its links to previously seen but absent objects (in proportion to how much confidence sits on the absent ones), while words not on the trial shift confidence the other way. Each word's update is scaled by its own certainty (one minus the normalised entropy of its distribution); uncertain words drift toward uniform. Novel words start with a bias b toward the novel objects they appear with.

### inc_bayesian_decay: Bayesian learner with decay (Bayesian_decay)
An approximately Bayesian learner of P(word | object). Each trial multiplies the hypothesis matrix by a likelihood that keeps pairs consistent with the trial (both present or both absent) and down-weights inconsistent ones by alpha, with co-present pairs boosted by delta; columns are renormalised. Choice is a power-Luce rule with exponent ch_dec.

### baseline: Co-occurrence counting
Learners simply count how often each word and object occur together across trials, with no attention, competition or forgetting.

## BEST LEADERBOARD THEORY: xsl_13
**Name:** Commitment-protected, consistency-scaled contiguity exclusion

**Description:** This keeps xsl_12's account: competitive, load-dependent, interference-limited learning, a fading contiguity trace, and commitment-gated exclusion with redistributed credit. It changes how repetition and interference treat strong associations. (1) Encoding efficiency depends on whether a word's evidence is consistent. A word that has not committed gets diminishing returns on exposure (negative kappa), so ambiguous, frequent words do not snowball. A word whose top object dominates its association (commitment) gets increasing returns, with exponent kpos on exposure, so repetition pays off in easy, consistent designs. (2) Retrieval-induced interference on absent partners is gated by commitment. The links of a committed word-object pair are shielded from erosion in proportion to how dominant that link is within the word's association. Fillers, or other words and objects on the trial, therefore cannot erode a well-supported pair. (3) Exclusion still only withdraws claimed objects from other words, never from the committed word's own credit. At test, objects claimed by other words are discounted by share^b, and choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- lam: [0.05, 6]
- Mmax: [0.3, 30]
- fl: [0.0, 0.5]
- b: [0.0, 3.0]
- s: [0.01, 4]
- r: [0.0, 12.0]
- ch_dec: [0.3, 3]
- p: [-0.5, 3.0]
- rho: [0.0, 0.9]
- th: [0.2, 0.9]
- kc: [0, 40]
- xi: [0.0, 1.0]
- kappa: [-0.6, 1.0]
- kpos: [0.0, 0.6]
- pg: [0.0, 1.0]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    d = params['d']
    lam = params['lam']
    Mmax = params['Mmax']
    fl = params['fl']
    b = params['b']
    s = params['s']
    r = params['r']
    p = params['p']
    rho = params['rho']
    th = params['th']
    kc = params['kc']
    xi = params['xi']
    kappa = params['kappa']
    kpos = params['kpos']
    pg = params['pg']
    q = 0.07
    c = 0.5
    g = 0.5
    eps = 0.1
    steep = 8.0
    M = np.zeros((n_words, n_objects))
    prev = np.zeros((n_words, n_objects))
    cnt = np.zeros(n_words)
    for words, objs in trials:
        words = np.asarray(words)
        objs = np.asarray(objs)
        nw = len(words)
        no = len(objs)
        enc = 1.0 / (1.0 + (no / c) ** g)
        deff = min(0.95, d * (nw * no / 12.0) ** q * (no / 4.0) ** p)
        ix = np.ix_(words, objs)
        rsA = M / (M.sum(1, keepdims=True) + 0.3)
        mxA = rsA.max(1)
        zA = np.clip(steep * (mxA - th), -30.0, 30.0)
        cA = 1.0 / (1.0 + np.exp(-zA))
        PA = pg * cA[:, None] * (rsA / (mxA[:, None] + 1e-9)) ** 4
        rs = rsA[words, :]
        mx = mxA[words]
        commit = cA[words]
        rsb = rs[:, objs]
        ratio = (rsb / (mx[:, None] + 1e-9)) ** 4
        blk = M[ix].copy()
        boost = 1.0 + r * prev[ix] + kc * commit[:, None] * ratio
        contrib = commit[:, None] * rsb
        tot = contrib.sum(0, keepdims=True)
        cl = np.clip(tot - contrib, 0.0, 1.0)
        M[words, :] *= (1.0 - deff * (1.0 - PA[words, :]))
        M[:, objs] *= (1.0 - deff * (1.0 - PA[:, objs]))
        st = (blk + eps) ** s
        stw = st * (1.0 - xi * cl)
        rw = stw / (stw.sum(1, keepdims=True) + 1e-12)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        e = kappa * (1.0 - commit) + kpos * commit
        expo = ((1.0 + cnt[words]) ** e)[:, None]
        M[ix] = blk + lam * enc * expo * boost * inc * sat
        cnt[words] += 1.0
        prev *= rho
        prev[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```

Per-condition fit, worst first:
| label | condition | human acc | model acc (full fit) | held-out NLL | held-out NLL per choice |
|---|---|---|---|---|---|
| filt0E_3L | filt0E_3L | 0.372 | 0.505 | 486.3 | 0.654 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.734 | 349.0 | 0.625 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.302 | 343.9 | 0.616 |
| 218 | 3x4 6/8 | 0.419 | 0.334 | 144.1 | 0.616 |
| freq369_36mx | freq369_36mx | 0.454 | 0.460 | 729.0 | 0.614 |
| filt0E_9L | filt0E_9L | 0.586 | 0.608 | 455.4 | 0.612 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.302 | 330.0 | 0.611 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.607 | 669.9 | 0.607 |
| 205 | 2x4 | 0.303 | 0.300 | 359.2 | 0.605 |
| 219 | 2x3 6/9 | 0.547 | 0.565 | 345.4 | 0.600 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.885 | 410.0 | 0.599 |
| filt0E_6L | filt0E_6L | 0.487 | 0.576 | 329.5 | 0.597 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.306 | 333.1 | 0.597 |
| 217 | 2x4 6/12 | 0.325 | 0.338 | 149.1 | 0.592 |
| freq369_39mx | freq369_39mx | 0.622 | 0.445 | 421.6 | 0.586 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.288 | 314.9 | 0.583 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.852 | 447.8 | 0.579 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.481 | 639.5 | 0.573 |
| 222 | 1x3 | 0.392 | 0.487 | 410.9 | 0.571 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.318 | 313.7 | 0.562 |
| filt3E_6L | filt3E_6L | 0.482 | 0.480 | 310.3 | 0.562 |
| 220 | 3x4 9/12 | 0.685 | 0.471 | 331.6 | 0.558 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.414 | 339.8 | 0.555 |
| 223 | 4x4 | 0.312 | 0.280 | 763.5 | 0.551 |
| 224 | 3x3 | 0.426 | 0.495 | 355.9 | 0.549 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.735 | 349.1 | 0.539 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.521 | 989.0 | 0.528 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.384 | 302.4 | 0.525 |
| 204 | 3x4 +6o | 0.272 | 0.277 | 188.5 | 0.524 |
| filt9E_6L | filt9E_6L | 0.513 | 0.470 | 288.4 | 0.522 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.384 | 279.2 | 0.517 |
| filt3E_9L | filt3E_9L | 0.575 | 0.498 | 379.5 | 0.510 |
| filt3E_3L | filt3E_3L | 0.449 | 0.462 | 362.2 | 0.503 |
| 203 | 3x4 1/.66 | 0.211 | 0.296 | 225.7 | 0.502 |
| 221 | 1x4 | 0.190 | 0.244 | 350.3 | 0.487 |
| 201 | 3x4 | 0.191 | 0.293 | 211.7 | 0.471 |
| filt6E_3L | filt6E_3L | 0.424 | 0.456 | 332.5 | 0.462 |
| filt6E_9L | filt6E_9L | 0.590 | 0.487 | 343.2 | 0.461 |
| 216 | 2x3 | 0.580 | 0.581 | 190.4 | 0.460 |
| 206 | 3x3 +1w/o | 0.167 | 0.242 | 321.4 | 0.458 |
| 202 | 3x4 1/.5 | 0.224 | 0.306 | 205.4 | 0.456 |
| filt6E_6L | filt6E_6L | 0.553 | 0.471 | 250.6 | 0.454 |
| filt9E_3L | filt9E_3L | 0.414 | 0.456 | 336.8 | 0.453 |
| filt9E_9L | filt9E_9L | 0.601 | 0.485 | 326.6 | 0.439 |
| 215 | 1x3 | 0.744 | 0.515 | 178.6 | 0.431 |
| 225 | 2x2 | 0.787 | 0.667 | 145.4 | 0.425 |
| 207 | 4x4 +2w/o | 0.103 | 0.089 | 246.6 | 0.351 |

## CURRENT BEST CANDIDATE (this loop) — held-out NLL 16886.6, BIC 33886.7, params d=0.1577, lam=6, Mmax=9.58, fl=0.008242, b=0.9266, s=0.2652, r=2.504, ch_dec=0.5007, p=0.408, rho=0.2276, th=0.7594, kc=17.61, xi=1, kappa=-0.6, kpos=0.03843, pg=0, lapse=0.001, ability_sd=1.441 — at bound: lam, xi, kappa, pg, lapse
**Name:** Commitment-protected, consistency-scaled contiguity exclusion

**Description:** This keeps xsl_12's account: competitive, load-dependent, interference-limited learning, a fading contiguity trace, and commitment-gated exclusion with redistributed credit. It changes how repetition and interference treat strong associations. (1) Encoding efficiency depends on whether a word's evidence is consistent. A word that has not committed gets diminishing returns on exposure (negative kappa), so ambiguous, frequent words do not snowball. A word whose top object dominates its association (commitment) gets increasing returns, with exponent kpos on exposure, so repetition pays off in easy, consistent designs. (2) Retrieval-induced interference on absent partners is gated by commitment. The links of a committed word-object pair are shielded from erosion in proportion to how dominant that link is within the word's association. Fillers, or other words and objects on the trial, therefore cannot erode a well-supported pair. (3) Exclusion still only withdraws claimed objects from other words, never from the committed word's own credit. At test, objects claimed by other words are discounted by share^b, and choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- lam: [0.05, 6]
- Mmax: [0.3, 30]
- fl: [0.0, 0.5]
- b: [0.0, 3.0]
- s: [0.01, 4]
- r: [0.0, 12.0]
- ch_dec: [0.3, 3]
- p: [-0.5, 3.0]
- rho: [0.0, 0.9]
- th: [0.2, 0.9]
- kc: [0, 40]
- xi: [0.0, 1.0]
- kappa: [-0.6, 1.0]
- kpos: [0.0, 0.6]
- pg: [0.0, 1.0]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    d = params['d']
    lam = params['lam']
    Mmax = params['Mmax']
    fl = params['fl']
    b = params['b']
    s = params['s']
    r = params['r']
    p = params['p']
    rho = params['rho']
    th = params['th']
    kc = params['kc']
    xi = params['xi']
    kappa = params['kappa']
    kpos = params['kpos']
    pg = params['pg']
    q = 0.07
    c = 0.5
    g = 0.5
    eps = 0.1
    steep = 8.0
    M = np.zeros((n_words, n_objects))
    prev = np.zeros((n_words, n_objects))
    cnt = np.zeros(n_words)
    for words, objs in trials:
        words = np.asarray(words)
        objs = np.asarray(objs)
        nw = len(words)
        no = len(objs)
        enc = 1.0 / (1.0 + (no / c) ** g)
        deff = min(0.95, d * (nw * no / 12.0) ** q * (no / 4.0) ** p)
        ix = np.ix_(words, objs)
        rsA = M / (M.sum(1, keepdims=True) + 0.3)
        mxA = rsA.max(1)
        zA = np.clip(steep * (mxA - th), -30.0, 30.0)
        cA = 1.0 / (1.0 + np.exp(-zA))
        PA = pg * cA[:, None] * (rsA / (mxA[:, None] + 1e-9)) ** 4
        rs = rsA[words, :]
        mx = mxA[words]
        commit = cA[words]
        rsb = rs[:, objs]
        ratio = (rsb / (mx[:, None] + 1e-9)) ** 4
        blk = M[ix].copy()
        boost = 1.0 + r * prev[ix] + kc * commit[:, None] * ratio
        contrib = commit[:, None] * rsb
        tot = contrib.sum(0, keepdims=True)
        cl = np.clip(tot - contrib, 0.0, 1.0)
        M[words, :] *= (1.0 - deff * (1.0 - PA[words, :]))
        M[:, objs] *= (1.0 - deff * (1.0 - PA[:, objs]))
        st = (blk + eps) ** s
        stw = st * (1.0 - xi * cl)
        rw = stw / (stw.sum(1, keepdims=True) + 1e-12)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        e = kappa * (1.0 - commit) + kpos * commit
        expo = ((1.0 + cnt[words]) ** e)[:, None]
        M[ix] = blk + lam * enc * expo * boost * inc * sat
        cnt[words] += 1.0
        prev *= rho
        prev[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```

Per-condition fit, worst first:
| label | condition | human acc | model acc (full fit) | held-out NLL | held-out NLL minus xsl_13's |
|---|---|---|---|---|---|
| 201 | 3x4 | 0.191 | 0.293 | 211.7 | +0.0 |
| 202 | 3x4 1/.5 | 0.224 | 0.306 | 205.4 | +0.0 |
| 203 | 3x4 1/.66 | 0.211 | 0.296 | 225.7 | +0.0 |
| 204 | 3x4 +6o | 0.272 | 0.277 | 188.5 | +0.0 |
| 205 | 2x4 | 0.303 | 0.300 | 359.2 | +0.0 |
| 206 | 3x3 +1w/o | 0.167 | 0.242 | 321.4 | +0.0 |
| 207 | 4x4 +2w/o | 0.103 | 0.089 | 246.6 | +0.0 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.852 | 447.8 | +0.0 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.885 | 410.0 | +0.0 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.734 | 349.0 | +0.0 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.735 | 349.1 | +0.0 |
| 215 | 1x3 | 0.744 | 0.515 | 178.6 | +0.0 |
| 216 | 2x3 | 0.580 | 0.581 | 190.4 | +0.0 |
| 217 | 2x4 6/12 | 0.325 | 0.338 | 149.1 | +0.0 |
| 218 | 3x4 6/8 | 0.419 | 0.334 | 144.1 | +0.0 |
| 219 | 2x3 6/9 | 0.547 | 0.565 | 345.4 | +0.0 |
| 220 | 3x4 9/12 | 0.685 | 0.471 | 331.6 | +0.0 |
| 221 | 1x4 | 0.190 | 0.244 | 350.3 | +0.0 |
| 222 | 1x3 | 0.392 | 0.487 | 410.9 | +0.0 |
| 223 | 4x4 | 0.312 | 0.280 | 763.5 | +0.0 |
| 224 | 3x3 | 0.426 | 0.495 | 355.9 | +0.0 |
| 225 | 2x2 | 0.787 | 0.667 | 145.4 | +0.0 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.607 | 669.9 | +0.0 |
| filt0E_3L | filt0E_3L | 0.372 | 0.505 | 486.3 | +0.0 |
| filt3E_3L | filt3E_3L | 0.449 | 0.462 | 362.2 | +0.0 |
| filt6E_3L | filt6E_3L | 0.424 | 0.456 | 332.5 | +0.0 |
| filt9E_3L | filt9E_3L | 0.414 | 0.456 | 336.8 | +0.0 |
| filt0E_6L | filt0E_6L | 0.487 | 0.576 | 329.5 | +0.0 |
| filt3E_6L | filt3E_6L | 0.482 | 0.480 | 310.3 | +0.0 |
| filt6E_6L | filt6E_6L | 0.553 | 0.471 | 250.6 | +0.0 |
| filt9E_6L | filt9E_6L | 0.513 | 0.470 | 288.4 | +0.0 |
| filt0E_9L | filt0E_9L | 0.586 | 0.608 | 455.4 | +0.0 |
| filt3E_9L | filt3E_9L | 0.575 | 0.498 | 379.5 | +0.0 |
| filt6E_9L | filt6E_9L | 0.590 | 0.487 | 343.2 | +0.0 |
| filt9E_9L | filt9E_9L | 0.601 | 0.485 | 326.6 | +0.0 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.302 | 330.0 | +0.0 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.288 | 314.9 | +0.0 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.414 | 339.8 | +0.0 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.521 | 989.0 | +0.0 |
| freq369_36mx | freq369_36mx | 0.454 | 0.460 | 729.0 | +0.0 |
| freq369_39mx | freq369_39mx | 0.622 | 0.445 | 421.6 | +0.0 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.481 | 639.5 | +0.0 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.384 | 279.2 | +0.0 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.384 | 302.4 | +0.0 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.318 | 313.7 | +0.0 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.306 | 333.1 | +0.0 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.302 | 343.9 | +0.0 |

## LOOP HISTORY
(first iteration)

## Response

{"name":"Evidence-conflict-gated diminishing returns exclusion","theory":"Keeps xsl_13's core of competitive, load-dependent interference learning, a fading contiguity trace, and commitment-gated exclusion with redistributed credit. It changes how exposure affects encoding. Diminishing returns are driven by accumulated unresolved evidence, not by raw word frequency. Each time a word is heard while its association is still spread over several objects, its 'unresolved exposure' grows and encoding efficiency drops as a power of that count. Once a word's top object dominates (commitment), unresolved exposure stops accruing, so consistent repetition keeps paying off with mild increasing returns. Words in consistent, sparse designs (2x2, filtered, 4x4 mixed-frequency) therefore do not stall with more exposure. Words in ambiguous designs still do not snowball. Exclusion of claimed objects and the share^b test-time discount are kept.","learn":"def learn(params, trials, n_words, n_objects):\n    d = params['d']\n    lam = params['lam']\n    Mmax = params['Mmax']\n    fl = params['fl']\n    b = params['b']\n    s = params['s']\n    r = params['r']\n    p = params['p']\n    rho = params['rho']\n    th = params['th']\n    kc = params['kc']\n    xi = params['xi']\n    kappa = params['kappa']\n    kpos = params['kpos']\n    q = 0.07\n    c = 0.5\n    g = 0.5\n    eps = 0.1\n    steep = 8.0\n    M = np.zeros((n_words, n_objects))\n    prev = np.zeros((n_words, n_objects))\n    cnt = np.zeros(n_words)\n    unr = np.zeros(n_words)\n    for words, objs in trials:\n        words = np.asarray(words)\n        objs = np.asarray(objs)\n        nw = len(words)\n        no = len(objs)\n        enc = 1.0 / (1.0 + (no / c) ** g)\n        deff = min(0.95, d * (nw * no / 12.0) ** q * (no / 4.0) ** p)\n        ix = np.ix_(words, objs)\n        rsA = M / (M.sum(1, keepdims=True) + 0.3)\n        mxA = rsA.max(1)\n        zA = np.clip(steep * (mxA - th), -30.0, 30.0)\n        cA = 1.0 / (1.0 + np.exp(-zA))\n        rs = rsA[words, :]\n        mx = mxA[words]\n        commit = cA[words]\n        rsb = rs[:, objs]\n        ratio = (rsb / (mx[:, None] + 1e-9)) ** 4\n        blk = M[ix].copy()\n        boost = 1.0 + r * prev[ix] + kc * commit[:, None] * ratio\n        contrib = commit[:, None] * rsb\n        tot = contrib.sum(0, keepdims=True)\n        cl = np.clip(tot - contrib, 0.0, 1.0)\n        M[words, :] *= (1.0 - deff)\n        M[:, objs] *= (1.0 - deff)\n        st = (blk + eps) ** s\n        stw = st * (1.0 - xi * cl)\n        rw = stw / (stw.sum(1, keepdims=True) + 1e-12)\n        ro = st / st.sum(0, keepdims=True)\n        inc = rw * ro\n        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)\n        expo = ((1.0 + unr[words]) ** kappa) * ((1.0 + cnt[words]) ** (kpos * commit))\n        M[ix] = blk + lam * enc * expo[:, None] * boost * inc * sat\n        cnt[words] += 1.0\n        unr[words] += 1.0 - commit\n        prev *= rho\n        prev[ix] = 1.0\n    col = M.sum(0, keepdims=True)\n    share = (M + 1e-9) / (col + 0.05)\n    out = M * share ** b\n    return out + fl + 1e-6\n","parameters":{"d":"[0.0, 1.0]","lam":"[0.05, 12]","Mmax":"[0.3, 30]","fl":"[0.0, 0.5]","b":"[0.0, 3.0]","s":"[0.01, 4]","r":"[0.0, 12.0]","ch_dec":"[0.3, 3]","p":"[-0.5, 3.0]","rho":"[0.0, 0.9]","th":"[0.2, 0.9]","kc":"[0, 40]","xi":1,"kappa":"[-1.2, 0.5]","kpos":"[0.0, 0.6]"},"rationale":"The recurring xsl_13 failure is flat or declining accuracy with exposure in sparse, consistent designs (filt*_9L, 220, 225), caused by diminishing returns tied to raw word frequency (kappa pinned at its bound). Tying the penalty to unresolved exposure (trials on which the word was not yet committed) keeps the useful anti-snowball effect in ambiguous designs but lets words that have found their referent keep gaining from repetition. The model also drops the unused graded-protection term (pg fitted to 0) and pins the exclusion strength xi at its bound of 1, which removes two free parameters and should help held-out generalisation."}

## Usage

```json
{
  "input_tokens": 21504,
  "output_tokens": 3355,
  "reasoning_tokens": 1472,
  "provider": "Claude Platform on AWS"
}
```
