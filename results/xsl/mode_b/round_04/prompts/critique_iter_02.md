# critique_iter_02

## System Prompt

You are a critical reviewer of cognitive models of cross-situational word learning. You are shown a candidate theory, its fit to the human corpus, and the leaderboard. Diagnose where and why it misfits, and decide whether the loop should stop with this candidate ('continue') or revise it ('regenerate'). Say 'continue' only if the candidate was ACCEPTED and you see no concrete, promising revision; otherwise give one specific mechanistic change to try.

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
`parameters` maps each name to a string interval "[lo, hi]" (fitted to the data) or a bare number (held fixed). Every free parameter costs log(N) BIC (see below), so add one only if it buys a real improvement in fit. Every theory automatically gets one extra fitted parameter, `lapse`: the rate of pure guessing (uniform over the choices) at test. Do not declare `lapse` yourself, and do not model random guessing inside `learn`.

## HOW THEORIES ARE SCORED
Each theory is FITTED to all 47 conditions at once (one parameter set for everyone) by differential evolution, minimising the negative log-likelihood (NLL) of every human test choice: multinomial over objects for n-AFC tests (so WHICH wrong object people choose matters, not just accuracy), binomial correct/incorrect for m-AFC tests. The score is BIC = 2*NLL + k*ln(N) with k free parameters (yours plus `lapse`) and N = 31266 choices. Lower is better. The n_subj-weighted SSE on per-word accuracy (the xsl-bakeoff objective) is shown for reference but is not the score.

## OUTPUT
Return one JSON object with these fields:
- interpretation: Where the candidate fits the human data well or badly, and which mechanism is responsible.
- verdict: 'continue' to accept the candidate as this loop's answer, or 'regenerate' to ask for a revision.
- rationale: If 'regenerate': the concrete change to the mechanism or parameters to try next. If 'continue': why it is good enough.


## User Prompt

## CANDIDATE — BIC 31077.1, NLL 15492.0, wSSE 0.4177 — ACCEPTED
Fitted parameters: d=0.2088, q=0.3556, lam=0.06911, Mmax=3.588, fl=0.0007805, b=0.1948, r=1.482, ch_dec=1.332, lapse=0.2629

**Name:** Consecutive-repetition boosted interference learning

**Description:** Learning is competitive and interference-limited: each pair gains the product of its word-side and object-side responsibilities (soft mutual exclusivity and explaining away), the gain saturates, and retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words at a rate that is a power law of trial load (words x objects). In addition, learners hold a one-trial working-memory trace of the previous trial's co-occurrences. A word-object pair that is also present on the immediately preceding trial is recognised as a repeat, so it receives amplified credit (1 + r). This explains the benefit of temporal contiguity: when consecutive trials share items, repeated pairs are boosted before interference erodes them, whereas in randomly ordered designs consecutive repeats are rare and the boost is negligible. At test, an object already claimed by other words is a poorer referent: association is multiplied by the tested word's share of that object's total association, raised to a power b, plus a retrieval floor. Choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.05, 4]
- Mmax: [0.3, 10]
- fl: [0.0, 3.0]
- b: [0.0, 3.0]
- r: [0.0, 30]
- ch_dec: [0.3, 6]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    d = params['d']
    q = params['q']
    lam = params['lam']
    Mmax = params['Mmax']
    fl = params['fl']
    b = params['b']
    r = params['r']
    s = 4.0
    c = 0.5
    g = 0.5
    eps = 0.1
    M = np.zeros((n_words, n_objects))
    prev = np.zeros((n_words, n_objects))
    for words, objs in trials:
        words = np.asarray(words)
        objs = np.asarray(objs)
        nw = len(words)
        no = len(objs)
        enc = 1.0 / (1.0 + (no / c) ** g)
        deff = min(0.95, d * (nw * no / 12.0) ** q)
        ix = np.ix_(words, objs)
        blk = M[ix].copy()
        M[words, :] *= (1.0 - deff)
        M[:, objs] *= (1.0 - deff)
        st = (blk + eps) ** s
        rw = st / st.sum(1, keepdims=True)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        bonus = 1.0 + r * prev[ix]
        M[ix] = blk + lam * enc * inc * sat * bonus
        prev = np.zeros((n_words, n_objects))
        prev[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc | NLL | NLL minus xsl_04's |
|---|---|---|---|---|---|
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.468 | 737.7 | +12.3 |
| freq369_39mx | freq369_39mx | 0.622 | 0.445 | 441.6 | +8.5 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.295 | 436.4 | +5.0 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.301 | 434.7 | +3.2 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.326 | 427.3 | +2.7 |
| filt6E_3L | filt6E_3L | 0.424 | 0.402 | 73.5 | +2.3 |
| filt9E_3L | filt9E_3L | 0.414 | 0.375 | 70.2 | +1.9 |
| 220 | 3x4 9/12 | 0.685 | 0.442 | 395.4 | +1.8 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.424 | 400.4 | +1.7 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.331 | 471.6 | +1.7 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.697 | 56.7 | +1.4 |
| 216 | 2x3 | 0.580 | 0.548 | 311.2 | +1.0 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.671 | 53.7 | +0.9 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.646 | 815.3 | +0.6 |
| 219 | 2x3 6/9 | 0.547 | 0.520 | 380.8 | +0.5 |
| filt3E_3L | filt3E_3L | 0.449 | 0.459 | 103.7 | +0.3 |
| filt9E_6L | filt9E_6L | 0.513 | 0.415 | 77.8 | +0.2 |
| filt6E_6L | filt6E_6L | 0.553 | 0.455 | 65.8 | +0.1 |
| freq369_36mx | freq369_36mx | 0.454 | 0.453 | 587.8 | -0.1 |
| filt0E_9L | filt0E_9L | 0.586 | 0.646 | 70.0 | -0.4 |
| 225 | 2x2 | 0.787 | 0.689 | 196.7 | -0.6 |
| filt6E_9L | filt6E_9L | 0.590 | 0.469 | 90.3 | -0.7 |
| filt3E_6L | filt3E_6L | 0.482 | 0.540 | 63.5 | -0.8 |
| 215 | 1x3 | 0.744 | 0.518 | 283.0 | -0.9 |
| 217 | 2x4 6/12 | 0.325 | 0.355 | 307.6 | -1.0 |
| filt9E_9L | filt9E_9L | 0.601 | 0.428 | 127.5 | -1.0 |
| 222 | 1x3 | 0.392 | 0.500 | 483.9 | -1.2 |
| 223 | 4x4 | 0.312 | 0.300 | 642.9 | -1.3 |
| 205 | 2x4 | 0.303 | 0.317 | 470.9 | -1.3 |
| 224 | 3x3 | 0.426 | 0.472 | 457.8 | -1.4 |
| filt3E_9L | filt3E_9L | 0.575 | 0.563 | 79.7 | -1.7 |
| filt0E_6L | filt0E_6L | 0.487 | 0.599 | 67.8 | -1.8 |
| 207 | 4x4 +2w/o | 0.103 | 0.098 | 580.0 | -1.9 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.349 | 453.6 | -2.1 |
| 202 | 3x4 1/.5 | 0.224 | 0.331 | 426.7 | -2.2 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.592 | 47.2 | -2.2 |
| 201 | 3x4 | 0.191 | 0.317 | 452.2 | -2.8 |
| 203 | 3x4 1/.66 | 0.211 | 0.318 | 442.1 | -3.1 |
| 204 | 3x4 +6o | 0.272 | 0.335 | 461.7 | -3.4 |
| 218 | 3x4 6/8 | 0.419 | 0.359 | 268.1 | -3.5 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.595 | 63.7 | -3.7 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.471 | 548.3 | -3.8 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.465 | 431.0 | -4.0 |
| filt0E_3L | filt0E_3L | 0.372 | 0.490 | 85.6 | -5.4 |
| 206 | 3x3 +1w/o | 0.167 | 0.294 | 585.7 | -5.4 |
| 221 | 1x4 | 0.190 | 0.294 | 577.1 | -7.1 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.465 | 385.7 | -30.5 |

## CORPUS
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

## LEADERBOARD
| rank | label | name | k | BIC | NLL | wSSE | fitted parameters |
|---|---|---|---|---|---|---|---|
| 1 | xsl_04 | Interference learning with test-time object-claim competition | 8 | 31165.9 | 15541.6 | 0.4280 | d=0.229, q=0.4687, lam=0.07155, Mmax=1.767, fl=0.0007025, b=0.1984, ch_dec=1.325, lapse=0.2755 |
| 2 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 6 | 31863.2 | 15900.6 | 0.5101 | d=0.2836, q=0.03972, lam=0.381, Mmax=9.816, ch_dec=0.6064, lapse=0.4524 |
| 3 | xsl_02 | Two-stage verify-and-consolidate with object-driven load | 6 | 32002.0 | 15969.9 | 0.4980 | a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g |
| 4 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 32045.4 | 15996.8 | 0.5529 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 5 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 32391.7 | 16175.2 | 0.5668 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 6 | baseline | Co-occurrence counting | 1 | 39486.6 | 19738.1 | 1.3660 | lapse=0.3881 |

## LOOP HISTORY
- iter 0: Capacity-gated interference learning with claim competition — BIC 32301.9 — ACCEPTED
  critic: Drop the dead and misfit components: remove the share-based claim term (b), the capacity gate (c, g) and the large floor fl, which saves about 3 parameters. Use xsl_04's structure as the base. That means a small lam, Mmax around 2, a weak test-time object-claim exponent (about 0.2), a tiny floor, and no per-trial enc factor.

Make one targeted change to pruning so repetition accumulates. Scale the pruning of links to absent partners by an inverse-strength protection factor, deff_ij = deff * 1/(1 + M_ij/K), with K as one new parameter or fixed at about 0.5. Well-consolidated pairs then survive high load, which lets accuracy rise with frequency (215, filt*E_9L, 220). Weak spurious links from ambiguous trials (206, 207, 201) are still pruned in proportion to trial load.

Keep the power-law load scaling (words x objects)^q. Let the lapse be fitted high, so it is not absorbed by the floor.
- iter 1: Strength-protected interference learning with claim competition — BIC 31176.0 — ACCEPTED
  critic: Drop K and the absolute-strength protection, which saves one parameter. Replace it with protection by RELATIVE (competitive) strength, so the dominant link of each word and object survives high-load pruning while weaker rivals are erased. Compute rel_ij = M_ij / max(max_k M_ik, max_k M_kj) (or M_ij divided by the row sum, floored at 0). Then prune with deff_ij = deff * (1 - rel_ij)**p, with p fixed at about 2 or fitted in [0,4]. The pruning would then be selective for the correct referent. Accuracy would rise with frequency in 220, 215 and the filt*_9L conditions, because the winning link is protected. Spurious links would still be wiped out in 201–204, 206 and 207, which should lower their accuracy toward human levels. Also let the Mmax saturation cap act more softly, for example by raising Mmax or using an exponent, so the correct link can grow beyond ~1.7 with repetition. To address the 301 overprediction, check that the test-time share exponent b and the floor fl do not rely on a large number of objects; letting the floor scale as fl/n_objects would be a cheap fix.

## Response

```json
{
  "interpretation": "The candidate is accepted and beats the leaderboard leader (BIC 31077 vs 31166). The gain comes almost entirely from the previous-trial repeat boost r, which fixes the temporal-contiguity condition 1_max_temp_spat_cont_orig (NLL 30 better than xsl_04). The remaining misfit is systematic and has two parts. (1) The model under-uses repetition. Conditions where humans get high accuracy from high frequency or low ambiguity are badly underpredicted: 220 (0.685 human vs 0.442 model), 215 (0.744 vs 0.518), freq369_39mx (0.622 vs 0.445), freq369-3x3hiCD (0.555 vs 0.468), the filt*_9E/9L conditions (about 0.6 vs 0.43), and 214 (0.688 vs 0.595). (2) Hard, high-ambiguity or sparse designs are overpredicted: 201–204, 206, 221, 222, 301, filt0E_3L and 1x3 (e.g. 201 0.191 vs 0.317, 221 0.190 vs 0.294, 222 0.392 vs 0.500). Accuracy therefore rises too weakly with frequency and with low load, and falls too weakly with ambiguity. The cause is that pruning is uniform across links. Interference removes the correct link's competitors only slowly, and it erodes weak, partly learned correct links in the same way as spurious ones. The saturating cap Mmax and the sharp (blk+eps)^s competition also stop a leading link from pulling away. The result is a compressed accuracy range across conditions, compensated by a high lapse (0.26) and a floor.",
  "verdict": "regenerate",
  "rationale": "Keep the structure (load-scaled interference, competitive gain, r-boost, share-based test) and change the pruning so it protects links that already dominate. Replace the uniform deff with deff_ij = deff * (1 - rel_ij)**p. Here rel_ij = M_ij / max(max_k M_ik, max_k M_kj), computed on the pre-prune matrix, and p is fitted in [0,4] or fixed near 2. Apply it to the absent-partner pruning of rows (words) and columns (objects). Links that lead their word and object are then retained, and this should let accuracy keep rising with frequency in 220, 215, freq369_39mx and the 9E/9L filtering conditions. Spurious, non-dominant links are still erased in proportion to load^q, which should pull the overpredicted 201–204, 206, 207, 221 and 222 down. To pay for the extra parameter, fix s=4 as now and consider dropping Mmax or raising its lower bound, since the protection takes over the role of capping growth. Also test dividing the floor fl by n_objects so that 301 (24 objects) and the 11-AFC conditions are not inflated."
}
```

## Usage

```json
{
  "input_tokens": 9106,
  "output_tokens": 1940,
  "reasoning_tokens": 958,
  "provider": "Claude Platform on AWS"
}
```
