# critique_iter_00

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

## CANDIDATE — BIC 32301.9, NLL 16099.2, wSSE 0.4542 — ACCEPTED
Fitted parameters: d=0.2909, q=0.7481, lam=7.959, Mmax=10, fl=0.1939, b=0, c=1.199, g=2.311, ch_dec=1.075, lapse=0.1019 — at bound: Mmax, b

**Name:** Capacity-gated interference learning with claim competition

**Description:** Learners strengthen word-object pairs by the product of word-side and object-side responsibilities (soft mutual exclusivity and explaining away). Encoding is gated by a steep, fitted object-capacity function: trials showing more objects than working memory can parse are encoded sharply less, giving a near-threshold drop from 3 to 4 objects. Retrieval-induced interference prunes links to absent partners at a rate that grows as a power law of trial load (words x objects). Gain saturates as strengths approach a ceiling. At test, an object already claimed by other words is a poor referent, so a candidate's strength is its association times the tested word's share of that object's total association, raised to a power. A retrieval floor and a power-law Luce choice rule follow.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.05, 8]
- Mmax: [0.3, 10]
- fl: [0.0, 3.0]
- b: [0.0, 3.0]
- c: [1.0, 8.0]
- g: [0.3, 8.0]
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
    c = params['c']
    g = params['g']
    s = 4.0
    eps = 0.1
    M = np.zeros((n_words, n_objects))
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
        M[ix] = blk + lam * enc * inc * sat
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc | NLL | NLL minus xsl_04's |
|---|---|---|---|---|---|
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.470 | 493.2 | +94.6 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.519 | 633.9 | +81.8 |
| filt6E_9L | filt6E_9L | 0.590 | 0.419 | 166.3 | +75.3 |
| freq369_36mx | freq369_36mx | 0.454 | 0.491 | 661.5 | +73.6 |
| filt3E_9L | filt3E_9L | 0.575 | 0.417 | 144.9 | +63.5 |
| 215 | 1x3 | 0.744 | 0.421 | 329.4 | +45.4 |
| filt3E_6L | filt3E_6L | 0.482 | 0.406 | 108.3 | +44.0 |
| filt6E_6L | filt6E_6L | 0.553 | 0.410 | 105.8 | +40.2 |
| 223 | 4x4 | 0.312 | 0.372 | 678.3 | +34.1 |
| filt9E_9L | filt9E_9L | 0.601 | 0.420 | 156.4 | +27.8 |
| 220 | 3x4 9/12 | 0.685 | 0.408 | 415.8 | +22.2 |
| filt9E_6L | filt9E_6L | 0.513 | 0.411 | 97.9 | +20.3 |
| 206 | 3x3 +1w/o | 0.167 | 0.398 | 611.0 | +20.0 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.323 | 448.2 | +16.8 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.372 | 485.2 | +15.3 |
| 205 | 2x4 | 0.303 | 0.298 | 487.0 | +14.8 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.371 | 468.6 | +13.0 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.724 | 65.7 | +12.9 |
| 219 | 2x3 6/9 | 0.547 | 0.486 | 392.3 | +12.0 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.775 | 67.3 | +12.0 |
| 202 | 3x4 1/.5 | 0.224 | 0.343 | 439.5 | +10.6 |
| freq369_39mx | freq369_39mx | 0.622 | 0.488 | 443.5 | +10.4 |
| 225 | 2x2 | 0.787 | 0.650 | 205.0 | +7.6 |
| 201 | 3x4 | 0.191 | 0.336 | 462.1 | +7.2 |
| 224 | 3x3 | 0.426 | 0.510 | 465.9 | +6.7 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.335 | 438.0 | +6.4 |
| 216 | 2x3 | 0.580 | 0.475 | 314.0 | +3.8 |
| 203 | 3x4 1/.66 | 0.211 | 0.335 | 447.6 | +2.4 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.373 | 426.3 | +1.7 |
| 218 | 3x4 6/8 | 0.419 | 0.332 | 273.2 | +1.5 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.366 | 435.0 | -0.1 |
| filt9E_3L | filt9E_3L | 0.414 | 0.384 | 67.0 | -1.3 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.644 | 47.4 | -2.0 |
| 222 | 1x3 | 0.392 | 0.424 | 482.1 | -3.1 |
| 217 | 2x4 6/12 | 0.325 | 0.287 | 302.9 | -5.7 |
| filt0E_9L | filt0E_9L | 0.586 | 0.598 | 64.4 | -6.0 |
| filt6E_3L | filt6E_3L | 0.424 | 0.380 | 65.1 | -6.1 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.366 | 408.5 | -7.8 |
| filt0E_6L | filt0E_6L | 0.487 | 0.531 | 55.7 | -13.8 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.492 | 711.4 | -14.0 |
| 204 | 3x4 +6o | 0.272 | 0.321 | 448.1 | -17.0 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.684 | 46.3 | -21.1 |
| filt3E_3L | filt3E_3L | 0.449 | 0.371 | 79.5 | -23.9 |
| filt0E_3L | filt0E_3L | 0.372 | 0.405 | 66.4 | -24.6 |
| 207 | 4x4 +2w/o | 0.103 | 0.093 | 555.9 | -26.1 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.571 | 788.4 | -26.3 |
| 221 | 1x4 | 0.190 | 0.244 | 542.8 | -41.4 |

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
(first iteration)

## Response

```json
{"interpretation":"The candidate is not competitive. Its BIC of 32301.9 is about 1,100 worse than xsl_04 (31165.9) and worse than xsl_01 to xsl_03; it only beats kachergis and the baseline. Several parts of the mechanism are inert or mismatched to the data. The test-time claim-competition exponent b sits at 0, so the 'claim competition' in the name is not used. Mmax and lam are at or near their bounds. The capacity gate (c=1.2, g=2.3) gives enc of about 0.27, 0.11 and 0.05 for 2, 3 and 4 objects. That is a smooth scale-down, not a 3-to-4 threshold, and it mostly rescales lam. The fit is also flat in frequency. In the filtering conditions (2x2, 11-AFC), human accuracy rises with frequency, for example filt6E_9L at 0.59 and filt3E_9L at 0.575, but the model stays near 0.41 to 0.42 everywhere. The same flatness hurts 215 (1x3, 108 trials: human 0.74, model 0.42) and 220 (human 0.685, model 0.41). Frequency gains are being erased because load-scaled pruning hits all links in a trial, including well-learned ones, and a high floor (fl=0.19) compresses the choice probabilities. In the other direction, the model overpredicts the hardest high-ambiguity conditions: 206 (human 0.167, model 0.398), 201 and 203 (about 0.2 vs 0.34), and cont_div6-12 (0.435 vs 0.519). So it learns too little from repetition when ambiguity is low and too much when ambiguity is high. The lapse of 0.10 is much lower than the 0.28 to 0.45 that the better models use, and the large floor and gate seem to be absorbing guessing behaviour. Worst NLL gaps against xsl_04 are freq369-3x3loCD (+95), cont_div6-12 (+82), the filt*E_9L and filt*E_6L conditions (+20 to +75), and freq369_36mx (+74).","verdict":"regenerate","rationale":"Drop the dead and misfit components: remove the share-based claim term (b), the capacity gate (c, g) and the large floor fl, which saves about 3 parameters. Use xsl_04's structure as the base. That means a small lam, Mmax around 2, a weak test-time object-claim exponent (about 0.2), a tiny floor, and no per-trial enc factor.\n\nMake one targeted change to pruning so repetition accumulates. Scale the pruning of links to absent partners by an inverse-strength protection factor, deff_ij = deff * 1/(1 + M_ij/K), with K as one new parameter or fixed at about 0.5. Well-consolidated pairs then survive high load, which lets accuracy rise with frequency (215, filt*E_9L, 220). Weak spurious links from ambiguous trials (206, 207, 201) are still pruned in proportion to trial load.\n\nKeep the power-law load scaling (words x objects)^q. Let the lapse be fitted high, so it is not absorbed by the floor."}
```

## Usage

```json
{
  "input_tokens": 8144,
  "output_tokens": 2047,
  "reasoning_tokens": 972,
  "provider": "Claude Platform on AWS"
}
```
