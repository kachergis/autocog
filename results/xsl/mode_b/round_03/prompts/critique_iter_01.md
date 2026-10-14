# critique_iter_01

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

## CANDIDATE — BIC 31217.5, NLL 15567.4, wSSE 0.4201 — REJECTED (not better than this loop's best)
Fitted parameters: d=0.2501, q=0.4966, lam=0.1015, Mmax=8.652, fl=0.001487, h=2.208, ch_dec=1.306, lapse=0.2634

**Name:** Strength-protected interference learning

**Description:** Learning combines mutual-responsibility credit assignment (word-side times object-side share of sharpened strength), a saturating gain, and retrieval-induced interference: hearing a word erodes its links to absent objects, and seeing an object erodes its links to absent words, at a rate that grows as a power law of trial load (words x objects). The new claim is that interference is strength-dependent: weak, ambiguous links are pruned at the full rate, whereas links that have been repeatedly confirmed become consolidated and resist interference (the decay is deff / (1 + (M/h)^2)). At test, a small retrieval-noise floor is added to every candidate's strength and choice follows a power-law Luce rule. Frequent, low-load training therefore snowballs into durable knowledge, while high-load, low-frequency designs never reach the protected regime and stay near floor.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.05, 4]
- Mmax: [0.3, 10]
- fl: [0.0, 3.0]
- h: [0.05, 3]
- ch_dec: [0.3, 6]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    d = params['d']
    q = params['q']
    lam = params['lam']
    Mmax = params['Mmax']
    fl = params['fl']
    h = params['h']
    s = 4.0
    c = 0.5
    g = 0.5
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
        R = M[words, :]
        M[words, :] = R * (1.0 - deff / (1.0 + (R / h) ** 2))
        C = M[:, objs]
        M[:, objs] = C * (1.0 - deff / (1.0 + (C / h) ** 2))
        st = (blk + eps) ** s
        rw = st / st.sum(1, keepdims=True)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        M[ix] = blk + lam * enc * inc * sat
    return M + fl + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc | NLL | NLL minus xsl_03's |
|---|---|---|---|---|---|
| 220 | 3x4 9/12 | 0.685 | 0.440 | 395.1 | +35.5 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.640 | 810.7 | +32.6 |
| filt9E_9L | filt9E_9L | 0.601 | 0.416 | 143.9 | +25.0 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.489 | 725.1 | +22.4 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.581 | 66.5 | +18.0 |
| filt9E_6L | filt9E_6L | 0.513 | 0.405 | 88.1 | +15.1 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.328 | 417.3 | +13.2 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.322 | 427.4 | +12.6 |
| 222 | 1x3 | 0.392 | 0.489 | 481.1 | +9.3 |
| 224 | 3x3 | 0.426 | 0.483 | 459.3 | +7.3 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.326 | 429.9 | +7.3 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.328 | 436.1 | +6.9 |
| freq369_39mx | freq369_39mx | 0.622 | 0.466 | 428.9 | +5.4 |
| filt0E_9L | filt0E_9L | 0.586 | 0.642 | 69.1 | +4.9 |
| 218 | 3x4 6/8 | 0.419 | 0.320 | 272.8 | +2.8 |
| filt0E_6L | filt0E_6L | 0.487 | 0.597 | 67.1 | +2.0 |
| 217 | 2x4 6/12 | 0.325 | 0.316 | 310.2 | +2.0 |
| filt6E_9L | filt6E_9L | 0.590 | 0.439 | 115.5 | +1.9 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.692 | 55.1 | +0.9 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.581 | 49.1 | +0.0 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.665 | 52.6 | -1.0 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.329 | 455.6 | -1.5 |
| 206 | 3x3 +1w/o | 0.167 | 0.291 | 592.6 | -1.8 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.327 | 426.0 | -2.9 |
| 221 | 1x4 | 0.190 | 0.294 | 583.0 | -4.3 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.330 | 470.3 | -4.4 |
| 223 | 4x4 | 0.312 | 0.320 | 648.2 | -4.8 |
| 205 | 2x4 | 0.303 | 0.314 | 473.3 | -5.4 |
| 216 | 2x3 | 0.580 | 0.524 | 311.8 | -5.8 |
| 219 | 2x3 6/9 | 0.547 | 0.524 | 380.1 | -6.8 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.487 | 550.5 | -8.5 |
| 203 | 3x4 1/.66 | 0.211 | 0.322 | 445.5 | -12.4 |
| 201 | 3x4 | 0.191 | 0.320 | 454.5 | -12.9 |
| filt6E_6L | filt6E_6L | 0.553 | 0.425 | 79.2 | -14.6 |
| 204 | 3x4 +6o | 0.272 | 0.347 | 468.6 | -15.7 |
| 202 | 3x4 1/.5 | 0.224 | 0.338 | 427.6 | -16.0 |
| 207 | 4x4 +2w/o | 0.103 | 0.099 | 590.1 | -18.1 |
| freq369_36mx | freq369_36mx | 0.454 | 0.470 | 583.6 | -18.9 |
| filt9E_3L | filt9E_3L | 0.414 | 0.368 | 70.0 | -19.6 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.442 | 392.6 | -22.0 |
| filt3E_6L | filt3E_6L | 0.482 | 0.501 | 57.2 | -30.4 |
| 225 | 2x2 | 0.787 | 0.685 | 197.9 | -36.3 |
| filt6E_3L | filt6E_3L | 0.424 | 0.381 | 67.1 | -39.2 |
| filt0E_3L | filt0E_3L | 0.372 | 0.498 | 88.4 | -46.8 |
| filt3E_3L | filt3E_3L | 0.449 | 0.437 | 89.6 | -51.3 |
| 215 | 1x3 | 0.744 | 0.494 | 290.8 | -75.7 |
| filt3E_9L | filt3E_9L | 0.575 | 0.522 | 72.4 | -81.3 |

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
| 1 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 6 | 31863.2 | 15900.6 | 0.5101 | d=0.2836, q=0.03972, lam=0.381, Mmax=9.816, ch_dec=0.6064, lapse=0.4524 |
| 2 | xsl_02 | Two-stage verify-and-consolidate with object-driven load | 6 | 32002.0 | 15969.9 | 0.4980 | a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g |
| 3 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 32045.4 | 15996.8 | 0.5529 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 4 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 32391.7 | 16175.2 | 0.5668 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 5 | baseline | Co-occurrence counting | 1 | 39486.6 | 19738.1 | 1.3660 | lapse=0.3881 |

## LOOP HISTORY
- iter 0: Load-scaled interference learning with strength-dependent retrieval floor — BIC 31207.1 — ACCEPTED
  critic: Make interference strength-dependent so that consolidated pairs are protected. Replace the uniform pruning M[words,:] *= (1-deff) and M[:,objs] *= (1-deff) with an element-wise decay of (1 - deff/(1+(M/h)**p)), where h is a new protection scale in [0.1, 3] and p is fixed at about 2. Weak, ambiguous links then still get pruned and compete away, but links that have been repeatedly confirmed resist interference. This should (a) steepen the frequency effect (220, the filt*E conditions, 215/225), (b) keep high-load low-frequency conditions (201-204, 206, 221) near floor, because their links never reach the protected regime, and (c) let the lapse drop. Keep the mutual-responsibility credit with sharpness s, the saturating gain and the retrieval floor. If a parameter has to go to pay for h, fix c and g, or fix fl at a small value.

## Response

{"interpretation":"Adding strength-protected interference (h) did not pay for itself. BIC is 31217.5 against 31207.1 for the iteration-0 parent, so the extra parameter cost more than the likelihood it bought. Frequency and structure effects are still underpredicted. 220 (3x4 9/12) has human accuracy 0.685 and model 0.44. 218 and 214 (the 6/8 subset-pool designs) are also too low. filt9E_9L, filt9E_6L and freq369-3x3hiCD are underpredicted as well, and 215 (1x3) is too low (0.49 vs 0.74). Meanwhile the model is too high on the hard, high-load, low-frequency conditions (201-204, 206, 221, 301) and on 222 and 225 relative to the humans. The model therefore compresses accuracy toward the middle. Its per-trial erosion of absent links grows with the number of trials regardless of what the learner already knows, so it cannot produce a steep rise with repetition. Protecting strong links by their own strength only partly fixes this, because the pruning of weak links is still undirected. The 215 vs 222 discrepancy (same design, 0.744 vs 0.392) is a data inconsistency that no theory can fit, so it should not drive the revision.","verdict":"regenerate","rationale":"Drop h and the (1+(M/h)^2) protection, which saves one parameter. Make interference competition-gated instead of uniform. When word w is heard, erode its links to absent objects by deff * conf_w, where conf_w = max over present objects of blk[w,:] / Mmax, clipped to [0,1] (optionally conf_w**2). Do the same on the object side, scaling the erosion of an object's links to absent words by that object's confirmation among the present words. Early, when nothing is confirmed, there is almost no forgetting, so ambiguous co-occurrence evidence accumulates. Once a word has a confirmed partner, its distractor links are pruned strongly, which gives a mutual-exclusivity snowball. This should steepen the frequency effect (220, filt*E, 215) and make small-pool designs (6/8, 9/12) benefit from the repeated structure. It should also leave high-load, low-frequency conditions near floor, since their words never become confirmed. Keep the mutual-responsibility credit with sharpness s, the saturating gain, the load-scaled enc, the retrieval floor fl and ch_dec. The parameters are d, q, lam, Mmax, fl and ch_dec, so the parameter count goes down. If fit is poor, allow a fitted exponent on conf (range [0.5, 3]) instead of h."}

## Usage

```json
{
  "input_tokens": 8376,
  "output_tokens": 2038,
  "reasoning_tokens": 1147,
  "provider": "Claude Platform on AWS"
}
```
