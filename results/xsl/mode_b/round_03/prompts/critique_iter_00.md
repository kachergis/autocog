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

## CANDIDATE — BIC 31207.1, NLL 15567.3, wSSE 0.4200 — ACCEPTED
Fitted parameters: d=0.2516, q=0.5078, lam=0.1023, Mmax=8.295, fl=0.001485, ch_dec=1.3, lapse=0.2638

**Name:** Load-scaled interference learning with strength-dependent retrieval floor

**Description:** Learning follows the leaderboard-best mechanism (xsl_03): mutual-responsibility credit assignment (word-side times object-side), saturating gain, and retrieval-induced interference that prunes links to absent partners at a rate growing as a power law of trial load. The new claim is a retrieval-noise floor: at test, each candidate's strength is its learned association plus a constant background noise level, so weakly encoded words are chosen near randomly and strongly encoded words nearly deterministically. Accuracy is then graded in accumulated strength, i.e. in repetitions and interference.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.05, 4]
- Mmax: [0.3, 10]
- fl: [0.0, 3.0]
- ch_dec: [0.3, 6]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    d = params['d']
    q = params['q']
    lam = params['lam']
    Mmax = params['Mmax']
    fl = params['fl']
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
        M[words, :] *= (1.0 - deff)
        M[:, objs] *= (1.0 - deff)
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
| 220 | 3x4 9/12 | 0.685 | 0.440 | 395.3 | +35.7 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.639 | 810.0 | +31.9 |
| filt9E_9L | filt9E_9L | 0.601 | 0.416 | 144.3 | +25.4 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.489 | 725.0 | +22.3 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.582 | 66.2 | +17.8 |
| filt9E_6L | filt9E_6L | 0.513 | 0.405 | 88.3 | +15.4 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.328 | 417.2 | +13.1 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.323 | 427.3 | +12.6 |
| 222 | 1x3 | 0.392 | 0.489 | 481.0 | +9.2 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.326 | 429.9 | +7.3 |
| 224 | 3x3 | 0.426 | 0.483 | 459.2 | +7.2 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.328 | 436.1 | +6.9 |
| freq369_39mx | freq369_39mx | 0.622 | 0.466 | 428.9 | +5.4 |
| filt0E_9L | filt0E_9L | 0.586 | 0.642 | 69.2 | +5.0 |
| 218 | 3x4 6/8 | 0.419 | 0.320 | 272.9 | +2.9 |
| filt6E_9L | filt6E_9L | 0.590 | 0.439 | 116.1 | +2.6 |
| filt0E_6L | filt0E_6L | 0.487 | 0.598 | 67.3 | +2.2 |
| 217 | 2x4 6/12 | 0.325 | 0.316 | 310.1 | +1.9 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.693 | 55.1 | +0.9 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.582 | 49.0 | -0.1 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.666 | 52.7 | -1.0 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.329 | 455.6 | -1.5 |
| 206 | 3x3 +1w/o | 0.167 | 0.291 | 592.3 | -2.1 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.328 | 425.9 | -3.0 |
| 221 | 1x4 | 0.190 | 0.295 | 583.1 | -4.2 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.330 | 470.3 | -4.4 |
| 223 | 4x4 | 0.312 | 0.320 | 648.2 | -4.7 |
| 205 | 2x4 | 0.303 | 0.314 | 473.3 | -5.4 |
| 216 | 2x3 | 0.580 | 0.524 | 311.8 | -5.7 |
| 219 | 2x3 6/9 | 0.547 | 0.523 | 380.1 | -6.8 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.487 | 550.5 | -8.5 |
| 203 | 3x4 1/.66 | 0.211 | 0.322 | 445.5 | -12.4 |
| 201 | 3x4 | 0.191 | 0.320 | 454.4 | -13.1 |
| filt6E_6L | filt6E_6L | 0.553 | 0.424 | 79.5 | -14.3 |
| 204 | 3x4 +6o | 0.272 | 0.347 | 468.6 | -15.7 |
| 202 | 3x4 1/.5 | 0.224 | 0.338 | 427.6 | -16.1 |
| 207 | 4x4 +2w/o | 0.103 | 0.098 | 590.0 | -18.2 |
| freq369_36mx | freq369_36mx | 0.454 | 0.470 | 583.7 | -18.8 |
| filt9E_3L | filt9E_3L | 0.414 | 0.369 | 69.9 | -19.7 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.442 | 392.8 | -21.9 |
| filt3E_6L | filt3E_6L | 0.482 | 0.501 | 57.2 | -30.4 |
| 225 | 2x2 | 0.787 | 0.684 | 198.0 | -36.1 |
| filt6E_3L | filt6E_3L | 0.424 | 0.381 | 66.8 | -39.5 |
| filt0E_3L | filt0E_3L | 0.372 | 0.499 | 88.9 | -46.3 |
| filt3E_3L | filt3E_3L | 0.449 | 0.437 | 89.2 | -51.7 |
| 215 | 1x3 | 0.744 | 0.494 | 290.6 | -75.8 |
| filt3E_9L | filt3E_9L | 0.575 | 0.521 | 72.4 | -81.3 |

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
(first iteration)

## Response

{"interpretation":"The candidate has the best BIC on the leaderboard (31207 vs 31863), and the retrieval floor and sharper choice rule (ch_dec=1.3) help many conditions. But the misfits are systematic. (1) It underestimates frequency and repetition gains. 220 (3x4 at freq 9) is human 0.685 vs model 0.44. filt9E_9L, filt9E_6L and filt6E_9L are human ~0.5-0.6 vs model ~0.4-0.44, and 2x2 24-AFC is overshot in the opposite direction. 214 and 218 are also too low. (2) It overpredicts the hard high-load n-AFC conditions: 201-204, 206, 221 and 1x4 are human 0.17-0.27 vs model 0.29-0.35. (3) It underpredicts the easy 1x3 and 2x2 conditions (215: 0.744 vs 0.494; 225: 0.787 vs 0.684). Together these suggest a learning curve that is too flat in repetitions and too uniform across load. The cause is that interference prunes every link to absent partners at the same rate, including well-established pairs. Accumulated strength therefore cannot escape decay, and the high lapse of 0.26 is compensating for this. The saturating gain and the additive floor fl cannot create the steeper, convex repetition effect that the data show.","verdict":"regenerate","rationale":"Make interference strength-dependent so that consolidated pairs are protected. Replace the uniform pruning M[words,:] *= (1-deff) and M[:,objs] *= (1-deff) with an element-wise decay of (1 - deff/(1+(M/h)**p)), where h is a new protection scale in [0.1, 3] and p is fixed at about 2. Weak, ambiguous links then still get pruned and compete away, but links that have been repeatedly confirmed resist interference. This should (a) steepen the frequency effect (220, the filt*E conditions, 215/225), (b) keep high-load low-frequency conditions (201-204, 206, 221) near floor, because their links never reach the protected regime, and (c) let the lapse drop. Keep the mutual-responsibility credit with sharpness s, the saturating gain and the retrieval floor. If a parameter has to go to pay for h, fix c and g, or fix fl at a small value."}

## Usage

```json
{
  "input_tokens": 7854,
  "output_tokens": 1445,
  "reasoning_tokens": 654,
  "provider": "Claude Platform on AWS"
}
```
