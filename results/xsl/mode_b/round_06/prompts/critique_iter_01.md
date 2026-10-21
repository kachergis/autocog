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
`parameters` maps each name to a string interval "[lo, hi]" (fitted to the data) or a bare number (held fixed). Add a parameter only if it captures something that generalises across conditions (see scoring below). Every theory automatically gets one extra fitted parameter, `lapse`: the rate of pure guessing (uniform over the choices) at test. Do not declare `lapse` yourself, and do not model random guessing inside `learn`.

## HOW THEORIES ARE SCORED
Each theory is FITTED to all 47 conditions at once (one parameter set for everyone) by differential evolution, minimising the negative log-likelihood (NLL) of every human test choice: multinomial over objects for n-AFC tests (so WHICH wrong object people choose matters, not just accuracy), binomial correct/incorrect for m-AFC tests.

The SCORE is the held-out NLL from 5-fold cross-validation over conditions: the theory is fitted to four fifths of the conditions and must predict the human choices in the remaining fifth, for each fifth in turn (always the same split), and the five held-out NLLs are summed. Lower is better. Extra parameters that only fit the training conditions do not help here and usually hurt. Also shown, for reference only: BIC = 2*NLL + k*ln(N) of the full-corpus fit (k free parameters incl. `lapse`, N = 31266 choices) and the n_subj-weighted SSE on per-word accuracy (the xsl-bakeoff objective).

## OUTPUT
You have no tools and cannot run code: do not write tool calls, XML tags or `<invoke>` blocks. Reply with the JSON object only — no prose before or after it.
Return one JSON object with these fields:
- interpretation: Where the candidate fits the human data well or badly, and which mechanism is responsible.
- verdict: 'continue' to accept the candidate as this loop's answer, or 'regenerate' to ask for a revision.
- rationale: If 'regenerate': the concrete change to the mechanism or parameters to try next. If 'continue': why it is good enough.


## User Prompt

## CANDIDATE — held-out NLL 15560.8, BIC 30787.0, wSSE (full fit) 0.3881 — ACCEPTED
Fitted parameters: d=0.2214, q=0.3918, lam=0.9383, Mmax=1.113, fl=0.01533, b=0.127, s=1, ch_dec=1.149, lapse=0.1769 — at bound: s

**Name:** Sharpness-fitted interference learning with claim competition

**Description:** Learning is competitive and interference-limited. Each word splits credit over the co-present objects, and each object over the co-present words, in proportion to current strength raised to a sharpness exponent s. A pair gains the product of the two responsibilities, which gives soft mutual exclusivity and explaining away. The gain saturates at a ceiling and is weakly gated down when more objects are visible. Retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words. Its rate is a power law of trial load (words x objects). The sharpness s is fitted, not fixed. It sets how strongly an early lead is amplified, so it controls how all-or-none or graded learning is across frequency and diversity manipulations. At test, an object already claimed by other words is a poor referent. A candidate's strength is its association times the tested word's share of that object's total association, raised to b. A small retrieval floor is added, and choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.05, 4]
- Mmax: [0.3, 10]
- fl: [0.0, 3.0]
- b: [0.0, 3.0]
- s: [1.0, 6.0]
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
    s = params['s']
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
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc (full fit) | held-out NLL | held-out NLL minus xsl_04's |
|---|---|---|---|---|---|
| 215 | 1x3 | 0.744 | 0.461 | 331.6 | +28.0 |
| freq369_36mx | freq369_36mx | 0.454 | 0.470 | 597.5 | +6.9 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.504 | 563.4 | +5.6 |
| filt0E_3L | filt0E_3L | 0.372 | 0.494 | 125.1 | +5.2 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.439 | 405.0 | +5.1 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.754 | 61.0 | +5.0 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.705 | 58.6 | +4.6 |
| 225 | 2x2 | 0.787 | 0.670 | 203.8 | +4.2 |
| filt0E_6L | filt0E_6L | 0.487 | 0.600 | 84.9 | +2.8 |
| 202 | 3x4 1/.5 | 0.224 | 0.355 | 433.6 | +2.7 |
| 206 | 3x3 +1w/o | 0.167 | 0.325 | 597.7 | +2.1 |
| 201 | 3x4 | 0.191 | 0.342 | 462.8 | +1.7 |
| 203 | 3x4 1/.66 | 0.211 | 0.343 | 449.2 | +1.0 |
| 219 | 2x3 6/9 | 0.547 | 0.532 | 379.6 | +0.3 |
| 204 | 3x4 +6o | 0.272 | 0.356 | 461.4 | -0.2 |
| freq369_39mx | freq369_39mx | 0.622 | 0.461 | 437.4 | -0.3 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.338 | 473.9 | -0.6 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.335 | 457.2 | -0.9 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.325 | 436.3 | -0.9 |
| filt0E_9L | filt0E_9L | 0.586 | 0.647 | 69.6 | -1.0 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.330 | 411.2 | -1.1 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.330 | 436.9 | -1.3 |
| 207 | 4x4 +2w/o | 0.103 | 0.099 | 575.0 | -1.3 |
| 216 | 2x3 | 0.580 | 0.528 | 309.9 | -1.5 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.326 | 434.6 | -1.6 |
| 205 | 2x4 | 0.303 | 0.342 | 472.5 | -1.9 |
| 218 | 3x4 6/8 | 0.419 | 0.339 | 268.8 | -2.0 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.336 | 423.3 | -2.1 |
| 224 | 3x3 | 0.426 | 0.502 | 458.1 | -2.7 |
| 217 | 2x4 6/12 | 0.325 | 0.337 | 305.5 | -2.9 |
| filt9E_3L | filt9E_3L | 0.414 | 0.415 | 71.8 | -3.0 |
| 223 | 4x4 | 0.312 | 0.329 | 641.7 | -5.7 |
| filt3E_6L | filt3E_6L | 0.482 | 0.516 | 58.9 | -6.1 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.646 | 47.1 | -6.3 |
| 222 | 1x3 | 0.392 | 0.451 | 466.9 | -9.5 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.488 | 727.6 | -10.4 |
| filt6E_3L | filt6E_3L | 0.424 | 0.419 | 72.5 | -16.3 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.649 | 57.5 | -17.2 |
| filt9E_6L | filt9E_6L | 0.513 | 0.471 | 55.1 | -18.7 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.607 | 807.8 | -24.3 |
| filt3E_9L | filt3E_9L | 0.575 | 0.554 | 78.5 | -24.3 |
| 220 | 3x4 9/12 | 0.685 | 0.482 | 385.9 | -24.3 |
| filt6E_6L | filt6E_6L | 0.553 | 0.481 | 61.2 | -27.5 |
| filt3E_3L | filt3E_3L | 0.449 | 0.438 | 83.0 | -31.3 |
| 221 | 1x4 | 0.190 | 0.285 | 587.9 | -32.9 |
| filt9E_9L | filt9E_9L | 0.601 | 0.498 | 94.3 | -54.8 |
| filt6E_9L | filt6E_9L | 0.590 | 0.512 | 77.8 | -58.8 |

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
| rank | label | name | k | held-out NLL (score) | held-out wSSE | BIC (full fit) | fitted parameters (full fit) |
|---|---|---|---|---|---|---|---|
| 1 | xsl_04 | Interference learning with test-time object-claim competition | 8 | 15879.4 | 0.5113 | 31165.9 | d=0.229, q=0.4687, lam=0.07155, Mmax=1.767, fl=0.0007025, b=0.1984, ch_dec=1.325, lapse=0.2755 |
| 2 | xsl_05 | Consecutive-repetition boosted interference learning | 9 | 16040.9 | 0.5465 | 31077.1 | d=0.2088, q=0.3556, lam=0.06911, Mmax=3.588, fl=0.0007805, b=0.1948, r=1.482, ch_dec=1.332, lapse=0.2629 |
| 3 | xsl_06 | Consolidation-protected boosted interference learning | 10 | 16095.6 | 0.5452 | 31093.9 | d=0.2258, q=0.4937, lam=0.06846, Mmax=9.398, fl=0.0007459, b=0.1973, r=0.8932, h=28.48, ch_dec=1.331, lapse=0.2648 |
| 4 | xsl_02 | Two-stage verify-and-consolidate with object-driven load | 6 | 16104.9 | 0.5384 | 32002.0 | a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g |
| 5 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 6 | 16128.6 | 0.5626 | 31863.2 | d=0.2836, q=0.03972, lam=0.381, Mmax=9.816, ch_dec=0.6064, lapse=0.4524 |
| 6 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 16146.9 | 0.5900 | 32045.4 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 7 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 16274.8 | 0.5976 | 32391.7 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 8 | baseline | Co-occurrence counting | 1 | 19786.1 | 1.3811 | 39486.6 | lapse=0.3881 |

## LOOP HISTORY
- iter 0: Object-ambiguity-gated interference learning with claim competition — held-out NLL 16421.2 (BIC 32292.1) — ACCEPTED
  critic: Drop the (3/no)**g encoding gate, which duplicates the load term, and return to the xsl_04 backbone (load-scaled interference d*(nw*no/12)^q, saturation Mmax, test-time claim competition with b>0). Fix the failure to separate easy from hard conditions by removing the large additive floor fl (keep it near 0, as in xsl_04) and letting lapse absorb guessing. Add one mechanistic term for what is missing: a frequency-dependent strengthening of the winner. Make the responsibility sharpening exponent s a fitted parameter (about 1-6) instead of fixing s=4, so that repeated co-occurrence of a consistent pair is amplified and high-frequency or low-diversity conditions (220, freq369, filt*L) reach high accuracy. Keep b in about 0.2-1 so the model prefers unclaimed objects. Require Mmax to be an interior value, with an upper bound of about 5, not pinned at 10.

## Response

{"interpretation":"The candidate is the new leaderboard leader (held-out NLL 15560.8 vs 15879.4 for xsl_04). Its fit is still clearly imperfect. The main misfits fall into three groups.\n\n(1) Easy, low-ambiguity conditions are underfit. 215 (1x3) has human accuracy 0.744 against model 0.461, and 225 (2x2) has 0.787 against 0.670. 220 (3x4 9/12) has 0.685 against 0.482, and the high-frequency filt*_9L conditions are about 0.1 too low. The model cannot reach high accuracy on well-learnable conditions. The causes are the large retrieval floor (fl is small but lam is large), a lapse of 0.177, and saturation at Mmax≈1.1. Together these cap the strength gap between target and competitors.\n\n(2) Hard, high-load conditions are overpredicted. 201-206 and 3x4 sit at 0.19-0.27 for humans against 0.33-0.36 for the model. 4x4 conditions are overpredicted by a similar margin. 221 (1x4) has human 0.19 against model 0.285. The model therefore compresses the dynamic range: it is too high for hard conditions and too low for easy ones.\n\n(3) Frequency and diversity effects are misordered. freq369_39mx has human 0.622 against model 0.461, and freq369-3x3hiCD has 0.555 against 0.488. The sign of some contiguity effects is also wrong (1_max_temp_spat_cont human 0.489 against model 0.330, and temporal contiguity only 0.405 against 0.330). Together this shows the model has no sensitivity to temporal order or contiguity beyond load. Within-condition frequency (3/6/9 mixtures) is also not amplified enough.\n\nThe sharpness exponent s sits at its upper bound of 1, so the winner-amplification the previous critic asked for was not used. The fit ends up nearly linear in responsibilities. This indicates that the all-or-none dynamics are instead absorbed by lam, the saturation ceiling, and the share-power term b.\n\nThe per-condition NLL gaps are spread out rather than concentrated. 215 (+28) is the biggest outlier, which fits the story that the 1x3 low-ambiguity condition is badly underpredicted.","verdict":"regenerate","rationale":"Widen the model's dynamic range so that easy conditions are learned strongly and hard ones stay near chance. Three concrete changes:\n\n(a) Let s range below 1, e.g. [0.3, 6], or replace the fixed eps=0.1 with a fitted eps. Currently s is stuck at its lower bound of 1, so the bound is the constraint and the sharpening is not engaged. Alternatively, make the sharpening explicitly frequency-dependent by raising the strength of a pair to a power that grows with the pair's count of co-occurrences. That gives a rich-get-richer effect for consistently repeated pairs, which should help 215, 225, 220, freq369_39mx and filt*_9L.\n\n(b) Remove the additive retrieval floor, or cap it near 0 as in xsl_04, and let lapse absorb guessing. The floor flattens the difference between target and non-target. Raise the Mmax upper bound enough for strongly learned pairs to dominate, but keep Mmax interior.\n\n(c) Add a recency or contiguity term. For example, give each trial's increment a bonus when the same word-object pair also occurred in the previous trial, or apply interference at a reduced rate to pairs seen in the immediately preceding trial. This targets the contiguity conditions, which are currently underpredicted (0.33 against 0.40-0.49).\n\nKeep the load-scaled interference d*(nw*no/12)^q and the test-time claim competition with b."}

## Usage

```json
{
  "input_tokens": 9036,
  "output_tokens": 1307,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
