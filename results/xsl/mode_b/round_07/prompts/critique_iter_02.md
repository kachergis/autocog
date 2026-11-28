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

## CANDIDATE — held-out NLL 15496.1, BIC 30691.9, wSSE (full fit) 0.3765 — ACCEPTED
Fitted parameters: d=0.1936, q=0.05846, lam=1.136, Mmax=4.123, fl=0.01979, b=0.1702, s=0.7385, r=1.17, ch_dec=1.162, p=1.272, rho=0.3678, lapse=0.2

**Name:** Decaying-trace contiguity interference learning

**Description:** Learning is competitive and interference-limited. Credit for a pair is the product of word-side and object-side responsibilities (power-sharpened shares), and the gain saturates and is gated by the number of visible objects. Retrieval-induced interference prunes links to absent partners, with a rate set by the number of objects and by words x objects load. The new claim is that the memory of recent co-occurrence is a graded, fading trace rather than a one-trial flag. A pair seen on the previous trial is strongly marked, one seen two or three trials ago is marked more weakly, and the mark decays geometrically. Marked pairs receive amplified credit (1 + r*trace), which is how temporal contiguity helps. This lets designs with overlap every 2 or 3 trials differ from designs with immediate repeats, and from random orderings. At test, objects claimed by other words are discounted (share^b), a retrieval floor is added, and choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.05, 4]
- Mmax: [0.3, 10]
- fl: [0.0, 1.0]
- b: [0.0, 3.0]
- s: [0.3, 4.0]
- r: [0.0, 3.0]
- ch_dec: [0.3, 6]
- p: [-0.5, 1.5]
- rho: [0.0, 0.9]

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
    r = params['r']
    p = params['p']
    rho = params['rho']
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
        deff = min(0.95, d * (nw * no / 12.0) ** q * (no / 4.0) ** p)
        ix = np.ix_(words, objs)
        blk = M[ix].copy()
        boost = 1.0 + r * prev[ix]
        M[words, :] *= (1.0 - deff)
        M[:, objs] *= (1.0 - deff)
        st = (blk + eps) ** s
        rw = st / st.sum(1, keepdims=True)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        M[ix] = blk + lam * enc * boost * inc * sat
        prev *= rho
        prev[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc (full fit) | held-out NLL | held-out NLL minus xsl_07's |
|---|---|---|---|---|---|
| freq369_39mx | freq369_39mx | 0.622 | 0.436 | 453.2 | +6.4 |
| 206 | 3x3 +1w/o | 0.167 | 0.329 | 598.4 | +5.9 |
| filt3E_9L | filt3E_9L | 0.575 | 0.568 | 81.6 | +3.6 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.361 | 458.2 | +2.7 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.348 | 477.2 | +2.7 |
| 225 | 2x2 | 0.787 | 0.661 | 206.0 | +2.5 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.490 | 748.2 | +2.1 |
| freq369_36mx | freq369_36mx | 0.454 | 0.447 | 599.0 | +2.0 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.704 | 58.2 | +1.4 |
| 217 | 2x4 6/12 | 0.325 | 0.363 | 305.5 | +1.3 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.438 | 429.4 | +1.0 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.414 | 410.5 | +0.6 |
| filt6E_6L | filt6E_6L | 0.553 | 0.483 | 61.0 | +0.5 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.751 | 63.9 | +0.3 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.649 | 47.0 | +0.3 |
| 207 | 4x4 +2w/o | 0.103 | 0.104 | 575.0 | +0.2 |
| filt6E_3L | filt6E_3L | 0.424 | 0.411 | 81.6 | +0.1 |
| filt9E_6L | filt9E_6L | 0.513 | 0.468 | 56.8 | +0.1 |
| 219 | 2x3 6/9 | 0.547 | 0.522 | 379.4 | -0.3 |
| filt3E_6L | filt3E_6L | 0.482 | 0.520 | 59.1 | -0.4 |
| filt0E_9L | filt0E_9L | 0.586 | 0.644 | 69.6 | -0.4 |
| filt9E_3L | filt9E_3L | 0.414 | 0.406 | 78.9 | -0.5 |
| 204 | 3x4 +6o | 0.272 | 0.338 | 458.5 | -0.6 |
| 223 | 4x4 | 0.312 | 0.318 | 639.4 | -0.7 |
| 218 | 3x4 6/8 | 0.419 | 0.376 | 266.1 | -0.8 |
| 205 | 2x4 | 0.303 | 0.325 | 470.3 | -0.8 |
| filt0E_6L | filt0E_6L | 0.487 | 0.590 | 76.7 | -0.8 |
| 202 | 3x4 1/.5 | 0.224 | 0.342 | 432.9 | -0.9 |
| filt0E_3L | filt0E_3L | 0.372 | 0.475 | 103.3 | -0.9 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.344 | 423.9 | -1.0 |
| 203 | 3x4 1/.66 | 0.211 | 0.333 | 447.6 | -1.0 |
| filt9E_9L | filt9E_9L | 0.601 | 0.501 | 80.5 | -1.1 |
| filt3E_3L | filt3E_3L | 0.449 | 0.429 | 81.1 | -1.1 |
| filt6E_9L | filt6E_9L | 0.590 | 0.522 | 73.3 | -1.2 |
| 201 | 3x4 | 0.191 | 0.332 | 459.0 | -1.5 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.341 | 435.9 | -1.7 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.438 | 390.2 | -1.8 |
| 220 | 3x4 9/12 | 0.685 | 0.485 | 383.5 | -2.0 |
| 216 | 2x3 | 0.580 | 0.550 | 309.0 | -2.1 |
| 222 | 1x3 | 0.392 | 0.464 | 466.7 | -2.3 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.660 | 53.8 | -2.3 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.337 | 437.8 | -2.6 |
| 224 | 3x3 | 0.426 | 0.473 | 454.2 | -3.7 |
| 221 | 1x4 | 0.190 | 0.247 | 574.3 | -7.0 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.475 | 550.4 | -7.0 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.601 | 799.5 | -9.1 |
| 215 | 1x3 | 0.744 | 0.483 | 330.5 | -11.5 |

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
| 1 | xsl_07 | Soft-sharpness interference learning with contiguity boost | 10 | 15529.5 | 0.4394 | 30721.2 | d=0.2157, q=0.3407, lam=1.53, Mmax=7.773, fl=0.03068, b=0.04131, s=0.7391, r=0.9038, ch_dec=1.165, lapse=0.1924 |
| 2 | xsl_04 | Interference learning with test-time object-claim competition | 8 | 15879.4 | 0.5113 | 31165.9 | d=0.229, q=0.4687, lam=0.07155, Mmax=1.767, fl=0.0007025, b=0.1984, ch_dec=1.325, lapse=0.2755 |
| 3 | xsl_05 | Consecutive-repetition boosted interference learning | 9 | 16040.9 | 0.5465 | 31077.1 | d=0.2088, q=0.3556, lam=0.06911, Mmax=3.588, fl=0.0007805, b=0.1948, r=1.482, ch_dec=1.332, lapse=0.2629 |
| 4 | xsl_06 | Consolidation-protected boosted interference learning | 10 | 16095.6 | 0.5452 | 31093.9 | d=0.2258, q=0.4937, lam=0.06846, Mmax=9.398, fl=0.0007459, b=0.1973, r=0.8932, h=28.48, ch_dec=1.331, lapse=0.2648 |
| 5 | xsl_02 | Two-stage verify-and-consolidate with object-driven load | 6 | 16104.9 | 0.5384 | 32002.0 | a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g |
| 6 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 6 | 16128.6 | 0.5626 | 31863.2 | d=0.2836, q=0.03972, lam=0.381, Mmax=9.816, ch_dec=0.6064, lapse=0.4524 |
| 7 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 16146.9 | 0.5900 | 32045.4 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 8 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 16274.8 | 0.5976 | 32391.7 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 9 | baseline | Co-occurrence counting | 1 | 19786.1 | 1.3811 | 39486.6 | lapse=0.3881 |

## LOOP HISTORY
- iter 0: Time-decay plus object-weighted interference learning — held-out NLL 15515.3 (BIC 30725.3) — ACCEPTED
  critic: Drop the dead parameter gd (it sits at its bound of 0, so time decay does nothing) and the weakly motivated p/enc terms to cut complexity. Make learning cooperative and nonlinear in a way that separates easy from hard designs. Replace the fixed sat=1-M/Mmax gain with a self-reinforcing, hypothesis-confirming update. For each word, add an extra increment proportional to the current normalized strength of its best-supported object, raised to a power, for example inc *= (1 + k*(row_share)^s2). This gives an accelerating, threshold-like onset: words with several confirming exposures take off, while words in high-ambiguity designs such as 201-206 stay near chance. It should also raise accuracy in the high-frequency and low-ambiguity conditions (220, freq369_39mx, 225) without raising the 3x4 conditions. Keep interference only for absent partners. Add a mild word-frequency-dependent protection of strong links (decay scaled by 1 - row_share of the existing link) so that already-learned pairs are not eroded by later crowded trials. Re-fit with a lower lapse bound if possible, and judge by whether 215, 206 and 220 move toward the human values without hurting the filt* conditions.
- iter 1: Share-protected self-confirming interference learning — held-out NLL 15548.3 (BIC 30734.3) — REJECTED
  critic: Drop the inert pr protection mechanism (it is pinned at 0) and the weak k term. Instead make the learning rule explicitly hypothesis-testing, with competition computed against the current best hypothesis rather than a flat share. (1) Set the increment to inc = rw*ro * (1 + k*conf), where conf is the word's row-normalized strength on the object raised to a high power (s2 about 2-3, with a free upper bound of 6), and k has a bound up to 20. Starting from a symmetry-broken state, repeated co-occurrence in low-ambiguity designs (1x3, 2x2, 3x4 with high frequency) should then take off super-linearly, while 3x4 and 3x3+1w/o stay near chance. (2) Make interference apply only to absent partners of the words/objects present, and scale it by (1 - row_share of the link), so that links with a high share survive, using a continuous exponent instead of the binary pr. (3) Remove the enc saturation by number of objects, which appears to over-penalize 1xN designs, and fold the load effect into d alone with a single exponent, dropping p and q if they are redundant. Cut the parameter count to at most 9 for better cross-validation, and look for improvement on 215, 220, 206, 301 and freq369_39mx.

## Response

{"interpretation":"The candidate is accepted and is the best on the leaderboard (held-out NLL 15496.1 against 15529.5 for the next model). The margin is only about 33 NLL, with 11 mechanism parameters plus lapse, and lapse sits at 0.2, which looks like a bound.\n\nThe remaining misfits are systematic. (1) High-accuracy, low-ambiguity conditions are underpredicted: 215 (1x3, human 0.744 vs model 0.483), 220 (3x4 9/12, 0.685 vs 0.485), freq369_39mx (0.622 vs 0.436) and 225 (2x2, 0.787 vs 0.661). The model cannot produce enough accuracy for a word whose evidence builds slowly and consistently, and the competitive plus saturating gain is too flat. (2) Hard, high-ambiguity conditions are overpredicted: 201-203 and 206 (human 0.17-0.22 vs model about 0.33) and 202/203. The model's floor of about 0.33 is too high, which suggests the benefit of competitive credit is not suppressed enough under heavy ambiguity. (3) Contiguity-control conditions are overpredicted: 1olap3tr_contr and temp_cont_1olap2tr_contr (human 0.28-0.32 vs model about 0.35), and 2olap2tr is underpredicted. The graded trace therefore does not capture contiguity effects well.\n\nThe per-condition NLL gaps are small except for 215, 301 and 220, so the gains available are modest.","verdict":"regenerate","rationale":"Try one change: replace the saturating gain (sat = 1 - blk/Mmax) with a word-level, hypothesis-confirming nonlinear increment, and drop the weak parts to cut parameters. Concretely: (a) multiply inc by (1 + k * rowshare_prev^s2), where rowshare_prev is the pre-update normalized strength of that pair within its word's row over all objects (not only the present ones), so that consistently co-occurring pairs take off superlinearly. This should raise 215, 220, 225 and freq369_39mx, while 3x4 and 3x3+1w/o stay near chance because their row shares remain diffuse. (b) Make the interference decay depend on load only through the single quantity nw*no, dropping the separate p term and the enc saturation (c, g, eps stay fixed). (c) Keep the contiguity trace, but fix rho or tie it to r, to reduce the parameter count to about 10 including lapse. (d) Widen the ch_dec upper range and let lapse go lower if it sits at its bound, so that the floor of the hard conditions can come down. Judge the result by whether 215, 220, 206 and 201-203 move toward the human values without hurting the filt* conditions."}

## Usage

```json
{
  "input_tokens": 9895,
  "output_tokens": 948,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
