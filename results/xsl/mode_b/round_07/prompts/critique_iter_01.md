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

## CANDIDATE — held-out NLL 15548.3, BIC 30734.3, wSSE (full fit) 0.3789 — REJECTED (not better than this loop's best)
Fitted parameters: d=0.2235, q=0.1495, lam=1.323, Mmax=3.674, fl=0.02457, b=0.0998, s=0.7557, r=0.899, ch_dec=1.164, p=1.047, k=0.1465, pr=0, lapse=0.1945 — at bound: pr

**Name:** Share-protected self-confirming interference learning

**Description:** Learning is competitive and interference-limited. Each pair gains the product of its word-side and object-side responsibilities, each a power-sharpened share, and the gain saturates. A pair repeated on the immediately preceding trial gets a contiguity boost. Retrieval-induced interference prunes links to absent partners at a rate that scales with word-by-object load and with the number of objects on screen. Two further mechanisms are added. First, learning is self-confirming: a pair that already wins both its word-side and object-side competition receives an extra, nonlinear increment, so words with some early confirmation take off while words in highly ambiguous designs stay near chance. Second, strong links are protected. Interference removes less of a link the larger its share of the word's (or object's) total association, so a word that has converged on a referent keeps it under later crowded trials, while diffuse, unresolved words keep being eroded. At test, objects claimed by other words are discounted (share^b), a retrieval floor is added, and choice is a power-law Luce rule.

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
- k: [0.0, 6.0]
- pr: [0.0, 1.0]

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
    k = params['k']
    pr = params['pr']
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
        sub = M[words, :]
        rs = sub / (sub.sum(1, keepdims=True) + 1e-9)
        M[words, :] = sub * (1.0 - deff * (1.0 - pr * rs))
        sub = M[:, objs]
        cs = sub / (sub.sum(0, keepdims=True) + 1e-9)
        M[:, objs] = sub * (1.0 - deff * (1.0 - pr * cs))
        st = (blk + eps) ** s
        rw = st / st.sum(1, keepdims=True)
        ro = st / st.sum(0, keepdims=True)
        rr = rw * ro
        inc = rr * (1.0 + k * rr)
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        M[ix] = blk + lam * enc * boost * inc * sat
        prev[:, :] = 0.0
        prev[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc (full fit) | held-out NLL | held-out NLL minus xsl_07's |
|---|---|---|---|---|---|
| 301 | 2x2 5x 24AFC | 0.514 | 0.613 | 821.5 | +12.9 |
| 206 | 3x3 +1w/o | 0.167 | 0.335 | 601.9 | +9.4 |
| filt0E_3L | filt0E_3L | 0.372 | 0.481 | 113.2 | +9.0 |
| 220 | 3x4 9/12 | 0.685 | 0.482 | 393.9 | +8.4 |
| freq369_39mx | freq369_39mx | 0.622 | 0.444 | 450.3 | +3.6 |
| filt0E_6L | filt0E_6L | 0.487 | 0.593 | 80.0 | +2.5 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.313 | 442.8 | +2.4 |
| 221 | 1x4 | 0.190 | 0.249 | 583.5 | +2.2 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.313 | 439.6 | +2.0 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.706 | 58.6 | +1.9 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.654 | 57.6 | +1.5 |
| filt3E_9L | filt3E_9L | 0.575 | 0.563 | 79.5 | +1.5 |
| 202 | 3x4 1/.5 | 0.224 | 0.349 | 435.1 | +1.3 |
| 225 | 2x2 | 0.787 | 0.672 | 204.7 | +1.2 |
| 203 | 3x4 1/.66 | 0.211 | 0.338 | 449.7 | +1.1 |
| 205 | 2x4 | 0.303 | 0.329 | 472.0 | +0.9 |
| 223 | 4x4 | 0.312 | 0.316 | 640.9 | +0.7 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.423 | 410.4 | +0.5 |
| filt6E_6L | filt6E_6L | 0.553 | 0.484 | 60.6 | +0.1 |
| 217 | 2x4 6/12 | 0.325 | 0.349 | 304.3 | +0.1 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.431 | 392.1 | +0.1 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.431 | 428.5 | +0.0 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.753 | 63.6 | +0.0 |
| filt0E_9L | filt0E_9L | 0.586 | 0.646 | 69.9 | -0.1 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.642 | 46.6 | -0.1 |
| 218 | 3x4 6/8 | 0.419 | 0.364 | 266.7 | -0.2 |
| 219 | 2x3 6/9 | 0.547 | 0.535 | 379.4 | -0.2 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.351 | 455.1 | -0.4 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.339 | 474.0 | -0.5 |
| filt6E_9L | filt6E_9L | 0.590 | 0.520 | 74.0 | -0.5 |
| 207 | 4x4 +2w/o | 0.103 | 0.100 | 574.1 | -0.7 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.336 | 424.2 | -0.7 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.475 | 745.4 | -0.7 |
| 201 | 3x4 | 0.191 | 0.337 | 459.8 | -0.7 |
| filt9E_9L | filt9E_9L | 0.601 | 0.501 | 80.6 | -1.0 |
| filt3E_6L | filt3E_6L | 0.482 | 0.519 | 58.4 | -1.1 |
| 204 | 3x4 +6o | 0.272 | 0.350 | 457.9 | -1.2 |
| 216 | 2x3 | 0.580 | 0.548 | 309.8 | -1.3 |
| filt9E_6L | filt9E_6L | 0.513 | 0.471 | 55.2 | -1.6 |
| freq369_36mx | freq369_36mx | 0.454 | 0.455 | 595.2 | -1.9 |
| 224 | 3x3 | 0.426 | 0.489 | 455.8 | -2.2 |
| 222 | 1x3 | 0.392 | 0.468 | 466.5 | -2.4 |
| filt9E_3L | filt9E_3L | 0.414 | 0.410 | 76.4 | -3.1 |
| filt6E_3L | filt6E_3L | 0.424 | 0.414 | 78.3 | -3.2 |
| filt3E_3L | filt3E_3L | 0.449 | 0.430 | 79.0 | -3.3 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.486 | 554.0 | -3.5 |
| 215 | 1x3 | 0.744 | 0.483 | 327.7 | -14.3 |

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

## Response

{"interpretation":"The candidate (held-out NLL 15548) is essentially tied with the leader xsl_07 (15529.5), with 2 more parameters and a worse wSSE (0.379 vs 0.439 on the full fit, but worse on held-out). The self-confirming (k=0.15) and share-protection (pr=0, at its bound) additions did not help. pr=0 means protection is switched off, and k is small, so the extra machinery is mostly inert and adds cross-validation variance. The misfits are systematic. Hard dense designs are over-predicted: 206 (0.167 vs 0.335), 201/202/203 (about 0.34 vs 0.19-0.22), 204, 224, 222. Easy or high-frequency designs are under-predicted: 215 1x3 (0.744 vs 0.483, the single biggest NLL gain available), 220 3x4 9/12 (0.685 vs 0.482), freq369_39mx and 225. The model also over-predicts 301 (24AFC, 0.514 vs 0.613 model). The pattern is a model that is too flat across conditions. It does not give enough accuracy for unambiguous, repeated, low-load designs, and it gives too much for crowded, ambiguous ones. The large lapse (0.19) and ch_dec near 1.16 compress the range. Self-confirmation is only weakly engaged because the early share is near uniform, so there is no threshold-like takeoff. The 1x3 case (215), where each trial has one word and three objects, should be solved by a cross-trial intersection, but the model learns it too slowly, probably because per-trial interference d·(no/4)^p erodes links as fast as the weak enc-scaled increments build them.","verdict":"regenerate","rationale":"Drop the inert pr protection mechanism (it is pinned at 0) and the weak k term. Instead make the learning rule explicitly hypothesis-testing, with competition computed against the current best hypothesis rather than a flat share. (1) Set the increment to inc = rw*ro * (1 + k*conf), where conf is the word's row-normalized strength on the object raised to a high power (s2 about 2-3, with a free upper bound of 6), and k has a bound up to 20. Starting from a symmetry-broken state, repeated co-occurrence in low-ambiguity designs (1x3, 2x2, 3x4 with high frequency) should then take off super-linearly, while 3x4 and 3x3+1w/o stay near chance. (2) Make interference apply only to absent partners of the words/objects present, and scale it by (1 - row_share of the link), so that links with a high share survive, using a continuous exponent instead of the binary pr. (3) Remove the enc saturation by number of objects, which appears to over-penalize 1xN designs, and fold the load effect into d alone with a single exponent, dropping p and q if they are redundant. Cut the parameter count to at most 9 for better cross-validation, and look for improvement on 215, 220, 206, 301 and freq369_39mx."}

## Usage

```json
{
  "input_tokens": 9616,
  "output_tokens": 1010,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
