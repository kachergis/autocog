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

## CANDIDATE — held-out NLL 15458.5, BIC 30624.3, wSSE (full fit) 0.3613 — ACCEPTED
Fitted parameters: d=0.2066, lam=3.871, Mmax=10, fl=0.08363, b=0.04547, s=0.3446, r=1.649, ch_dec=1.129, p=1.489, rho=0.3227, th=0.643, kc=3, xi=0.6479, lapse=0.2242 — at bound: Mmax, kc

**Name:** Commitment-gated claim exclusion on decaying-trace interference learning

**Description:** Learning is competitive and interference-limited, with a fading trace of recent co-occurrence that amplifies credit for recently repeated pairs. The new claim is word-level hypothesis commitment. When a word's strongest object holds a dominant share of that word's total association (a sigmoid of the max share), the learner treats the word as having a working hypothesis. The committed pair is strengthened further, and the committed object is partly withdrawn as a competitor for other co-present words, which gives soft mutual exclusivity. Commitment is graded and appears only after repeated consistent evidence, so ambiguous dense designs (3x4, 4x4) stay uncommitted while sparse or frequent designs snowball. At test, objects claimed by other words are discounted, a retrieval floor is added, and choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- lam: [0.05, 4]
- Mmax: [0.3, 10]
- fl: [0.0, 1.0]
- b: [0.0, 3.0]
- s: [0.3, 4.0]
- r: [0.0, 3.0]
- ch_dec: [0.3, 6]
- p: [-0.5, 1.5]
- rho: [0.0, 0.9]
- th: [0.2, 0.9]
- kc: [0.0, 3.0]
- xi: [0.0, 1.0]

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
    q = 0.07
    c = 0.5
    g = 0.5
    eps = 0.1
    steep = 8.0
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
        rows = M[words, :]
        rs = rows / (rows.sum(1, keepdims=True) + 0.3)
        mx = rs.max(1)
        z = np.clip(steep * (mx - th), -30.0, 30.0)
        commit = 1.0 / (1.0 + np.exp(-z))
        rsb = rs[:, objs]
        ratio = (rsb / (mx[:, None] + 1e-9)) ** 4
        blk = M[ix].copy()
        boost = 1.0 + r * prev[ix] + kc * commit[:, None] * ratio
        contrib = commit[:, None] * rsb
        tot = contrib.sum(0, keepdims=True)
        cl = np.clip(tot - contrib, 0.0, 1.0)
        M[words, :] *= (1.0 - deff)
        M[:, objs] *= (1.0 - deff)
        st = (blk + eps) ** s
        rw = st / st.sum(1, keepdims=True)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro * (1.0 - xi * cl)
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
| label | condition | human acc | model acc (full fit) | held-out NLL | held-out NLL minus xsl_08's |
|---|---|---|---|---|---|
| 301 | 2x2 5x 24AFC | 0.514 | 0.632 | 829.5 | +30.0 |
| filt0E_3L | filt0E_3L | 0.372 | 0.472 | 115.7 | +12.5 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.487 | 758.0 | +9.8 |
| freq369_39mx | freq369_39mx | 0.622 | 0.426 | 462.9 | +9.6 |
| filt0E_6L | filt0E_6L | 0.487 | 0.591 | 82.6 | +6.0 |
| filt6E_9L | filt6E_9L | 0.590 | 0.521 | 77.4 | +4.1 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.336 | 441.8 | +4.0 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.478 | 553.1 | +2.7 |
| freq369_36mx | freq369_36mx | 0.454 | 0.440 | 601.3 | +2.3 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.404 | 412.8 | +2.2 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.332 | 437.8 | +1.9 |
| 205 | 2x4 | 0.303 | 0.315 | 471.8 | +1.5 |
| 202 | 3x4 1/.5 | 0.224 | 0.340 | 434.2 | +1.3 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.344 | 424.7 | +0.8 |
| 224 | 3x3 | 0.426 | 0.482 | 454.8 | +0.6 |
| filt9E_9L | filt9E_9L | 0.601 | 0.516 | 80.9 | +0.5 |
| 223 | 4x4 | 0.312 | 0.314 | 639.7 | +0.3 |
| filt0E_9L | filt0E_9L | 0.586 | 0.642 | 69.8 | +0.1 |
| 219 | 2x3 6/9 | 0.547 | 0.534 | 379.5 | +0.1 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.440 | 429.4 | -0.0 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.751 | 63.7 | -0.2 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.440 | 390.0 | -0.2 |
| 218 | 3x4 6/8 | 0.419 | 0.373 | 265.9 | -0.2 |
| 217 | 2x4 6/12 | 0.325 | 0.355 | 305.4 | -0.2 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.647 | 46.5 | -0.5 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.706 | 57.5 | -0.6 |
| filt3E_6L | filt3E_6L | 0.482 | 0.510 | 58.5 | -0.6 |
| 203 | 3x4 1/.66 | 0.211 | 0.327 | 446.8 | -0.8 |
| 204 | 3x4 +6o | 0.272 | 0.331 | 457.5 | -0.9 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.661 | 52.7 | -1.0 |
| 225 | 2x2 | 0.787 | 0.675 | 204.9 | -1.0 |
| 222 | 1x3 | 0.392 | 0.462 | 465.6 | -1.1 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.349 | 476.2 | -1.1 |
| 207 | 4x4 +2w/o | 0.103 | 0.099 | 573.9 | -1.1 |
| 201 | 3x4 | 0.191 | 0.324 | 457.7 | -1.4 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.362 | 456.9 | -1.4 |
| 216 | 2x3 | 0.580 | 0.555 | 307.5 | -1.6 |
| filt6E_6L | filt6E_6L | 0.553 | 0.489 | 59.2 | -1.8 |
| filt9E_6L | filt9E_6L | 0.513 | 0.485 | 52.2 | -4.6 |
| filt9E_3L | filt9E_3L | 0.414 | 0.424 | 73.7 | -5.2 |
| filt3E_3L | filt3E_3L | 0.449 | 0.433 | 74.2 | -6.9 |
| filt6E_3L | filt6E_3L | 0.424 | 0.425 | 73.6 | -8.0 |
| filt3E_9L | filt3E_9L | 0.575 | 0.544 | 72.1 | -9.6 |
| 206 | 3x3 +1w/o | 0.167 | 0.296 | 580.8 | -17.6 |
| 221 | 1x4 | 0.190 | 0.232 | 555.6 | -18.6 |
| 215 | 1x3 | 0.744 | 0.490 | 309.7 | -20.8 |
| 220 | 3x4 9/12 | 0.685 | 0.519 | 362.6 | -20.8 |

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
| 1 | xsl_08 | Decaying-trace contiguity interference learning | 12 | 15496.1 | 0.4334 | 30691.9 | d=0.1936, q=0.05846, lam=1.136, Mmax=4.123, fl=0.01979, b=0.1702, s=0.7385, r=1.17, ch_dec=1.162, p=1.272, rho=0.3678, lapse=0.2 |
| 2 | xsl_07 | Soft-sharpness interference learning with contiguity boost | 10 | 15529.5 | 0.4394 | 30721.2 | d=0.2157, q=0.3407, lam=1.53, Mmax=7.773, fl=0.03068, b=0.04131, s=0.7391, r=0.9038, ch_dec=1.165, lapse=0.1924 |
| 3 | xsl_04 | Interference learning with test-time object-claim competition | 8 | 15879.4 | 0.5113 | 31165.9 | d=0.229, q=0.4687, lam=0.07155, Mmax=1.767, fl=0.0007025, b=0.1984, ch_dec=1.325, lapse=0.2755 |
| 4 | xsl_05 | Consecutive-repetition boosted interference learning | 9 | 16040.9 | 0.5465 | 31077.1 | d=0.2088, q=0.3556, lam=0.06911, Mmax=3.588, fl=0.0007805, b=0.1948, r=1.482, ch_dec=1.332, lapse=0.2629 |
| 5 | xsl_06 | Consolidation-protected boosted interference learning | 10 | 16095.6 | 0.5452 | 31093.9 | d=0.2258, q=0.4937, lam=0.06846, Mmax=9.398, fl=0.0007459, b=0.1973, r=0.8932, h=28.48, ch_dec=1.331, lapse=0.2648 |
| 6 | xsl_02 | Two-stage verify-and-consolidate with object-driven load | 6 | 16104.9 | 0.5384 | 32002.0 | a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g |
| 7 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 6 | 16128.6 | 0.5626 | 31863.2 | d=0.2836, q=0.03972, lam=0.381, Mmax=9.816, ch_dec=0.6064, lapse=0.4524 |
| 8 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 16146.9 | 0.5900 | 32045.4 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 9 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 16274.8 | 0.5976 | 32391.7 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 10 | baseline | Co-occurrence counting | 1 | 19786.1 | 1.3811 | 39486.6 | lapse=0.3881 |

## LOOP HISTORY
- iter 0: Decaying-trace contiguity interference with passive time decay — held-out NLL 15505.2 (BIC 30705.8) — ACCEPTED
  critic: Drop the dead passive-decay term (gd) and replace it with a mechanism that targets the real residuals. Add a word-level, trial-level confidence or hypothesis-commitment step: once a word's strongest object has a sufficiently dominant share (a sigmoid or threshold on the max-share, one steepness parameter and one threshold), the model gives that pair an extra boost and withdraws the object from competing for other words. Drive the interference rate by the learned strength of the competing pair rather than by trial load alone. This should help high-frequency or low-ambiguity conditions (220, 2x2, freq369_39mx, filt*_9L) without raising accuracy in the heavily ambiguous 3x4 and 4x4 conditions, where nothing has yet been committed. To keep the parameter count down, remove Mmax and gd, fix eps, c and g, and share the exponent between p and q if possible. The aim is at most about 10 free parameters and a held-out NLL below xsl_08's 15496.

## Response

{"interpretation":"The candidate edges out the leaderboard leader on held-out NLL (15458.5 vs 15496.1), a gain of only about 38 NLL. Its wSSE is better (0.361 vs 0.433), so the commitment mechanism helps a little. But it uses 14 parameters (13 fitted plus lapse) against 12 for xsl_08, and two sit at bounds: Mmax=10 (saturation is effectively off) and kc=3. Several constants (q, c, g, eps, steep) are hard-coded. Parameter count and bound-hitting suggest the gain is marginal and partly overfit.\n\nResiduals show the intended mechanism is not delivering. The commitment machinery should lift high-frequency or unambiguous conditions, yet those remain the largest underpredictions. 215 (1x3, human .744, model .490), 220 (3x4 9/12, .685 vs .519), freq369_39mx (.622 vs .426), freq369-3x3hiCD (.555 vs .487) and 225 (2x2, .787 vs .675) are all still under-learned. So commitment snowballing is too weak or too slow in these designs.\n\nConversely, the dense ambiguous conditions are overpredicted: 201 (.191 vs .324), 203, 206 (.167 vs .296), 202 and 301 (2x2 24AFC, .514 vs .632, the single worst condition). The model keeps a floor of about .3 in conditions where people are near chance. Large spill-over to wrong objects in the dense designs is likely a cause. Both patterns point to a model that is too flat across conditions, with too little difference between learnable and unlearnable designs.\n\nLikely causes:\n- The additive floor fl plus the lapse (.22) compress the dynamic range.\n- The saturation term is inert (Mmax at bound).\n- Commitment is gated by a max-share on a 0.3-regularised row. That share is small in dense designs and grows too slowly in 1x3, which has no competition.\n- The 11-AFC filt0E_3L and 0E_6L conditions are overpredicted. Early-ambiguity effects are not captured.","verdict":"regenerate","rationale":"Keep the commitment idea, but make it sharper and cheaper.\n\n1. Remove Mmax and the saturation term, since Mmax sits at its bound and is inert.\n2. Remove fl, or fold it into lapse.\n3. Tie p and q into a single load exponent.\n4. Fix kc and rho at sensible values.\n\nThen change how commitment is computed. Currently the max share is taken over the entire row of M, with a +0.3 regulariser. Instead, define commitment from evidence margin: the ratio of the top object's association to the second-highest, or an entropy-based sharpness of the word's row. Raise it to a power so that it works as a winner-take-all. Use that to do two things:\n- Strongly amplify the top pair's increment, so that unambiguous words (1x3, 2x2, high-frequency, 9/12) snowball to high accuracy.\n- Redistribute the withdrawn credit to the competing words, not just discount it.\n\nAlso apply the confusion-reducing effect across all conditions through a stronger winner-take-all shaping at test, i.e. a free ch_dec on the committed-object-discounted strengths. This should pull the dense 3x4, 3x3+1w/o and 4x4 conditions down toward the human level of about .1–.2, because nothing in those conditions becomes committed. The target is about 10 free parameters and a held-out NLL below 15458."}

## Usage

```json
{
  "input_tokens": 9781,
  "output_tokens": 1209,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
