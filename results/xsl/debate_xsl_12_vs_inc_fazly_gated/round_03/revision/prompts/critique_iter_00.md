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
`parameters` maps each name to a string interval "[lo, hi]" (fitted to the data) or a bare number (held fixed). Add a parameter only if it captures something that generalises across conditions (see scoring below). Do not hard-code values taken from the fitted parameters shown below: those fits used every condition, including each cross-validation fold's held-out ones, so a parameter you pin is refitted as free (and counted) whenever a leaderboard theory fits a parameter of the same name. Every theory automatically gets one extra fitted parameter, `lapse`: the rate of pure guessing (uniform over the choices) at test. Do not declare `lapse` yourself, and do not model random guessing inside `learn`.

## HOW THEORIES ARE SCORED
Each theory is FITTED to all 47 conditions at once (one parameter set for everyone) by differential evolution, minimising the negative log-likelihood (NLL) of the human test responses: each person's test responses: whether they chose the correct referent on each item. People differ in how much they learn, so every person gets an ability offset on the log-odds of your predicted accuracy, drawn from a normal distribution whose spread (`ability_sd`) is fitted automatically, like `lapse` - do not declare it. The score is about item-level ACCURACY; which wrong object people chose does not count. Model what an average learner does; individual differences are handled by `ability_sd`.

The SCORE is the held-out NLL from 5-fold cross-validation over conditions: the theory is fitted to four fifths of the conditions and must predict the human choices in the remaining fifth, for each fifth in turn (always the same split), and the five held-out NLLs are summed. Lower is better. Extra parameters that only fit the training conditions do not help here and usually hurt. Also shown, for reference only: BIC = 2*NLL + k*ln(N) of the full-corpus fit (k free parameters incl. `lapse`, N = 31266 choices) and the n_subj-weighted SSE on per-word accuracy (the xsl-bakeoff objective).

## OUTPUT
You have no tools and cannot run code: do not write tool calls, XML tags or `<invoke>` blocks. Reply with the JSON object only — no prose before or after it.
Return one JSON object with these fields:
- interpretation: Where the candidate fits the human data well or badly, and which mechanism is responsible.
- verdict: 'continue' to accept the candidate as this loop's answer, or 'regenerate' to ask for a revision.
- rationale: If 'regenerate': the concrete change to the mechanism or parameters to try next. If 'continue': why it is good enough.


## User Prompt

## CANDIDATE — held-out NLL 16882.7, BIC 33880.0, wSSE (full fit) 0.3696 — REJECTED (not better than this loop's best)
Fitted parameters: d=0.1659, lam=5.997, Mmax=9.735, fl=0.004447, b=1.192, s=0.3032, r=2.112, ch_dec=0.4579, p=1.071, rho=0.2278, th=0.7971, kc=17.16, xi=1, kappa=-0.8983, kpos=0.1937, protect=0.4958, lapse=0.001, ability_sd=1.429 — at bound: lam, xi, lapse

**Name:** Commitment-protected pair-count accumulation

**Description:** Keeps xsl_14's competitive, load-limited, commitment-gated learner. Two changes govern committed words. First, retrieval-induced interference on absent partners is shielded for a committed word's dominant link, in proportion to how committed the word is and how dominant that link is within its row. Filler and unrelated trials therefore stop eroding well-supported pairs. Second, a committed pair's gain grows with its own co-occurrence count, with a wider exponent range than in xsl_14, so late repetitions keep paying off. Uncommitted, ambiguous words keep the ambiguity-scaled diminishing returns, so dense designs stay near floor.

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
- kappa: [-1.5, 0.5]
- kpos: [0.0, 2.0]
- protect: [0.0, 1.0]

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
    protect = params['protect']
    q = 0.07
    c = 0.5
    g = 0.5
    eps = 0.1
    steep = 8.0
    M = np.zeros((n_words, n_objects))
    prev = np.zeros((n_words, n_objects))
    pc = np.zeros((n_words, n_objects))
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
        rs = rsA[words, :]
        mx = mxA[words]
        commit = cA[words]
        rsb = rs[:, objs]
        ratio = (rsb / (mx[:, None] + 1e-9)) ** 4
        blk = M[ix].copy()
        Mw = M[words, :]
        rsum = Mw.sum(1)
        rn = Mw / (rsum[:, None] + 1e-12)
        neff = 1.0 / (np.sum(rn ** 2, axis=1) + 1e-12)
        neff = np.where(rsum > 1e-6, neff, float(no))
        neff = np.maximum(neff, 1.0)
        amb = 1.0 - 1.0 / neff
        boost = 1.0 + r * prev[ix] + kc * commit[:, None] * ratio
        contrib = commit[:, None] * rsb
        tot = contrib.sum(0, keepdims=True)
        cl = np.clip(tot - contrib, 0.0, 1.0)
        # commitment-based protection of dominant links from interference
        domA = (rsA / (mxA[:, None] + 1e-9)) ** 4
        shield = np.clip(protect * cA[:, None] * domA, 0.0, 1.0)
        M[words, :] *= (1.0 - deff * (1.0 - shield[words, :]))
        M[:, objs] *= (1.0 - deff * (1.0 - shield[:, objs]))
        st = (blk + eps) ** s
        stw = st * (1.0 - xi * cl)
        rw = stw / (stw.sum(1, keepdims=True) + 1e-12)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        e_unc = kappa * amb * (1.0 - commit)
        expo_u = np.exp(np.log(1.0 + cnt[words]) * e_unc)[:, None]
        expo_c = (1.0 + pc[ix]) ** (kpos * commit[:, None])
        expo = expo_u * expo_c
        M[ix] = blk + lam * enc * expo * boost * inc * sat
        cnt[words] += 1.0
        pc[ix] += 1.0
        prev *= rho
        prev[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc (full fit) | held-out NLL | held-out NLL minus xsl_14's |
|---|---|---|---|---|---|
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.293 | 314.3 | +1.2 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.308 | 329.6 | +1.0 |
| freq369_39mx | freq369_39mx | 0.622 | 0.440 | 422.3 | +0.9 |
| 205 | 2x4 | 0.303 | 0.298 | 359.1 | +0.5 |
| filt3E_9L | filt3E_9L | 0.575 | 0.503 | 380.5 | +0.4 |
| 221 | 1x4 | 0.190 | 0.232 | 349.9 | +0.3 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.386 | 279.5 | +0.3 |
| 225 | 2x2 | 0.787 | 0.665 | 145.6 | +0.2 |
| 222 | 1x3 | 0.392 | 0.483 | 410.0 | +0.1 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.386 | 302.4 | +0.1 |
| filt6E_9L | filt6E_9L | 0.590 | 0.492 | 343.0 | +0.1 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.886 | 409.9 | +0.1 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.407 | 339.5 | +0.1 |
| filt6E_6L | filt6E_6L | 0.553 | 0.469 | 250.6 | +0.1 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.731 | 349.3 | +0.1 |
| freq369_36mx | freq369_36mx | 0.454 | 0.454 | 727.3 | +0.1 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.853 | 448.1 | +0.0 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.304 | 343.8 | +0.0 |
| filt6E_3L | filt6E_3L | 0.424 | 0.451 | 330.8 | +0.0 |
| 217 | 2x4 6/12 | 0.325 | 0.335 | 148.8 | +0.0 |
| 204 | 3x4 +6o | 0.272 | 0.281 | 188.5 | +0.0 |
| 224 | 3x3 | 0.426 | 0.486 | 356.0 | +0.0 |
| 201 | 3x4 | 0.191 | 0.295 | 211.8 | -0.0 |
| 223 | 4x4 | 0.312 | 0.283 | 763.4 | -0.0 |
| filt3E_3L | filt3E_3L | 0.449 | 0.458 | 362.2 | -0.0 |
| filt0E_9L | filt0E_9L | 0.586 | 0.603 | 455.4 | -0.0 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.307 | 332.8 | -0.0 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.319 | 313.8 | -0.0 |
| 207 | 4x4 +2w/o | 0.103 | 0.081 | 246.8 | -0.0 |
| 218 | 3x4 6/8 | 0.419 | 0.333 | 144.1 | -0.0 |
| 206 | 3x3 +1w/o | 0.167 | 0.232 | 320.8 | -0.0 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.470 | 638.9 | -0.0 |
| filt0E_6L | filt0E_6L | 0.487 | 0.571 | 329.3 | -0.1 |
| 219 | 2x3 6/9 | 0.547 | 0.565 | 345.1 | -0.1 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.520 | 991.9 | -0.1 |
| 202 | 3x4 1/.5 | 0.224 | 0.308 | 205.4 | -0.1 |
| 216 | 2x3 | 0.580 | 0.585 | 190.8 | -0.1 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.734 | 348.9 | -0.1 |
| filt0E_3L | filt0E_3L | 0.372 | 0.502 | 485.9 | -0.1 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.608 | 669.0 | -0.2 |
| filt9E_3L | filt9E_3L | 0.414 | 0.451 | 336.1 | -0.2 |
| 203 | 3x4 1/.66 | 0.211 | 0.298 | 225.9 | -0.2 |
| filt9E_6L | filt9E_6L | 0.513 | 0.469 | 288.5 | -0.2 |
| filt9E_9L | filt9E_9L | 0.601 | 0.492 | 326.7 | -0.3 |
| 215 | 1x3 | 0.744 | 0.518 | 178.9 | -0.3 |
| filt3E_6L | filt3E_6L | 0.482 | 0.478 | 310.7 | -0.3 |
| 220 | 3x4 9/12 | 0.685 | 0.497 | 330.4 | -0.8 |

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
| 1 | xsl_14 | Competition-scaled uncommitted returns with pair-count commitment gain | 17 | 16880.2 | 0.4002 | 33868.7 | d=0.1635, lam=5.999, Mmax=9.733, fl=0.004404, b=1.194, s=0.2981, r=2.114, ch_dec=0.4588, p=1.07, rho=0.2295, th=0.7934, kc=17.16, xi=1, kappa=-0.8956, kpos=0.195, lapse=0.001, ability_sd=1.436 — at bound: lam, xi, lapse |
| 2 | xsl_13 | Commitment-protected, consistency-scaled contiguity exclusion | 18 | 16886.6 | 0.4110 | 33886.7 | d=0.1577, lam=6, Mmax=9.58, fl=0.008242, b=0.9266, s=0.2652, r=2.504, ch_dec=0.5007, p=0.408, rho=0.2276, th=0.7594, kc=17.61, xi=1, kappa=-0.6, kpos=0.03843, pg=0, lapse=0.001, ability_sd=1.441 — at bound: lam, xi, kappa, pg, lapse |
| 3 | xsl_12 | Contiguity-boosted commitment-gated exclusion with exposure scaling | 16 | 16905.5 | 0.4364 | 33874.5 | d=0.1809, lam=6, Mmax=9.874, fl=0.01817, b=0.6339, s=0.2016, r=2.493, ch_dec=0.5767, p=0.2859, rho=0.1662, th=0.7441, kc=17.49, xi=1, kappa=-0.4109, lapse=0.001, ability_sd=1.439 — at bound: lam, xi, lapse |
| 4 | xsl_11 | Commitment-gated exclusion with count-based residual familiarity | 16 | 16913.6 | 0.4306 | 33887.1 | d=0.1912, lam=6, Mmax=30, fl=0.005666, b=1.075, s=0.1386, r=2.801, ch_dec=0.4371, p=0.565, rho=0.1436, th=0.7535, kc=17.76, xi=1, g2=0.01529, lapse=0.001, ability_sd=1.434 — at bound: lam, Mmax, xi, lapse |
| 5 | xsl_09 | Commitment-gated claim exclusion with redistributed credit | 15 | 16921.7 | 0.4333 | 33881.7 | d=0.1542, lam=6, Mmax=30, fl=0.008398, b=0.8015, s=0.1686, r=2.472, ch_dec=0.4477, p=1.32, rho=0.1521, th=0.7918, kc=25.05, xi=1, lapse=0.001, ability_sd=1.437 — at bound: lam, Mmax, xi, lapse |
| 6 | xsl_10 | Streamlined commitment-gated claim exclusion | 15 | 16922.0 | 0.4331 | 33881.7 | d=0.1548, lam=6, Mmax=30, fl=0.00844, b=0.8011, s=0.1683, r=2.472, ch_dec=0.448, p=1.321, rho=0.1498, th=0.7916, kc=25, xi=1, lapse=0.001, ability_sd=1.437 — at bound: lam, Mmax, xi, lapse |
| 7 | xsl_04 | Interference learning with test-time object-claim competition | 9 | 16991.3 | 0.4936 | 33988.3 | d=0.2364, q=0.2247, lam=0.2273, Mmax=0.3, fl=0.0001701, b=0.2557, ch_dec=0.4969, lapse=0.001, ability_sd=1.465 — at bound: Mmax, lapse |
| 8 | xsl_05 | Consecutive-repetition boosted interference learning | 10 | 16995.9 | 0.4968 | 34028.2 | d=0.3121, q=0.2477, lam=0.2913, Mmax=2.195, fl=0.0001764, b=0.02524, r=0.0693, ch_dec=0.4433, lapse=0.001, ability_sd=1.481 — at bound: lapse |
| 9 | xsl_06 | Consolidation-protected boosted interference learning | 11 | 16998.6 | 0.4933 | 34045.2 | d=0.3338, q=0.2506, lam=0.3096, Mmax=3.991, fl=0.0001509, b=4.32e-05, r=1.746e-05, h=16.68, ch_dec=0.4271, lapse=0.001, ability_sd=1.475 — at bound: lapse |
| 10 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 7 | 17050.6 | 0.5456 | 34097.1 | d=0.4057, q=0, lam=0.4589, Mmax=2.244, ch_dec=0.3, lapse=0.001, ability_sd=1.492 — at bound: q, ch_dec, lapse |
| 11 | xsl_08 | Decaying-trace contiguity interference learning | 13 | 17061.6 | 0.5091 | 34027.3 | d=0.2425, q=0, lam=0.6531, Mmax=3.301, fl=0.0002465, b=0.5619, s=1.572, r=1.168, ch_dec=0.4308, p=1.438, rho=0.04663, lapse=0.001, ability_sd=1.464 — at bound: q, lapse |
| 12 | xsl_07 | Soft-sharpness interference learning with contiguity boost | 11 | 17065.7 | 0.5168 | 33986.3 | d=0.1009, q=0.4014, lam=0.8151, Mmax=0.5607, fl=0.0001104, b=3, s=1.098, r=0.4271, ch_dec=0.4643, lapse=0.001, ability_sd=1.461 — at bound: b, lapse |
| 13 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 5 | 17135.1 | 0.6886 | 34253.6 | X=0.0999, B=5.305, C=0.996, lapse=0.001, ability_sd=1.559 — at bound: lapse |
| 14 | inc_fazly | Probabilistic incremental alignment (Fazly, Alishahi & Stevenson 2010) | 4 | 17153.6 | 0.5039 | 34301.1 | lambda=0.1469, beta=21.84, lapse=0.001, ability_sd=1.446 — at bound: lapse |
| 15 | inc_uncfam_attention | Uncertainty/familiarity associative learning with trial-level attention (uncfam_attention) | 5 | 17170.3 | 0.6964 | 34252.1 | X=0.0972, B=5.146, C=0.9953, lapse=0.001, ability_sd=1.562 — at bound: lapse |
| 16 | inc_uncfam_gamma | Uncertainty/familiarity associative learning with a familiarity exponent (uncfam_gamma) | 6 | 17177.2 | 0.7072 | 34212.0 | X=0.08456, B=2.391, C=0.9969, gamma=1.437, lapse=0.001, ability_sd=1.595 — at bound: lapse |
| 17 | inc_tilles | Confidence-flux associative learning (Tilles & Fontanari 2012) | 5 | 17357.2 | 0.7003 | 34678.9 | x=0.1546, b=0.2465, alpha_0=0.9716, lapse=0.001, ability_sd=1.548 — at bound: lapse |
| 18 | inc_bayesian_decay | Bayesian learner with decay (Bayesian_decay) | 5 | 17657.3 | 0.8615 | 35258.0 | alpha=1.297, delta=1.756, ch_dec=9.991, lapse=0.001, ability_sd=1.552 — at bound: ch_dec, lapse |
| 19 | baseline | Co-occurrence counting | 2 | 19568.7 | 1.3742 | 38891.5 | lapse=0.3495, ability_sd=2.874 |

## LOOP HISTORY
(first iteration)

## Response

{"interpretation":"The candidate is essentially xsl_14 plus a 'protect' shield on committed dominant links, and it adds nothing. Held-out NLL is 16882.7 against xsl_14's 16880.2, and wSSE is better (0.370 vs 0.400) but BIC is worse. The per-condition deltas are tiny (within about +/-1 NLL) and mostly slightly positive, so the shield barely changes behaviour. The fitted protect=0.50 is mid-range but the other parameters are almost identical to xsl_14, which means the extra parameter is nearly redundant.\n\nThe model fits poorly in three places.\n- It underpredicts the dense 4x4 frequency-variation conditions (3_x8_369_4x4: 0.29 vs 0.39; 2_x8_39_4x4: 0.31 vs 0.41; freq369_39mx: 0.44 vs 0.62). Their high-frequency words get too little credit, so frequency gains are too weak.\n- It overpredicts 4-AFC conditions with few words per trial (212: 0.89 vs 0.69; 211: 0.85 vs 0.67; 213: 0.73 vs 0.62). With ch_dec=0.46, choices are flat across the n-AFC candidates, so the model overestimates accuracy when the test has only 4 candidates.\n- It misses condition-specific effects: 215 (1x3) is underpredicted (0.52 vs 0.74), while 222 and filt0E_3L are overpredicted. The model is also weak on the 2x2 and filtering conditions (225, filt9E_9L).\n\nThe mechanism is a heavily parameterised pile of heuristics: a sigmoid commitment gate, competition scaling, a boost with kc=17, and several count exponents. Several parameters sit at their bounds (lam, xi). The best model's lead over simpler ones is only about 100 NLL, so complexity is not paying off. Refining the protection term is a dead end.","verdict":"regenerate","rationale":"Drop the protect/shield mechanism and revert to the xsl_14 structure. Then fix the 4-AFC versus n-AFC mismatch and the weak frequency effect with one structural change. The test-time Luce rule uses a single ch_dec, but the human data suggest sharper, more winner-take-all choices when the correct association is strong and few candidates are present. Make the output nonlinear in association strength: for example, output = M*share**b plus a floor, then apply a saturating transform such as 1-exp(-M/Mscale), or let the effective competitor weight scale with row-normalised dominance. Concretely, replace the additive floor fl with a floor that is proportional to each row's total mass, so a word with strong learning has near-zero noise on competitors. That would raise accuracy for high-frequency words in the 4x4 conditions. Also let the strength gain for a pair grow with its repetition count in a way that is not damped by within-trial ambiguity for committed words. This needs fewer parameters: remove kpos, protect and kappa, or fix kappa at its range edge. The aim is to match xsl_14 with fewer parameters, which would improve BIC and held-out NLL."}

## Usage

```json
{
  "input_tokens": 11873,
  "output_tokens": 1073,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
