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
Each theory is FITTED to all 47 conditions at once (one parameter set for everyone) by differential evolution, minimising the negative log-likelihood (NLL) of the human test responses: each person's test responses: whether they chose the correct referent on each item. People differ in how much they learn, so every person gets an ability offset on the log-odds of your predicted accuracy, drawn from a normal distribution whose spread (`ability_sd`) is fitted automatically, like `lapse` - do not declare it. The score is about item-level ACCURACY; which wrong object people chose does not count. Model what an average learner does; individual differences are handled by `ability_sd`.

The SCORE is the held-out NLL from 5-fold cross-validation over conditions: the theory is fitted to four fifths of the conditions and must predict the human choices in the remaining fifth, for each fifth in turn (always the same split), and the five held-out NLLs are summed. Lower is better. Extra parameters that only fit the training conditions do not help here and usually hurt. Also shown, for reference only: BIC = 2*NLL + k*ln(N) of the full-corpus fit (k free parameters incl. `lapse`, N = 31266 choices) and the n_subj-weighted SSE on per-word accuracy (the xsl-bakeoff objective).

## OUTPUT
You have no tools and cannot run code: do not write tool calls, XML tags or `<invoke>` blocks. Reply with the JSON object only — no prose before or after it.
Return one JSON object with these fields:
- interpretation: Where the candidate fits the human data well or badly, and which mechanism is responsible.
- verdict: 'continue' to accept the candidate as this loop's answer, or 'regenerate' to ask for a revision.
- rationale: If 'regenerate': the concrete change to the mechanism or parameters to try next. If 'continue': why it is good enough.


## User Prompt

## CANDIDATE — held-out NLL 16915.1, BIC 33829.9, wSSE (full fit) 0.3795 — REJECTED (not better than this loop's best)
Fitted parameters: lam=6, Mmax=30, rho=0.15, th=0.79, kc=25, xi=1, d=0.1556, fl=0.008442, b=0.8033, s=0.1671, r=2.486, ch_dec=0.4483, p=1.469, kappa=0, lapse=0.001, ability_sd=1.437 — at bound: kappa, lapse

**Name:** Pair-count protected commitment-gated exclusion

**Description:** Learning follows the commitment-gated claim-exclusion account. Credit is the product of word-side and object-side responsibilities, with a fading trace boosting repeated pairs. Retrieval-induced interference erodes links to absent partners. A word commits once one object dominates its association, and that object is withdrawn from competing co-present words, with the freed credit redistributed. The new claim is that interference damage is not uniform across links. A link that has co-occurred many times is better consolidated, so its vulnerability to retrieval-induced interference falls as 1/(1+kappa*count). Frequently repeated pairs, such as the 9-repetition words and the 1x3 designs with many repetitions, therefore keep their strength under load. Rarely seen pairs remain fully exposed to interference and are pruned.

**Parameters:**
- d: [0.0, 1.0]
- lam: 6
- Mmax: 30
- fl: [0.0, 0.5]
- b: [0.0, 3.0]
- s: [0.01, 4]
- r: [0.0, 3.0]
- ch_dec: [0.3, 3]
- p: [-0.5, 1.5]
- rho: 0.15
- th: 0.79
- kc: 25
- xi: 1
- kappa: [0.0, 2.0]

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
    q = 0.07
    c = 0.5
    g = 0.5
    eps = 0.1
    steep = 8.0
    M = np.zeros((n_words, n_objects))
    C = np.zeros((n_words, n_objects))
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
        prot = 1.0 / (1.0 + kappa * C)
        M[words, :] *= (1.0 - deff * prot[words, :])
        M[:, objs] *= (1.0 - deff * prot[:, objs])
        st = (blk + eps) ** s
        stw = st * (1.0 - xi * cl)
        rw = stw / (stw.sum(1, keepdims=True) + 1e-12)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        M[ix] = blk + lam * enc * boost * inc * sat
        C[ix] += 1.0
        prev *= rho
        prev[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc (full fit) | held-out NLL | held-out NLL minus xsl_10's |
|---|---|---|---|---|---|
| cont_div6-12 | cont_div6-12 | 0.435 | 0.460 | 639.5 | +0.5 |
| 218 | 3x4 6/8 | 0.419 | 0.348 | 144.4 | +0.2 |
| filt6E_9L | filt6E_9L | 0.590 | 0.494 | 343.5 | +0.2 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.312 | 343.2 | +0.2 |
| 206 | 3x3 +1w/o | 0.167 | 0.251 | 322.7 | +0.1 |
| 203 | 3x4 1/.66 | 0.211 | 0.309 | 226.3 | +0.1 |
| filt6E_6L | filt6E_6L | 0.553 | 0.478 | 251.2 | +0.1 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.399 | 341.0 | +0.1 |
| 201 | 3x4 | 0.191 | 0.306 | 211.9 | +0.1 |
| 202 | 3x4 1/.5 | 0.224 | 0.319 | 206.3 | +0.1 |
| 223 | 4x4 | 0.312 | 0.287 | 764.1 | +0.0 |
| 215 | 1x3 | 0.744 | 0.526 | 177.4 | +0.0 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.612 | 670.6 | +0.0 |
| 207 | 4x4 +2w/o | 0.103 | 0.109 | 247.0 | +0.0 |
| 219 | 2x3 6/9 | 0.547 | 0.544 | 345.8 | +0.0 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.744 | 348.4 | +0.0 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.331 | 313.4 | +0.0 |
| filt6E_3L | filt6E_3L | 0.424 | 0.458 | 335.8 | +0.0 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.319 | 332.8 | +0.0 |
| 225 | 2x2 | 0.787 | 0.679 | 145.7 | +0.0 |
| 205 | 2x4 | 0.303 | 0.317 | 360.0 | +0.0 |
| freq369_36mx | freq369_36mx | 0.454 | 0.435 | 730.1 | +0.0 |
| filt3E_6L | filt3E_6L | 0.482 | 0.490 | 310.9 | +0.0 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.857 | 449.6 | +0.0 |
| filt3E_3L | filt3E_3L | 0.449 | 0.467 | 363.1 | +0.0 |
| filt9E_3L | filt9E_3L | 0.414 | 0.458 | 338.3 | +0.0 |
| 204 | 3x4 +6o | 0.272 | 0.280 | 188.1 | +0.0 |
| filt9E_6L | filt9E_6L | 0.513 | 0.478 | 288.6 | +0.0 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.744 | 350.6 | -0.0 |
| 216 | 2x3 | 0.580 | 0.557 | 190.5 | -0.0 |
| filt0E_6L | filt0E_6L | 0.487 | 0.605 | 330.7 | -0.0 |
| filt3E_9L | filt3E_9L | 0.575 | 0.507 | 378.8 | -0.0 |
| 221 | 1x4 | 0.190 | 0.269 | 352.8 | -0.0 |
| 217 | 2x4 6/12 | 0.325 | 0.352 | 149.8 | -0.0 |
| freq369_39mx | freq369_39mx | 0.622 | 0.422 | 423.7 | -0.0 |
| filt0E_9L | filt0E_9L | 0.586 | 0.649 | 455.4 | -0.0 |
| 220 | 3x4 9/12 | 0.685 | 0.530 | 326.4 | -0.0 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.383 | 302.3 | -0.0 |
| filt0E_3L | filt0E_3L | 0.372 | 0.507 | 487.8 | -0.0 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.310 | 332.7 | -0.0 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.491 | 990.1 | -0.0 |
| 224 | 3x3 | 0.426 | 0.475 | 355.8 | -0.0 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.875 | 412.3 | -0.0 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.383 | 278.3 | -0.1 |
| 222 | 1x3 | 0.392 | 0.500 | 411.3 | -0.1 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.293 | 317.9 | -0.1 |
| filt9E_9L | filt9E_9L | 0.601 | 0.494 | 328.2 | -0.1 |

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
| 1 | xsl_10 | Streamlined commitment-gated claim exclusion | 9 | 16913.8 | 0.4301 | 33819.6 | lam=6, Mmax=30, rho=0.15, th=0.79, kc=25, xi=1, d=0.1536, fl=0.008372, b=0.8005, s=0.1675, r=2.471, ch_dec=0.4473, p=1.32, lapse=0.001, ability_sd=1.438 — at bound: lapse |
| 2 | xsl_09 | Commitment-gated claim exclusion with redistributed credit | 15 | 16921.7 | 0.4333 | 33881.7 | d=0.1542, lam=6, Mmax=30, fl=0.008398, b=0.8015, s=0.1686, r=2.472, ch_dec=0.4477, p=1.32, rho=0.1521, th=0.7918, kc=25.05, xi=1, lapse=0.001, ability_sd=1.437 — at bound: lam, Mmax, xi, lapse |
| 3 | xsl_04 | Interference learning with test-time object-claim competition | 9 | 16991.3 | 0.4936 | 33988.3 | d=0.2364, q=0.2247, lam=0.2273, Mmax=0.3, fl=0.0001701, b=0.2557, ch_dec=0.4969, lapse=0.001, ability_sd=1.465 — at bound: Mmax, lapse |
| 4 | xsl_05 | Consecutive-repetition boosted interference learning | 10 | 16995.9 | 0.4968 | 34028.2 | d=0.3121, q=0.2477, lam=0.2913, Mmax=2.195, fl=0.0001764, b=0.02524, r=0.0693, ch_dec=0.4433, lapse=0.001, ability_sd=1.481 — at bound: lapse |
| 5 | xsl_06 | Consolidation-protected boosted interference learning | 11 | 16998.6 | 0.4933 | 34045.2 | d=0.3338, q=0.2506, lam=0.3096, Mmax=3.991, fl=0.0001509, b=4.32e-05, r=1.746e-05, h=16.68, ch_dec=0.4271, lapse=0.001, ability_sd=1.475 — at bound: lapse |
| 6 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 7 | 17050.6 | 0.5456 | 34097.1 | d=0.4057, q=0, lam=0.4589, Mmax=2.244, ch_dec=0.3, lapse=0.001, ability_sd=1.492 — at bound: q, ch_dec, lapse |
| 7 | xsl_08 | Decaying-trace contiguity interference learning | 13 | 17061.6 | 0.5091 | 34027.3 | d=0.2425, q=0, lam=0.6531, Mmax=3.301, fl=0.0002465, b=0.5619, s=1.572, r=1.168, ch_dec=0.4308, p=1.438, rho=0.04663, lapse=0.001, ability_sd=1.464 — at bound: q, lapse |
| 8 | xsl_07 | Soft-sharpness interference learning with contiguity boost | 11 | 17065.7 | 0.5168 | 33986.3 | d=0.1009, q=0.4014, lam=0.8151, Mmax=0.5607, fl=0.0001104, b=3, s=1.098, r=0.4271, ch_dec=0.4643, lapse=0.001, ability_sd=1.461 — at bound: b, lapse |
| 9 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 5 | 17135.1 | 0.6886 | 34253.6 | X=0.0999, B=5.305, C=0.996, lapse=0.001, ability_sd=1.559 — at bound: lapse |
| 10 | inc_fazly | Probabilistic incremental alignment (Fazly, Alishahi & Stevenson 2010) | 4 | 17153.6 | 0.5039 | 34301.1 | lambda=0.1469, beta=21.84, lapse=0.001, ability_sd=1.446 — at bound: lapse |
| 11 | inc_uncfam_attention | Uncertainty/familiarity associative learning with trial-level attention (uncfam_attention) | 5 | 17170.3 | 0.6964 | 34252.1 | X=0.0972, B=5.146, C=0.9953, lapse=0.001, ability_sd=1.562 — at bound: lapse |
| 12 | inc_uncfam_gamma | Uncertainty/familiarity associative learning with a familiarity exponent (uncfam_gamma) | 6 | 17177.2 | 0.7072 | 34212.0 | X=0.08456, B=2.391, C=0.9969, gamma=1.437, lapse=0.001, ability_sd=1.595 — at bound: lapse |
| 13 | inc_tilles | Confidence-flux associative learning (Tilles & Fontanari 2012) | 5 | 17357.2 | 0.7003 | 34678.9 | x=0.1546, b=0.2465, alpha_0=0.9716, lapse=0.001, ability_sd=1.548 — at bound: lapse |
| 14 | inc_bayesian_decay | Bayesian learner with decay (Bayesian_decay) | 5 | 17657.3 | 0.8615 | 35258.0 | alpha=1.297, delta=1.756, ch_dec=9.991, lapse=0.001, ability_sd=1.552 — at bound: ch_dec, lapse |
| 15 | baseline | Co-occurrence counting | 2 | 19568.7 | 1.3742 | 38891.5 | lapse=0.3495, ability_sd=2.874 |

## LOOP HISTORY
- iter 0: Commitment-gated exclusion with count-based residual familiarity — held-out NLL 16914.5 (BIC 33827.1) — ACCEPTED
  critic: Drop the g2 familiarity term, which is fitted near zero and adds no held-out gain, and return to the xsl_10 structure with 9 parameters. Then add one generalising mechanism. Make the strength of the test-time choice competition depend on the number of candidate options, for example ch_dec_eff = ch_dec * (1 + k*log(n_candidates)), or alternatively apply a lower exponent for m-AFC than for n-AFC. This targets the systematic overprediction of 4-AFC conditions (211, 212, 213), where the model predicts 0.74-0.88 against human 0.62-0.69. Alternatively, or in addition, replace the hard saturation sat=1-M/Mmax with a repetition-sensitive commitment threshold, so that high-frequency words (220, freq369_39mx, 215) keep accumulating credit. This should reduce the underprediction at high frequency without adding free parameters beyond one.
- iter 1: Exposure-sensitive commitment-gated exclusion — held-out NLL 16915.2 (BIC 33832.4) — REJECTED
  critic: Drop u and the exposure-gated threshold, and return to the 9-parameter xsl_10 structure. Then add one mechanism that addresses the 4-AFC overprediction.

Make the test-time choice sharpness depend on the number of candidates. For example, use ch_dec_eff = ch_dec * (n_cand / n_objects)**k with one new parameter k, or equivalently a smaller exponent for m-AFC than for n-AFC. This needs the candidate count to reach the choice rule, which `learn` cannot see. Because `learn` cannot see it, implement it indirectly. Compress the output matrix toward the row mean with a strength that depends on the training design, or use the ch_dec rule with a fixed lower value together with a stronger share^b discount. The aim is to cut the 4-AFC accuracies of 0.74–0.88 to 0.62–0.69 without hurting the n-AFC conditions.

Alternatively, replace the hard sat=1-M/Mmax cap with a softer one, or reduce deff as a pair's repetition count grows (for example deff scaled by 1/(1+kappa*pair_count)). This lets high-frequency words (215, 220, freq369_39mx, filt*_9L) keep gaining credit instead of being eroded by interference. Try this one first, because it is cheap and needs only one parameter.

## Response

{"interpretation":"The candidate is effectively xsl_10 with an extra protection parameter, kappa, that the fit drove to its lower bound of 0. With kappa=0, prot=1 and the model is identical to xsl_10 plus a dead count matrix. Held-out NLL is 16915.1 against 16913.8, a small loss from the extra parameter and a different fit trajectory. The pair-count protection idea therefore gave no gain, and the data do not support it. The remaining misfits are structural. (1) 4-AFC conditions (211, 212, 213) are overpredicted at 0.74-0.88 against human 0.62-0.69. (2) High-frequency or high-repetition conditions are underpredicted: 215 at 0.53 vs 0.74, freq369_39mx at 0.42 vs 0.62, 220 at 0.53 vs 0.69, and filt*_9L. (3) Some low-overlap or filtering conditions are overpredicted: filt0E_3L at 0.51 vs 0.37 and filt0E_6L at 0.61 vs 0.49. (4) Several 3x4 and 1x4 n-AFC conditions are overpredicted: 201-203 at about 0.31 vs 0.19-0.22 and 221 at 0.27 vs 0.19. The model gets the large ranking right but is too flat across frequency. The 1x3 pair is also inconsistent: 215 is underpredicted while 222 is overpredicted, though both are nominally the same design. The shortfall at high frequency is not caused by interference vulnerability, since kappa found no use for it. It more likely comes from the saturation cap Mmax=30 combined with competition and the gated exclusion, which keep a repeated correct pair from out-growing its rivals.","verdict":"regenerate","rationale":"Drop kappa and the C count matrix and return to the 9-parameter xsl_10 structure. Then change the test-time readout so that it addresses the 4-AFC overprediction and the frequency flatness. The candidate-set size is not visible inside learn, so use a readout that is not tied to it. Replace the additive floor fl and the output sharpening out = M*share**b with a row-wise relative-dominance transform. For each word, compute dom = M[i,j]/(M[i].max()+delta), then out = dom**gamma weighted by an absolute-strength term (1-exp(-M/m0)). Then a word with a large, clearly dominant association gets a high choice probability. A word with a weak or diffuse association stays near chance, so choice among few candidates (4-AFC) is less inflated by within-row differences. Separately, make the saturation soft and frequency-sensitive by replacing sat=1-blk/Mmax with sat=1/(1+(blk/Mmax)**2), or by raising Mmax to a free parameter in [30,200]. That lets repeated correct pairs, such as 215, 220, freq369_39mx and filt*_9L, keep accumulating credit. Keep the other fixed constants. Add at most one new free parameter, either gamma or m0."}

## Usage

```json
{
  "input_tokens": 11351,
  "output_tokens": 1024,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
