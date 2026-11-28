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
Each theory is FITTED to all 47 conditions at once (one parameter set for everyone) by differential evolution, minimising the negative log-likelihood (NLL) of the human test responses: each person's test responses: whether they chose the correct referent on each item. People differ in how much they learn, so every person gets an ability offset on the log-odds of your predicted accuracy, drawn from a normal distribution whose spread (`ability_sd`) is fitted automatically, like `lapse` - do not declare it. The score is about item-level ACCURACY; which wrong object people chose does not count. Model what an average learner does; individual differences are handled by `ability_sd`.

The SCORE is the held-out NLL from 5-fold cross-validation over conditions: the theory is fitted to four fifths of the conditions and must predict the human choices in the remaining fifth, for each fifth in turn (always the same split), and the five held-out NLLs are summed. Lower is better. Extra parameters that only fit the training conditions do not help here and usually hurt. Also shown, for reference only: BIC = 2*NLL + k*ln(N) of the full-corpus fit (k free parameters incl. `lapse`, N = 31266 choices) and the n_subj-weighted SSE on per-word accuracy (the xsl-bakeoff objective).

## OUTPUT
You have no tools and cannot run code: do not write tool calls, XML tags or `<invoke>` blocks. Reply with the JSON object only — no prose before or after it.
Return one JSON object with these fields:
- interpretation: Where the candidate fits the human data well or badly, and which mechanism is responsible.
- verdict: 'continue' to accept the candidate as this loop's answer, or 'regenerate' to ask for a revision.
- rationale: If 'regenerate': the concrete change to the mechanism or parameters to try next. If 'continue': why it is good enough.


## User Prompt

## CANDIDATE — held-out NLL 16899.7, BIC 33823.7, wSSE (full fit) 0.3776 — ACCEPTED
Fitted parameters: lam=6, Mmax=30, rho=0.15, th=0.79, kc=25, xi=1, d=0.1952, fl=0.007991, b=0.4758, s=0.2179, r=3, ch_dec=0.4565, p=0.6977, g0=0, kappa=-0.3252, lapse=0.001, ability_sd=1.434 — at bound: r, g0, lapse

**Name:** Exposure-scaled commitment-gated exclusion with passive decay

**Description:** Learning is competitive and interference-limited. Credit is the product of word-side and object-side responsibilities, retrieval-induced interference erodes links to absent partners, and a fading trace amplifies repeated pairs. A word commits once one object dominates its association, and its object is then withdrawn from competing co-present words with the freed credit redistributed. Memory also fades passively over trials. The new claim is that encoding efficiency depends on a word's accumulated exposure. Credit per occurrence scales as (1+n_i)^kappa, where n_i is the number of earlier appearances of word i. A positive kappa means familiar words are encoded more efficiently, so learning keeps rising with frequency. A negative kappa means diminishing returns, which stops frequent words from snowballing to certainty. The data decide the sign of kappa. At test, objects claimed by other words are discounted by share^b and choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- lam: 6
- Mmax: 30
- fl: [0.0, 0.5]
- b: [0.0, 3.0]
- s: [0.01, 4]
- r: [0.0, 3.0]
- ch_dec: [0.3, 3]
- p: [-0.5, 3.0]
- rho: 0.15
- th: 0.79
- kc: 25
- xi: 1
- g0: [0.0, 0.1]
- kappa: [-0.6, 1.0]

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
    g0 = params['g0']
    kappa = params['kappa']
    q = 0.07
    c = 0.5
    g = 0.5
    eps = 0.1
    steep = 8.0
    M = np.zeros((n_words, n_objects))
    prev = np.zeros((n_words, n_objects))
    cnt = np.zeros(n_words)
    for words, objs in trials:
        words = np.asarray(words)
        objs = np.asarray(objs)
        nw = len(words)
        no = len(objs)
        enc = 1.0 / (1.0 + (no / c) ** g)
        deff = min(0.95, d * (nw * no / 12.0) ** q * (no / 4.0) ** p)
        ix = np.ix_(words, objs)
        M *= (1.0 - g0)
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
        stw = st * (1.0 - xi * cl)
        rw = stw / (stw.sum(1, keepdims=True) + 1e-12)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        expo = ((1.0 + cnt[words]) ** kappa)[:, None]
        M[ix] = blk + lam * enc * expo * boost * inc * sat
        cnt[words] += 1.0
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
| filt3E_3L | filt3E_3L | 0.449 | 0.480 | 365.8 | +2.6 |
| 220 | 3x4 9/12 | 0.685 | 0.500 | 328.9 | +2.4 |
| filt9E_6L | filt9E_6L | 0.513 | 0.482 | 290.0 | +1.4 |
| 215 | 1x3 | 0.744 | 0.499 | 178.7 | +1.4 |
| 205 | 2x4 | 0.303 | 0.292 | 360.7 | +0.7 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.465 | 639.7 | +0.6 |
| freq369_36mx | freq369_36mx | 0.454 | 0.440 | 730.5 | +0.4 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.305 | 343.4 | +0.3 |
| filt0E_3L | filt0E_3L | 0.372 | 0.522 | 488.1 | +0.3 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.844 | 449.8 | +0.2 |
| 223 | 4x4 | 0.312 | 0.280 | 764.2 | +0.1 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.731 | 350.7 | +0.1 |
| 204 | 3x4 +6o | 0.272 | 0.272 | 188.2 | +0.1 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.385 | 278.5 | +0.1 |
| 218 | 3x4 6/8 | 0.419 | 0.335 | 144.3 | +0.1 |
| 216 | 2x3 | 0.580 | 0.548 | 190.5 | +0.0 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.730 | 348.4 | +0.0 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.323 | 313.4 | +0.0 |
| 224 | 3x3 | 0.426 | 0.477 | 355.8 | +0.0 |
| filt0E_9L | filt0E_9L | 0.586 | 0.643 | 455.4 | +0.0 |
| filt9E_3L | filt9E_3L | 0.414 | 0.467 | 338.2 | -0.0 |
| 222 | 1x3 | 0.392 | 0.475 | 411.3 | -0.0 |
| 225 | 2x2 | 0.787 | 0.684 | 145.6 | -0.0 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.385 | 302.3 | -0.1 |
| filt6E_9L | filt6E_9L | 0.590 | 0.494 | 343.2 | -0.1 |
| 219 | 2x3 6/9 | 0.547 | 0.537 | 345.7 | -0.1 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.405 | 340.7 | -0.2 |
| filt0E_6L | filt0E_6L | 0.487 | 0.605 | 330.5 | -0.2 |
| filt6E_6L | filt6E_6L | 0.553 | 0.485 | 250.8 | -0.3 |
| 207 | 4x4 +2w/o | 0.103 | 0.101 | 246.6 | -0.3 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.872 | 412.0 | -0.3 |
| 217 | 2x4 6/12 | 0.325 | 0.332 | 149.5 | -0.4 |
| 203 | 3x4 1/.66 | 0.211 | 0.295 | 225.7 | -0.5 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.611 | 670.1 | -0.5 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.310 | 332.2 | -0.6 |
| filt6E_3L | filt6E_3L | 0.424 | 0.468 | 335.2 | -0.6 |
| 202 | 3x4 1/.5 | 0.224 | 0.308 | 205.5 | -0.7 |
| filt3E_6L | filt3E_6L | 0.482 | 0.504 | 310.0 | -0.8 |
| filt3E_9L | filt3E_9L | 0.575 | 0.514 | 378.0 | -0.8 |
| 206 | 3x3 +1w/o | 0.167 | 0.236 | 321.6 | -1.0 |
| 201 | 3x4 | 0.191 | 0.291 | 210.6 | -1.3 |
| 221 | 1x4 | 0.190 | 0.232 | 351.2 | -1.7 |
| freq369_39mx | freq369_39mx | 0.622 | 0.431 | 421.9 | -1.8 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.311 | 330.5 | -2.3 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.291 | 315.2 | -2.8 |
| filt9E_9L | filt9E_9L | 0.601 | 0.490 | 324.6 | -3.7 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.501 | 986.2 | -3.9 |

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
| 2 | xsl_11 | Commitment-gated exclusion with count-based residual familiarity | 10 | 16914.5 | 0.4374 | 33827.1 | lam=6, Mmax=30, rho=0.15, th=0.79, kc=25, xi=1, d=0.174, fl=0.005832, b=1.026, s=0.1498, r=2.558, ch_dec=0.4324, p=1.285, g2=0.0101, lapse=0.001, ability_sd=1.436 — at bound: lapse |
| 3 | xsl_09 | Commitment-gated claim exclusion with redistributed credit | 15 | 16921.7 | 0.4333 | 33881.7 | d=0.1542, lam=6, Mmax=30, fl=0.008398, b=0.8015, s=0.1686, r=2.472, ch_dec=0.4477, p=1.32, rho=0.1521, th=0.7918, kc=25.05, xi=1, lapse=0.001, ability_sd=1.437 — at bound: lam, Mmax, xi, lapse |
| 4 | xsl_04 | Interference learning with test-time object-claim competition | 9 | 16991.3 | 0.4936 | 33988.3 | d=0.2364, q=0.2247, lam=0.2273, Mmax=0.3, fl=0.0001701, b=0.2557, ch_dec=0.4969, lapse=0.001, ability_sd=1.465 — at bound: Mmax, lapse |
| 5 | xsl_05 | Consecutive-repetition boosted interference learning | 10 | 16995.9 | 0.4968 | 34028.2 | d=0.3121, q=0.2477, lam=0.2913, Mmax=2.195, fl=0.0001764, b=0.02524, r=0.0693, ch_dec=0.4433, lapse=0.001, ability_sd=1.481 — at bound: lapse |
| 6 | xsl_06 | Consolidation-protected boosted interference learning | 11 | 16998.6 | 0.4933 | 34045.2 | d=0.3338, q=0.2506, lam=0.3096, Mmax=3.991, fl=0.0001509, b=4.32e-05, r=1.746e-05, h=16.68, ch_dec=0.4271, lapse=0.001, ability_sd=1.475 — at bound: lapse |
| 7 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 7 | 17050.6 | 0.5456 | 34097.1 | d=0.4057, q=0, lam=0.4589, Mmax=2.244, ch_dec=0.3, lapse=0.001, ability_sd=1.492 — at bound: q, ch_dec, lapse |
| 8 | xsl_08 | Decaying-trace contiguity interference learning | 13 | 17061.6 | 0.5091 | 34027.3 | d=0.2425, q=0, lam=0.6531, Mmax=3.301, fl=0.0002465, b=0.5619, s=1.572, r=1.168, ch_dec=0.4308, p=1.438, rho=0.04663, lapse=0.001, ability_sd=1.464 — at bound: q, lapse |
| 9 | xsl_07 | Soft-sharpness interference learning with contiguity boost | 11 | 17065.7 | 0.5168 | 33986.3 | d=0.1009, q=0.4014, lam=0.8151, Mmax=0.5607, fl=0.0001104, b=3, s=1.098, r=0.4271, ch_dec=0.4643, lapse=0.001, ability_sd=1.461 — at bound: b, lapse |
| 10 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 5 | 17135.1 | 0.6886 | 34253.6 | X=0.0999, B=5.305, C=0.996, lapse=0.001, ability_sd=1.559 — at bound: lapse |
| 11 | inc_fazly | Probabilistic incremental alignment (Fazly, Alishahi & Stevenson 2010) | 4 | 17153.6 | 0.5039 | 34301.1 | lambda=0.1469, beta=21.84, lapse=0.001, ability_sd=1.446 — at bound: lapse |
| 12 | inc_uncfam_attention | Uncertainty/familiarity associative learning with trial-level attention (uncfam_attention) | 5 | 17170.3 | 0.6964 | 34252.1 | X=0.0972, B=5.146, C=0.9953, lapse=0.001, ability_sd=1.562 — at bound: lapse |
| 13 | inc_uncfam_gamma | Uncertainty/familiarity associative learning with a familiarity exponent (uncfam_gamma) | 6 | 17177.2 | 0.7072 | 34212.0 | X=0.08456, B=2.391, C=0.9969, gamma=1.437, lapse=0.001, ability_sd=1.595 — at bound: lapse |
| 14 | inc_tilles | Confidence-flux associative learning (Tilles & Fontanari 2012) | 5 | 17357.2 | 0.7003 | 34678.9 | x=0.1546, b=0.2465, alpha_0=0.9716, lapse=0.001, ability_sd=1.548 — at bound: lapse |
| 15 | inc_bayesian_decay | Bayesian learner with decay (Bayesian_decay) | 5 | 17657.3 | 0.8615 | 35258.0 | alpha=1.297, delta=1.756, ch_dec=9.991, lapse=0.001, ability_sd=1.552 — at bound: ch_dec, lapse |
| 16 | baseline | Co-occurrence counting | 2 | 19568.7 | 1.3742 | 38891.5 | lapse=0.3495, ability_sd=2.874 |

## LOOP HISTORY
- iter 0: Commitment-gated claim exclusion with passive time decay — held-out NLL 16908.7 (BIC 33821.6) — ACCEPTED
  critic: Drop the passive-decay parameter g0, which buys about 5 NLL for 5 extra parameters relative to xsl_10 and is not what limits the fit. Instead make the choice stage format-aware. Keep ch_dec, but apply a candidate-set-size-dependent sharpening at test. This is done inside `learn` as a competitor-normalised association. Subtract the word's row baseline, i.e. m_ij - fl_row, where fl_row is a fraction of the word's mean association over all objects. That gives a distinctiveness signal that is scale-free across n-AFC and 4-AFC and compresses weak, diffuse rows. Also make the commitment boost and the saturation depend on cumulative per-word exposure. For example, replace the constant lam*enc with lam*enc*(1+kappa*log(1+n_i)), where n_i is the number of prior appearances of word i. That would let learning rise with repetition (filt*_9L, freq369_39mx, 220, 225) without a snowball. Drop the unneeded fixed constants and keep the parameter count near 9-10. Fit kappa in [0,2]. Check that 211/212 (4AFC) drop toward 0.7 while 215 and the high-frequency conditions rise.

## Response

{"interpretation":"The candidate is accepted, but its held-out NLL (16899.7) beats the xsl_10 leader (16913.8) by only 14 NLL. That is within noise for a model with about 6 extra parameters (BIC 33823.7 vs 33819.6, so no BIC gain). The fitted kappa is -0.33, which means diminishing returns with exposure. That is the opposite of what the critic hoped for (frequency-driven rise). The systematic misfits therefore remain. (1) High-frequency and high-learning conditions are underpredicted: 220 (0.685 human vs 0.50 model), 215 (0.744 vs 0.50), freq369_39mx (0.62 vs 0.43), filt9E_9L, filt6E_9L, filt9E_6L, 2_x8_39_4x4, 3_x8_369_4x4. The negative kappa makes frequent words saturate too early, so repetition is penalised where humans keep improving. (2) The 4AFC conditions 211, 212 and 213 are overpredicted (0.84, 0.87, 0.73 vs about 0.62-0.69). The power-law choice rule with ch_dec=0.46 is shared by n-AFC and m-AFC, and it is too flat for n-AFC while still too confident for m-AFC. (3) filt0E_3L and filt0E_6L are overpredicted (0.52 vs 0.37, 0.61 vs 0.49). Passive decay (g0) hit its lower bound of 0, so that mechanism is unused. The same goes for r at its upper bound of 3 and lapse at its lower bound. The extra parameters g0 and kappa are therefore dead weight or counterproductive.","verdict":"regenerate","rationale":"Remove g0 and the kappa exposure term (or fix kappa to 0) to return to the xsl_10 core with about 9 parameters. Then address the real misfit, which is the test and choice stage and the learning-rate form, not the exposure scaling. (a) Replace the global share^b discount and ch_dec with a format-aware readout. Convert the row to a distinctiveness signal, m_ij - mean over the row of m_i, clipped at 0 and added to fl. Then apply ch_dec to this signal. This should lower the 4AFC predictions (211, 212 and 213 overpredicted by 0.1-0.2) without flattening n-AFC. (b) Raise r's upper bound above 3, since it sits at the bound, so consecutive-repetition boosting can be stronger. Also let r's boost scale with the within-word recency of the last co-occurrence. Together these should raise 215, 220 and the high-frequency filt*_9L conditions. (c) Make saturation, sat = 1 - blk/Mmax, depend on the number of competing candidates, no*nw. This softens the interference cap in small-context conditions such as 215 and 225 and lets learning keep rising with frequency. Keep the commitment-gated exclusion unchanged."}

## Usage

```json
{
  "input_tokens": 11210,
  "output_tokens": 1000,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
