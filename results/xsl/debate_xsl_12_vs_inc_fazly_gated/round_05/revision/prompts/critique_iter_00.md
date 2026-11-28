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

## CANDIDATE — held-out NLL 16914.7, BIC 33963.7, wSSE (full fit) 0.3726 — REJECTED (not better than this loop's best)
Fitted parameters: d=0.1085, lam=4.969, Mmax=6.213, fl=0.01149, b=0.7899, s=0.3353, r=3.007, ch_dec=0.5229, p=1.467, rho=0.2881, th=0.8956, kc=38.5, xi=0.7683, kappa=-0.4774, kpos=0.7777, pr=0.4624, ksat=0.359, pu=0.1751, lapse=0.001, ability_sd=1.616 — at bound: lapse

**Name:** Protected-commitment competitive learner with evidence-growing ceiling and winner sharpening

**Description:** This keeps xsl_14's competitive, load-limited learning with retrieval interference, a fading contiguity trace, commitment-gated exclusion, ambiguity-scaled returns for uncommitted words and pair-count gain for committed ones. It changes how learning behaves after a word commits. (1) Protected commitment: retrieval-induced interference erodes competitors, not the committed association. The decay applied to a word's row and an object's column is weakened for the word's dominant link, in proportion to commitment and to how dominant the link is. (2) Evidence-dependent ceiling: the saturation cap of a committed pair grows with the number of its consistent co-occurrences, Mmax*(1+n_pair)^ksat. Repetition then keeps strengthening committed pairs instead of flattening. (3) Winner sharpening: when a committed word's dominant object is present, the credit just given to its other co-present objects is partly pruned and moved to the dominant object. Repeated consistent evidence therefore sharpens the word's meaning, as in an incremental alignment learner.

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
- kpos: [0.0, 1.0]
- pr: [0.0, 1.0]
- ksat: [0.0, 1.5]
- pu: [0.0, 0.5]

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
    pr = params['pr']
    ksat = params['ksat']
    pu = params['pu']
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
        # protected decay: dominant link of a committed word is shielded
        ratioAll = (rsA / (mxA[:, None] + 1e-9)) ** 4
        P = pr * cA[:, None] * ratioAll
        M[words, :] *= (1.0 - deff * (1.0 - P[words, :]))
        M[:, objs] *= (1.0 - deff * (1.0 - P[:, objs]))
        st = (blk + eps) ** s
        stw = st * (1.0 - xi * cl)
        rw = stw / (stw.sum(1, keepdims=True) + 1e-12)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        Mm = Mmax * (1.0 + pc[ix] * commit[:, None]) ** ksat
        sat = np.clip(1.0 - blk / Mm, 0.0, 1.0)
        e_unc = kappa * amb * (1.0 - commit)
        expo_u = np.exp(np.log(1.0 + cnt[words]) * e_unc)[:, None]
        expo_c = (1.0 + pc[ix]) ** (kpos * commit[:, None])
        expo = expo_u * expo_c
        new = blk + lam * enc * expo * boost * inc * sat
        # winner sharpening: prune competitors of a present committed winner
        wpres = ratio.max(1)
        rem = pu * commit[:, None] * wpres[:, None] * (1.0 - ratio) * new
        new = new - rem
        jw = np.argmax(ratio, axis=1)
        ar = np.arange(nw)
        gain = rem.sum(1) * ratio[ar, jw]
        new[ar, jw] += gain
        M[ix] = new
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
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.512 | 1001.8 | +9.8 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.316 | 335.2 | +6.6 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.292 | 318.7 | +5.6 |
| freq369_36mx | freq369_36mx | 0.454 | 0.440 | 732.3 | +5.0 |
| freq369_39mx | freq369_39mx | 0.622 | 0.425 | 425.2 | +3.8 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.888 | 413.0 | +3.2 |
| 221 | 1x4 | 0.190 | 0.268 | 351.8 | +2.2 |
| 207 | 4x4 +2w/o | 0.103 | 0.098 | 249.0 | +2.1 |
| 225 | 2x2 | 0.787 | 0.670 | 147.0 | +1.6 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.311 | 334.4 | +1.6 |
| filt6E_6L | filt6E_6L | 0.553 | 0.486 | 252.0 | +1.4 |
| 205 | 2x4 | 0.303 | 0.302 | 360.0 | +1.4 |
| filt6E_3L | filt6E_3L | 0.424 | 0.462 | 332.0 | +1.2 |
| 206 | 3x3 +1w/o | 0.167 | 0.242 | 322.1 | +1.2 |
| filt3E_3L | filt3E_3L | 0.449 | 0.466 | 363.1 | +0.8 |
| 217 | 2x4 6/12 | 0.325 | 0.361 | 149.6 | +0.8 |
| filt0E_3L | filt0E_3L | 0.372 | 0.486 | 486.7 | +0.7 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.407 | 279.9 | +0.7 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.752 | 349.9 | +0.6 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.396 | 339.9 | +0.5 |
| 204 | 3x4 +6o | 0.272 | 0.264 | 188.9 | +0.4 |
| filt6E_9L | filt6E_9L | 0.590 | 0.515 | 343.3 | +0.4 |
| 219 | 2x3 6/9 | 0.547 | 0.549 | 345.5 | +0.4 |
| filt0E_6L | filt0E_6L | 0.487 | 0.571 | 329.7 | +0.4 |
| 202 | 3x4 1/.5 | 0.224 | 0.296 | 205.8 | +0.3 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.745 | 349.3 | +0.3 |
| filt9E_6L | filt9E_6L | 0.513 | 0.485 | 289.0 | +0.3 |
| 222 | 1x3 | 0.392 | 0.450 | 410.2 | +0.2 |
| filt0E_9L | filt0E_9L | 0.586 | 0.609 | 455.6 | +0.2 |
| 224 | 3x3 | 0.426 | 0.472 | 356.2 | +0.2 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.407 | 302.5 | +0.2 |
| filt9E_3L | filt9E_3L | 0.414 | 0.462 | 336.3 | +0.1 |
| 201 | 3x4 | 0.191 | 0.287 | 211.8 | -0.0 |
| 223 | 4x4 | 0.312 | 0.273 | 763.1 | -0.2 |
| 218 | 3x4 6/8 | 0.419 | 0.351 | 143.8 | -0.4 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.304 | 343.3 | -0.4 |
| 215 | 1x3 | 0.744 | 0.494 | 178.5 | -0.7 |
| filt3E_9L | filt3E_9L | 0.575 | 0.525 | 379.4 | -0.7 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.330 | 313.1 | -0.7 |
| 203 | 3x4 1/.66 | 0.211 | 0.290 | 225.3 | -0.8 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.839 | 446.6 | -1.5 |
| 216 | 2x3 | 0.580 | 0.591 | 189.4 | -1.5 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.625 | 667.6 | -1.6 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.455 | 637.2 | -1.7 |
| filt3E_6L | filt3E_6L | 0.482 | 0.493 | 309.2 | -1.9 |
| filt9E_9L | filt9E_9L | 0.601 | 0.514 | 324.8 | -2.2 |
| 220 | 3x4 9/12 | 0.685 | 0.530 | 325.9 | -5.3 |

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

{"interpretation":"The candidate (held-out NLL 16914.7) is rejected: it is worse than xsl_14 (16880.2) and has 18 free parameters against 17. The three new mechanisms (protected commitment, evidence-growing ceiling, winner sharpening) add 3 parameters (pr, ksat, pu) but gain nothing in held-out NLL. The full-fit wSSE of 0.373 is better than xsl_14's 0.400, so the extra flexibility overfits the training folds without generalising.\n\nThe worst misfits are the same ones xsl_14 already has, and they get slightly worse:\n- Conditions with 3x3 or 4x4 items per trial are under-predicted: freq369-3x3hiCD (0.555 human vs 0.512 model), 2_x8_39_4x4 (0.41 vs 0.32), 3_x8_369_4x4 (0.39 vs 0.29), freq369_39mx (0.62 vs 0.43). These are mixed-frequency and high-contextual-diversity designs. The model cannot exploit high-frequency words as anchors that disambiguate the others.\n- Easy small-load or 4AFC conditions are mis-scaled. 1x3 (215) is 0.74 human vs 0.49 model. 212 (2x3 6/9 4AFC) is 0.69 vs 0.89. 211 (1x3 6/18 4AFC) is 0.67 vs 0.84. 220 (3x4 9/12) is 0.69 vs 0.53. The model is too weak on unambiguous 1-word trials and too strong on 4AFC. This points to a poor interaction between the n-AFC and m-AFC tests and the ch_dec/b/lapse readout. It also suggests that learning accumulates too weakly with frequency in some designs and too strongly in others.\n- Several mechanisms are highly redundant. The decay, interference, contiguity trace, commitment gating, ambiguity exponent and pair-count gain overlap, so the parameters are poorly identified.\n\nThe added mechanisms are small perturbations of an already saturated model. The loop's progress has stalled in the 16880–16920 range.","verdict":"regenerate","rationale":"Revert to xsl_14, the best theory, and drop the pr, ksat and pu additions. Then simplify instead of adding. Replace the separate commitment, ambiguity and pair-count mechanisms with one principled mechanism: mutual-exclusivity or competitive normalisation driven by the learner's current knowledge of the other words. Concretely, before crediting a pair, discount each co-present object by the summed row-normalised strength it already has with other co-present words (explaining-away). Then let a word's credit be distributed by a softmax over its present objects, with temperature as a single parameter.\n\nTo fix the 1x3 and 4AFC mis-scaling, add a frequency-dependent, test-independent gain. Examples are strengthening that grows with the number of consistent co-occurrences of a pair (a power law with one exponent), or an unambiguous-trial boost for trials with a single word. Drop kc, xi, kappa, kpos and the commitment sigmoid (th, steep) if the explaining-away term covers the same function. Aim for 12 parameters or fewer, so that held-out NLL beats 16880."}

## Usage

```json
{
  "input_tokens": 12274,
  "output_tokens": 1118,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
