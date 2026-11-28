# query_xsl_13_attempt_00

## System Prompt

You are a cognitive scientist arguing for one theory of cross-situational word learning against another, using an existing corpus of human data.

## THE TASK AND THE DATA
On each training trial a learner hears several novel words and sees several novel objects, with no indication of which word goes with which; each word's referent is consistent across trials. At test, people choose each word's referent. The corpus has 47 adult conditions that vary words and objects per trial, word frequency, contextual diversity, temporal contiguity, filtering and test format. Both theories are fitted to all of it; their predictions are population means of per-item accuracy (people differ in ability, which the fit models explicitly).

## YOUR MOVE
Propose one CORPUS QUERY: a set of conditions and a statistic of their per-item accuracies on which YOUR theory (xsl_13) should match the human value more closely than the opponent (inc_fazly). Good queries isolate an effect your theory's mechanism produces and the opponent's does not (e.g. a contrast between conditions that differ in one design factor, or a pattern across items within a condition).

The query is evaluated on the human data and on both theories' predictions. It counts only if the two predictions differ by at least 2 standard errors of the human value (estimated by resampling people) - otherwise the data cannot decide it. The theory closer to the human value wins the query. Do not repeat a query already in the ledger.

## OUTPUT
You have no tools and cannot run code: do not write tool calls, XML tags or `<invoke>` blocks. Reply with the JSON object only — no prose before or after it.
Return one JSON object with these fields:
- conditions: Labels of the corpus conditions the statistic uses (at least one).
- metric: Python source defining def metric(acc) -> float, where acc maps each listed condition label to a numpy array of per-item accuracies (proportion correct, in corpus item order). `np` is available. Example: return acc['221'].mean() - acc['222'].mean()
- claim: Which theory should match people better on this statistic, and the mechanism that makes it so.


## User Prompt

## THEORIES
### THEORY 1: xsl_13 — held-out NLL 16886.6, held-out wSSE 0.4110
Fitted parameters: d=0.1577, lam=6, Mmax=9.58, fl=0.008242, b=0.9266, s=0.2652, r=2.504, ch_dec=0.5007, p=0.408, rho=0.2276, th=0.7594, kc=17.61, xi=1, kappa=-0.6, kpos=0.03843, pg=0, lapse=0.001, ability_sd=1.441

**Name:** Commitment-protected, consistency-scaled contiguity exclusion

**Description:** This keeps xsl_12's account: competitive, load-dependent, interference-limited learning, a fading contiguity trace, and commitment-gated exclusion with redistributed credit. It changes how repetition and interference treat strong associations. (1) Encoding efficiency depends on whether a word's evidence is consistent. A word that has not committed gets diminishing returns on exposure (negative kappa), so ambiguous, frequent words do not snowball. A word whose top object dominates its association (commitment) gets increasing returns, with exponent kpos on exposure, so repetition pays off in easy, consistent designs. (2) Retrieval-induced interference on absent partners is gated by commitment. The links of a committed word-object pair are shielded from erosion in proportion to how dominant that link is within the word's association. Fillers, or other words and objects on the trial, therefore cannot erode a well-supported pair. (3) Exclusion still only withdraws claimed objects from other words, never from the committed word's own credit. At test, objects claimed by other words are discounted by share^b, and choice is a power-law Luce rule.

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
- kappa: [-0.6, 1.0]
- kpos: [0.0, 0.6]
- pg: [0.0, 1.0]

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
    pg = params['pg']
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
        rsA = M / (M.sum(1, keepdims=True) + 0.3)
        mxA = rsA.max(1)
        zA = np.clip(steep * (mxA - th), -30.0, 30.0)
        cA = 1.0 / (1.0 + np.exp(-zA))
        PA = pg * cA[:, None] * (rsA / (mxA[:, None] + 1e-9)) ** 4
        rs = rsA[words, :]
        mx = mxA[words]
        commit = cA[words]
        rsb = rs[:, objs]
        ratio = (rsb / (mx[:, None] + 1e-9)) ** 4
        blk = M[ix].copy()
        boost = 1.0 + r * prev[ix] + kc * commit[:, None] * ratio
        contrib = commit[:, None] * rsb
        tot = contrib.sum(0, keepdims=True)
        cl = np.clip(tot - contrib, 0.0, 1.0)
        M[words, :] *= (1.0 - deff * (1.0 - PA[words, :]))
        M[:, objs] *= (1.0 - deff * (1.0 - PA[:, objs]))
        st = (blk + eps) ** s
        stw = st * (1.0 - xi * cl)
        rw = stw / (stw.sum(1, keepdims=True) + 1e-12)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        e = kappa * (1.0 - commit) + kpos * commit
        expo = ((1.0 + cnt[words]) ** e)[:, None]
        M[ix] = blk + lam * enc * expo * boost * inc * sat
        cnt[words] += 1.0
        prev *= rho
        prev[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


### THEORY 2: inc_fazly — held-out NLL 17153.6, held-out wSSE 0.5039
Fitted parameters: lambda=0.1469, beta=21.84, lapse=0.001, ability_sd=1.446

**Name:** Probabilistic incremental alignment (Fazly, Alishahi & Stevenson 2010)

**Description:** On each trial every word aligns to each co-present object in proportion to its current probability of meaning that object (smoothed for never-seen pairs); alignments accumulate as associations, and each word's meaning probabilities are the smoothed, normalised associations.

**Parameters:**
- lambda: [1e-10, 2]
- beta: [2, 20000]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    lam, beta = params["lambda"], params["beta"]
    alpha, eps = 10.0, 0.001  # XSLmodels fazly() defaults
    assoc = np.zeros((n_words, n_objects))
    probs = np.zeros((n_words, n_objects))
    t_unseen = np.full(n_words, 1.0 / beta)
    for words, objects in trials:
        tr = probs[np.ix_(words, objects)].copy()
        unseen = tr == 0
        tr[unseen] = np.broadcast_to(t_unseen[words][:, None], tr.shape)[unseen]
        align = (tr + eps) / (tr.sum(axis=0, keepdims=True) + alpha * eps)
        assoc[np.ix_(words, objects)] += align
        denom = assoc[words, :].sum(axis=1) + beta * lam
        probs[words, :] = (assoc[words, :] + lam) / denom[:, None]
        t_unseen[words] = lam / denom
    return probs
```


## CONDITIONS (mean accuracy; your theory is the first prediction column)
| label | condition | words x objects | trials | per trial | test | n_subj | human acc | xsl_13 acc | inc_fazly acc | held-out NLL xsl_13 minus inc_fazly |
|---|---|---|---|---|---|---|---|---|---|---|
| 201 | 3x4 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.191 | 0.346 | 0.424 | -5.2 |
| 202 | 3x4 1/.5 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.224 | 0.353 | 0.423 | -7.2 |
| 203 | 3x4 1/.66 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.211 | 0.347 | 0.422 | -4.0 |
| 204 | 3x4 +6o | 18x24 | 36 | 3.0x4.0 | 24-AFC | 20 | 0.272 | 0.333 | 0.410 | -2.3 |
| 205 | 2x4 | 18x18 | 54 | 2.0x4.0 | 18-AFC | 33 | 0.303 | 0.351 | 0.386 | -2.2 |
| 206 | 3x3 +1w/o | 18x18 | 54 | 3.0x3.0 | 18-AFC | 39 | 0.167 | 0.303 | 0.390 | -9.2 |
| 207 | 4x4 +2w/o | 18x18 | 54 | 4.0x4.0 | 18-AFC | 39 | 0.103 | 0.150 | 0.256 | -10.0 |
| 211 | 1x3 6/18 4AFC | 18x18 | 108 | 1.0x3.0 | 4-AFC | 43 | 0.674 | 0.785 | 0.701 | -14.1 |
| 212 | 2x3 6/9 4AFC | 18x18 | 54 | 2.0x3.0 | 4-AFC | 38 | 0.686 | 0.819 | 0.755 | -17.0 |
| 213 | 2x4 6/12 4AFC | 18x18 | 54 | 2.0x4.0 | 4-AFC | 31 | 0.624 | 0.677 | 0.711 | -15.0 |
| 214 | 3x4 6/8 4AFC | 18x18 | 36 | 3.0x4.0 | 4-AFC | 36 | 0.688 | 0.677 | 0.712 | -9.1 |
| 215 | 1x3 | 18x18 | 108 | 1.0x3.0 | 18-AFC | 23 | 0.744 | 0.511 | 0.382 | -15.1 |
| 216 | 2x3 | 18x18 | 54 | 2.0x3.0 | 18-AFC | 23 | 0.580 | 0.558 | 0.454 | -2.8 |
| 217 | 2x4 6/12 | 18x18 | 54 | 2.0x4.0 | 18-AFC | 14 | 0.325 | 0.380 | 0.386 | +2.0 |
| 218 | 3x4 6/8 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 13 | 0.419 | 0.377 | 0.419 | -0.6 |
| 219 | 2x3 6/9 | 18x18 | 54 | 2.0x3.0 | 18-AFC | 32 | 0.547 | 0.547 | 0.456 | -2.5 |
| 220 | 3x4 9/12 | 18x18 | 54 | 3.0x4.0 | 18-AFC | 33 | 0.685 | 0.479 | 0.479 | +3.3 |
| 221 | 1x4 | 18x18 | 108 | 1.0x4.0 | 18-AFC | 40 | 0.190 | 0.304 | 0.325 | -1.7 |
| 222 | 1x3 | 18x18 | 108 | 1.0x3.0 | 18-AFC | 40 | 0.392 | 0.491 | 0.377 | +4.3 |
| 223 | 4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 77 | 0.312 | 0.335 | 0.435 | -4.7 |
| 224 | 3x3 | 18x18 | 36 | 3.0x3.0 | 18-AFC | 36 | 0.426 | 0.496 | 0.486 | -0.4 |
| 225 | 2x2 | 18x18 | 54 | 2.0x2.0 | 18-AFC | 19 | 0.787 | 0.623 | 0.560 | -4.6 |
| 301 | 2x2 5x 24AFC | 24x24 | 60 | 2.0x2.0 | 24-AFC | 46 | 0.514 | 0.578 | 0.497 | +2.6 |
| filt0E_3L | filt0E_3L | 12x12 | 18 | 2.0x2.0 | 11-AFC | 31 | 0.372 | 0.504 | 0.513 | -0.3 |
| filt3E_3L | filt3E_3L | 12x12 | 27 | 2.0x2.0 | 11-AFC | 30 | 0.449 | 0.464 | 0.470 | -30.4 |
| filt6E_3L | filt6E_3L | 12x12 | 36 | 2.0x2.0 | 11-AFC | 30 | 0.424 | 0.457 | 0.459 | -28.6 |
| filt9E_3L | filt9E_3L | 12x12 | 45 | 2.0x2.0 | 11-AFC | 31 | 0.414 | 0.457 | 0.454 | -19.1 |
| filt0E_6L | filt0E_6L | 12x12 | 36 | 2.0x2.0 | 11-AFC | 23 | 0.487 | 0.555 | 0.629 | -2.6 |
| filt3E_6L | filt3E_6L | 12x12 | 45 | 2.0x2.0 | 11-AFC | 23 | 0.482 | 0.482 | 0.565 | +5.3 |
| filt6E_6L | filt6E_6L | 12x12 | 54 | 2.0x2.0 | 11-AFC | 23 | 0.553 | 0.473 | 0.538 | -8.0 |
| filt9E_6L | filt9E_6L | 12x12 | 63 | 2.0x2.0 | 11-AFC | 23 | 0.513 | 0.471 | 0.525 | +2.0 |
| filt0E_9L | filt0E_9L | 12x12 | 54 | 2.0x2.0 | 11-AFC | 31 | 0.586 | 0.578 | 0.694 | -2.8 |
| filt3E_9L | filt3E_9L | 12x12 | 63 | 2.0x2.0 | 11-AFC | 31 | 0.575 | 0.498 | 0.624 | -24.0 |
| filt6E_9L | filt6E_9L | 12x12 | 72 | 2.0x2.0 | 11-AFC | 31 | 0.590 | 0.488 | 0.589 | -2.8 |
| filt9E_9L | filt9E_9L | 12x12 | 81 | 2.0x2.0 | 11-AFC | 31 | 0.601 | 0.486 | 0.570 | +2.6 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.411 | 0.349 | 0.413 | +6.1 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.391 | 0.340 | 0.411 | +5.9 |
| freq369-3x3loCD | freq369-3x3loCD | 18x18 | 36 | 3.0x3.0 | 18-AFC | 34 | 0.485 | 0.434 | 0.428 | -16.5 |
| freq369-3x3hiCD | freq369-3x3hiCD | 18x18 | 36 | 3.0x3.0 | 18-AFC | 104 | 0.555 | 0.515 | 0.483 | -4.0 |
| freq369_36mx | freq369_36mx | 18x18 | 36 | 3.0x3.0 | 18-AFC | 66 | 0.454 | 0.469 | 0.456 | -14.4 |
| freq369_39mx | freq369_39mx | 18x18 | 36 | 3.0x3.0 | 18-AFC | 40 | 0.622 | 0.459 | 0.451 | -1.9 |
| cont_div6-12 | cont_div6-12 | 18x18 | 36 | 3.0x3.0 | 18-AFC | 62 | 0.435 | 0.486 | 0.481 | -3.1 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.489 | 0.415 | 0.435 | -1.1 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 18x18 | 27 | 4.0x4.0 | 18-AFC | 32 | 0.405 | 0.415 | 0.435 | +0.5 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.317 | 0.365 | 0.433 | -7.4 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.280 | 0.356 | 0.436 | -5.9 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.414 | 0.353 | 0.437 | -0.8 |

## LEDGER (all queries so far)
- proposed by xsl_12 on 221, 222: human 0.2014 (SE 0.0497); xsl_12 0.1936 (0.2 SE off), inc_fazly 0.0525 (3.0 SE off); winner: xsl_12
- proposed by inc_fazly on filt3E_9L, filt6E_9L, filt9E_9L: human 0.5887 (SE 0.0222); xsl_12 0.4911 (4.4 SE off), inc_fazly 0.5941 (0.2 SE off); winner: inc_fazly
- proposed by xsl_13 on 216, 219: human 0.5633 (SE 0.0431); xsl_13 0.5527 (0.2 SE off), inc_fazly 0.4549 (2.5 SE off); winner: xsl_13
- proposed by inc_fazly on filt6E_6L, filt9E_6L, filt9E_9L, 2_x8_39_4x4, 3_x8_369_4x4: human 0.4936 (SE 0.0205); xsl_13 0.4237 (3.4 SE off), inc_fazly 0.4914 (0.1 SE off); winner: inc_fazly

## Response

{"conditions":["207"],"metric":"def metric(acc):\n    return float(acc['207'].mean())","claim":"In 4x4 +2w/o (207), people are near floor (about 0.10). xsl_13 predicts about 0.15 and inc_fazly about 0.26, so xsl_13 should be closer. With four words and four objects per trial and many words competing, xsl_13 applies load-dependent, interference-limited encoding. Words that have not committed get diminishing returns on exposure, and the encoding efficiency falls as objects per trial rise. Ambiguous high-load designs therefore stay near floor. inc_fazly has no load or interference limit. Its alignment normalisation keeps accumulating cross-situational evidence, so it overpredicts accuracy in this high-ambiguity condition."}

## Usage

```json
{
  "input_tokens": 7256,
  "output_tokens": 1182,
  "reasoning_tokens": 910,
  "provider": "Claude Platform on AWS"
}
```
