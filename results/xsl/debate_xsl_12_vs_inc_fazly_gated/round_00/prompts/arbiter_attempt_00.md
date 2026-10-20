# arbiter_attempt_00

## System Prompt

You are the arbiter of a debate between two theories of cross-situational word learning, fitted to a corpus of 47 adult conditions. Each round, each side proposed a corpus query - a statistic on which it expected to match people better - and the queries were scored against the human data.

Decide which theory to revise this round, and how: 'new_model' keeps its central claim and changes its mechanism or implementation; 'new_theory' replaces it with a different account. Base the decision on what the queries reveal (which effects each theory gets right or wrong, by how many standard errors), on the cumulative ledger, and on the cross-validated held-out NLL of each theory. Give one concrete, mechanistic recommendation for the target.

## OUTPUT
You have no tools and cannot run code: do not write tool calls, XML tags or `<invoke>` blocks. Reply with the JSON object only — no prose before or after it.
Return one JSON object with these fields:
- interpretation: What this round's outcomes show about where each theory succeeds or fails, and why.
- target: Which theory to revise: 1 or 2 (as numbered in THEORIES).
- verdict: 'new_model': keep the target theory's central claim but change its mechanism or implementation; 'new_theory': replace it with a different account.
- recommendation: The concrete change to make to the target, grounded in the failed queries.


## User Prompt

## THEORIES
### THEORY 1: xsl_12 — held-out NLL 16905.5, held-out wSSE 0.4364
Fitted parameters: d=0.1809, lam=6, Mmax=9.874, fl=0.01817, b=0.6339, s=0.2016, r=2.493, ch_dec=0.5767, p=0.2859, rho=0.1662, th=0.7441, kc=17.49, xi=1, kappa=-0.4109, lapse=0.001, ability_sd=1.439

**Name:** Contiguity-boosted commitment-gated exclusion with exposure scaling

**Description:** Learning is competitive and interference-limited. Each pair gains the product of word-side and object-side responsibilities. Retrieval-induced interference erodes links to absent partners, at a rate set by trial load and by the number of objects. A fading trace of recent co-occurrence strongly amplifies repeated pairs, so temporal contiguity and immediate repetition have a large effect. A word commits once one object dominates its association, and that object is then withdrawn from other co-present words, which redistributes the freed credit. Encoding efficiency changes with a word's accumulated exposure, with diminishing returns when kappa is negative, so frequent words do not snowball to certainty. At test, objects claimed by other words are discounted by share^b, and choice is a power-law Luce rule. There is no passive decay, because it was fitted to zero.

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


## THIS ROUND'S QUERIES
- proposed by xsl_12 on 221, 222: human 0.2014 (SE 0.0497); xsl_12 0.1936 (0.2 SE off), inc_fazly 0.0525 (3.0 SE off); winner: xsl_12
  claim: xsl_12 should match people better on the drop in accuracy from 1 word x 3 objects (222) to 1 word x 4 objects (221). The two conditions have the same 108 trials, the same 18-AFC test and 40 subjects each. Humans show about +0.20 (0.392 vs 0.190). xsl_12 predicts about +0.19 (0.488 vs 0.294). Its encoding efficiency enc = 1/(1+(no/c)^g) and its object-count-dependent interference (the (no/4)^p term) both worsen steeply as the number of objects per trial grows, and its contiguity boost helps in the smaller-ambiguity case. inc_fazly predicts only about +0.05 (0.377 vs 0.325). Its normalised alignment makes accuracy depend only weakly on the number of candidate objects, so it underestimates the cost of the extra object.
- proposed by inc_fazly on filt3E_9L, filt6E_9L, filt9E_9L: human 0.5887 (SE 0.0222); xsl_12 0.4911 (4.4 SE off), inc_fazly 0.5941 (0.2 SE off); winner: inc_fazly
  claim: inc_fazly should match people better. In the 12x12, 2x2 conditions with 9 late trials per word, human accuracy is about 0.59, and inc_fazly predicts about 0.59 (mean over the three conditions). xsl_12 predicts only about 0.49. Fazly's mechanism accumulates alignment mass with each repeated, consistent co-occurrence. Its probabilities sharpen steadily with exposure and keep benefiting from the extra repetitions. xsl_12 has diminishing returns on exposure (negative kappa), and its commitment-gated exclusion and interference cap how far accuracy rises with repetition. It therefore under-predicts these high-repetition conditions by about 0.1, which is several standard errors of the human value.

## EARLIER ROUNDS
(no queries yet)

## CONDITIONS
| label | condition | words x objects | trials | per trial | test | n_subj | human acc | xsl_12 acc | inc_fazly acc | held-out NLL xsl_12 minus inc_fazly |
|---|---|---|---|---|---|---|---|---|---|---|
| 201 | 3x4 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.191 | 0.345 | 0.424 | -6.2 |
| 202 | 3x4 1/.5 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.224 | 0.355 | 0.423 | -7.2 |
| 203 | 3x4 1/.66 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.211 | 0.347 | 0.422 | -3.6 |
| 204 | 3x4 +6o | 18x24 | 36 | 3.0x4.0 | 24-AFC | 20 | 0.272 | 0.334 | 0.410 | -2.6 |
| 205 | 2x4 | 18x18 | 54 | 2.0x4.0 | 18-AFC | 33 | 0.303 | 0.347 | 0.386 | -1.2 |
| 206 | 3x3 +1w/o | 18x18 | 54 | 3.0x3.0 | 18-AFC | 39 | 0.167 | 0.302 | 0.390 | -9.1 |
| 207 | 4x4 +2w/o | 18x18 | 54 | 4.0x4.0 | 18-AFC | 39 | 0.103 | 0.150 | 0.256 | -10.2 |
| 211 | 1x3 6/18 4AFC | 18x18 | 108 | 1.0x3.0 | 4-AFC | 43 | 0.674 | 0.781 | 0.701 | -13.1 |
| 212 | 2x3 6/9 4AFC | 18x18 | 54 | 2.0x3.0 | 4-AFC | 38 | 0.686 | 0.812 | 0.755 | -15.7 |
| 213 | 2x4 6/12 4AFC | 18x18 | 54 | 2.0x4.0 | 4-AFC | 31 | 0.624 | 0.672 | 0.711 | -15.2 |
| 214 | 3x4 6/8 4AFC | 18x18 | 36 | 3.0x4.0 | 4-AFC | 36 | 0.688 | 0.676 | 0.712 | -8.3 |
| 215 | 1x3 | 18x18 | 108 | 1.0x3.0 | 18-AFC | 23 | 0.744 | 0.505 | 0.382 | -14.9 |
| 216 | 2x3 | 18x18 | 54 | 2.0x3.0 | 18-AFC | 23 | 0.580 | 0.546 | 0.454 | -2.9 |
| 217 | 2x4 6/12 | 18x18 | 54 | 2.0x4.0 | 18-AFC | 14 | 0.325 | 0.374 | 0.386 | +2.3 |
| 218 | 3x4 6/8 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 13 | 0.419 | 0.376 | 0.419 | -0.3 |
| 219 | 2x3 6/9 | 18x18 | 54 | 2.0x3.0 | 18-AFC | 32 | 0.547 | 0.538 | 0.456 | -2.1 |
| 220 | 3x4 9/12 | 18x18 | 54 | 3.0x4.0 | 18-AFC | 33 | 0.685 | 0.476 | 0.479 | +3.9 |
| 221 | 1x4 | 18x18 | 108 | 1.0x4.0 | 18-AFC | 40 | 0.190 | 0.294 | 0.325 | -1.8 |
| 222 | 1x3 | 18x18 | 108 | 1.0x3.0 | 18-AFC | 40 | 0.392 | 0.488 | 0.377 | +5.2 |
| 223 | 4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 77 | 0.312 | 0.334 | 0.435 | -3.9 |
| 224 | 3x3 | 18x18 | 36 | 3.0x3.0 | 18-AFC | 36 | 0.426 | 0.495 | 0.486 | -0.5 |
| 225 | 2x2 | 18x18 | 54 | 2.0x2.0 | 18-AFC | 19 | 0.787 | 0.631 | 0.560 | -4.4 |
| 301 | 2x2 5x 24AFC | 24x24 | 60 | 2.0x2.0 | 24-AFC | 46 | 0.514 | 0.581 | 0.497 | +3.6 |
| filt0E_3L | filt0E_3L | 12x12 | 18 | 2.0x2.0 | 11-AFC | 31 | 0.372 | 0.522 | 0.513 | +1.4 |
| filt3E_3L | filt3E_3L | 12x12 | 27 | 2.0x2.0 | 11-AFC | 30 | 0.449 | 0.476 | 0.470 | -26.8 |
| filt6E_3L | filt6E_3L | 12x12 | 36 | 2.0x2.0 | 11-AFC | 30 | 0.424 | 0.463 | 0.459 | -27.8 |
| filt9E_3L | filt9E_3L | 12x12 | 45 | 2.0x2.0 | 11-AFC | 31 | 0.414 | 0.460 | 0.454 | -18.1 |
| filt0E_6L | filt0E_6L | 12x12 | 36 | 2.0x2.0 | 11-AFC | 23 | 0.487 | 0.585 | 0.629 | -1.3 |
| filt3E_6L | filt3E_6L | 12x12 | 45 | 2.0x2.0 | 11-AFC | 23 | 0.482 | 0.492 | 0.565 | +4.0 |
| filt6E_6L | filt6E_6L | 12x12 | 54 | 2.0x2.0 | 11-AFC | 23 | 0.553 | 0.476 | 0.538 | -7.4 |
| filt9E_6L | filt9E_6L | 12x12 | 63 | 2.0x2.0 | 11-AFC | 23 | 0.513 | 0.472 | 0.525 | +3.2 |
| filt0E_9L | filt0E_9L | 12x12 | 54 | 2.0x2.0 | 11-AFC | 31 | 0.586 | 0.613 | 0.694 | -2.7 |
| filt3E_9L | filt3E_9L | 12x12 | 63 | 2.0x2.0 | 11-AFC | 31 | 0.575 | 0.504 | 0.624 | -24.8 |
| filt6E_9L | filt6E_9L | 12x12 | 72 | 2.0x2.0 | 11-AFC | 31 | 0.590 | 0.487 | 0.589 | -3.6 |
| filt9E_9L | filt9E_9L | 12x12 | 81 | 2.0x2.0 | 11-AFC | 31 | 0.601 | 0.482 | 0.570 | +1.0 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.411 | 0.344 | 0.413 | +7.6 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.391 | 0.336 | 0.411 | +7.5 |
| freq369-3x3loCD | freq369-3x3loCD | 18x18 | 36 | 3.0x3.0 | 18-AFC | 34 | 0.485 | 0.434 | 0.428 | -15.1 |
| freq369-3x3hiCD | freq369-3x3hiCD | 18x18 | 36 | 3.0x3.0 | 18-AFC | 104 | 0.555 | 0.507 | 0.483 | -4.0 |
| freq369_36mx | freq369_36mx | 18x18 | 36 | 3.0x3.0 | 18-AFC | 66 | 0.454 | 0.465 | 0.456 | -13.5 |
| freq369_39mx | freq369_39mx | 18x18 | 36 | 3.0x3.0 | 18-AFC | 40 | 0.622 | 0.459 | 0.451 | -2.2 |
| cont_div6-12 | cont_div6-12 | 18x18 | 36 | 3.0x3.0 | 18-AFC | 62 | 0.435 | 0.488 | 0.481 | -1.0 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.489 | 0.416 | 0.435 | -1.8 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 18x18 | 27 | 4.0x4.0 | 18-AFC | 32 | 0.405 | 0.416 | 0.435 | +0.4 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.317 | 0.365 | 0.433 | -7.5 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.280 | 0.355 | 0.436 | -6.6 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.414 | 0.352 | 0.437 | -0.8 |

## Response

{"interpretation":"The round split. xsl_12 won the 221 vs 222 contrast. Humans lose about 0.20 going from 3 to 4 objects per trial, xsl_12 predicts +0.19, and inc_fazly predicts only +0.05, 3 SE off. So xsl_12's object-count-dependent encoding and interference is a real mechanism that Fazly lacks. inc_fazly won on the late-repetition filler conditions (filt*E_9L). Humans reach about 0.59 there, Fazly gets 0.59, and xsl_12 gets 0.49, 4.4 SE low. In xsl_12 accuracy is flat or falling as extra trials are added (filt3E_9L 0.50, filt6E_9L 0.49, filt9E_9L 0.48). It does not benefit from repetition in the 12x12, 2x2 designs. The same pattern shows up elsewhere: 220 (0.476 vs human 0.685), 215 (0.505 vs 0.744), and 225 (0.63 vs 0.79). xsl_12 under-predicts conditions where learning is easy and repetition should pay off. The culprits are the negative kappa, which gives diminishing returns on exposure, plus erosion of links to absent partners and exclusion. These cap and even erode strong, consistent associations. On held-out fit xsl_12 is better overall (NLL 16905 vs 17154, wSSE 0.436 vs 0.504). Fazly is also far worse on the 3x4 and 4x4 high-ambiguity conditions, where it is 0.1 or more too high. xsl_12 is therefore the stronger base and its failures are localized, so it should be revised rather than replaced.","target":1,"verdict":"new_model","recommendation":"Keep xsl_12's competitive, interference-limited architecture, the load-dependent encoding, and the contiguity boost. Fix its handling of repetition in low-ambiguity, high-repetition regimes. (1) Remove or sign-flip the negative kappa exposure term. Replace it with a term that depends on how consistent a word's evidence is, so that a word whose top object has co-occurred on nearly every exposure gets an increasing, not decreasing, gain: expo = (1 + cnt*consistency)^kappa_pos, with kappa_pos >= 0. Keep diminishing returns only for words whose evidence is conflicting. (2) Make retrieval-induced interference on absent partners proportional to competing mass. This stops it from eroding a well-supported committed pair when the other word and object on the trial are fillers or unambiguous. Gate this with commit, so deff is scaled by (1 - commit*ratio) for the committed object's link. (3) Let commitment exclusion only redistribute credit away from claimed objects and never reduce the committed pair's own gain. Check the changes against 221/222, where the object-count effect must be kept, and against filt*_9L, 215, 220 and 225, where accuracy should rise by about 0.1 with repetition."}

## Usage

```json
{
  "input_tokens": 7084,
  "output_tokens": 1009,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
