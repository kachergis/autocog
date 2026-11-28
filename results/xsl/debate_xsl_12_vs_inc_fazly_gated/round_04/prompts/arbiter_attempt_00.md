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
### THEORY 1: xsl_14 — held-out NLL 16880.2, held-out wSSE 0.4002
Fitted parameters: d=0.1635, lam=5.999, Mmax=9.733, fl=0.004404, b=1.194, s=0.2981, r=2.114, ch_dec=0.4588, p=1.07, rho=0.2295, th=0.7934, kc=17.16, xi=1, kappa=-0.8956, kpos=0.195, lapse=0.001, ability_sd=1.436

**Name:** Competition-scaled uncommitted returns with pair-count commitment gain

**Description:** The core is kept from xsl_13: competitive, load-limited learning with retrieval-induced interference, a fading contiguity trace, and commitment-gated exclusion with redistributed credit. What changes is how repetition is credited. (1) The diminishing returns for an uncommitted word are no longer a blanket function of its exposure count. They scale with how ambiguous that word's association row currently is (1 - 1/effective number of candidate objects, from the inverse Simpson index). A word in a sparse, consistent design keeps near-linear returns from repetition. A word spread across many candidates, as in dense 4x4 designs, is still throttled and stays near floor. (2) Once a word has committed, its gain grows with the number of consistent co-occurrences of that specific word-object pair, (1+n_pair)^kpos. kpos can be large, so late repetitions keep lifting a committed pair instead of saturating with total word exposure.

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
        # ambiguity of each present word's association row
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
        M[words, :] *= (1.0 - deff)
        M[:, objs] *= (1.0 - deff)
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
- proposed by xsl_14 on 223: human 0.3124 (SE 0.0293); xsl_14 0.3379 (0.9 SE off), inc_fazly 0.4354 (4.2 SE off); winner: xsl_14
  claim: In the dense 4x4 condition (223), xsl_14 should match human accuracy (about 0.31) better than inc_fazly, which overpredicts at about 0.43. xsl_14 has competitive, load-limited learning with an encoding cost for many objects per trial, and ambiguity-scaled diminishing returns that keep a word spread over many candidate objects near floor. Fazly's normalised alignment keeps accumulating association from every co-occurrence, however many candidates are present, so it is too insensitive to the extra ambiguity of dense trials.
- proposed by inc_fazly on 2_x8_39_4x4, 3_x8_369_4x4, temp_cont_2olap2tr_contr, 218: human 0.4087 (SE 0.0268); xsl_14 0.3570 (1.9 SE off), inc_fazly 0.4199 (0.4 SE off); winner: inc_fazly
  claim: inc_fazly should match people better. In these four conditions (three 4x4 designs and 3x4 6/8 with an 18-AFC test) human accuracy is about 0.41. inc_fazly predicts about 0.42 and xsl_14 about 0.36. In inc_fazly, alignment is proportional to current association strength and accumulates additively, so repeated word-object co-occurrences gradually separate the correct referent from the foils, even in dense 4x4 contexts. xsl_14 has load-limited, competition-throttled learning with decay, and in dense designs it keeps uncommitted words near floor. That makes it under-predict accuracy here.

## EARLIER ROUNDS
- proposed by xsl_12 on 221, 222: human 0.2014 (SE 0.0497); xsl_12 0.1936 (0.2 SE off), inc_fazly 0.0525 (3.0 SE off); winner: xsl_12
- proposed by inc_fazly on filt3E_9L, filt6E_9L, filt9E_9L: human 0.5887 (SE 0.0222); xsl_12 0.4911 (4.4 SE off), inc_fazly 0.5941 (0.2 SE off); winner: inc_fazly
- proposed by xsl_13 on 216, 219: human 0.5633 (SE 0.0431); xsl_13 0.5527 (0.2 SE off), inc_fazly 0.4549 (2.5 SE off); winner: xsl_13
- proposed by inc_fazly on filt6E_6L, filt9E_6L, filt9E_9L, 2_x8_39_4x4, 3_x8_369_4x4: human 0.4936 (SE 0.0205); xsl_13 0.4237 (3.4 SE off), inc_fazly 0.4914 (0.1 SE off); winner: inc_fazly
- proposed by xsl_13 on 207: human 0.1026 (SE 0.0096); xsl_13 0.1498 (4.9 SE off), inc_fazly 0.2562 (16.1 SE off); winner: xsl_13
- proposed by inc_fazly on filt0E_3L, filt3E_3L, filt6E_3L, filt9E_3L, filt0E_9L, filt3E_9L, filt6E_9L, filt9E_9L: human 0.1734 (SE 0.0258); xsl_13 0.0425 (5.1 SE off), inc_fazly 0.1450 (1.1 SE off); winner: inc_fazly
- proposed by xsl_14 on 225, 207: human 0.6840 (SE 0.0549); xsl_14 0.4836 (3.7 SE off), inc_fazly 0.3038 (6.9 SE off); winner: xsl_14
- proposed by inc_fazly on filt6E_9L, filt9E_9L, filt6E_3L, filt9E_3L: human 0.1766 (SE 0.0377); xsl_14 0.0414 (3.6 SE off), inc_fazly 0.1231 (1.4 SE off); winner: inc_fazly

## CONDITIONS
| label | condition | words x objects | trials | per trial | test | n_subj | human acc | xsl_14 acc | inc_fazly acc | held-out NLL xsl_14 minus inc_fazly |
|---|---|---|---|---|---|---|---|---|---|---|
| 201 | 3x4 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.191 | 0.347 | 0.424 | -5.2 |
| 202 | 3x4 1/.5 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.224 | 0.355 | 0.423 | -7.1 |
| 203 | 3x4 1/.66 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.211 | 0.349 | 0.422 | -3.6 |
| 204 | 3x4 +6o | 18x24 | 36 | 3.0x4.0 | 24-AFC | 20 | 0.272 | 0.337 | 0.410 | -2.3 |
| 205 | 2x4 | 18x18 | 54 | 2.0x4.0 | 18-AFC | 33 | 0.303 | 0.348 | 0.386 | -2.7 |
| 206 | 3x3 +1w/o | 18x18 | 54 | 3.0x3.0 | 18-AFC | 39 | 0.167 | 0.295 | 0.390 | -9.7 |
| 207 | 4x4 +2w/o | 18x18 | 54 | 4.0x4.0 | 18-AFC | 39 | 0.103 | 0.140 | 0.256 | -9.8 |
| 211 | 1x3 6/18 4AFC | 18x18 | 108 | 1.0x3.0 | 4-AFC | 43 | 0.674 | 0.780 | 0.701 | -13.8 |
| 212 | 2x3 6/9 4AFC | 18x18 | 54 | 2.0x3.0 | 4-AFC | 38 | 0.686 | 0.820 | 0.755 | -17.2 |
| 213 | 2x4 6/12 4AFC | 18x18 | 54 | 2.0x4.0 | 4-AFC | 31 | 0.624 | 0.673 | 0.711 | -14.7 |
| 214 | 3x4 6/8 4AFC | 18x18 | 36 | 3.0x4.0 | 4-AFC | 36 | 0.688 | 0.677 | 0.712 | -9.2 |
| 215 | 1x3 | 18x18 | 108 | 1.0x3.0 | 18-AFC | 23 | 0.744 | 0.504 | 0.382 | -14.5 |
| 216 | 2x3 | 18x18 | 54 | 2.0x3.0 | 18-AFC | 23 | 0.580 | 0.560 | 0.454 | -2.3 |
| 217 | 2x4 6/12 | 18x18 | 54 | 2.0x4.0 | 18-AFC | 14 | 0.325 | 0.375 | 0.386 | +1.7 |
| 218 | 3x4 6/8 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 13 | 0.419 | 0.376 | 0.419 | -0.6 |
| 219 | 2x3 6/9 | 18x18 | 54 | 2.0x3.0 | 18-AFC | 32 | 0.547 | 0.545 | 0.456 | -2.7 |
| 220 | 3x4 9/12 | 18x18 | 54 | 3.0x4.0 | 18-AFC | 33 | 0.685 | 0.493 | 0.479 | +2.9 |
| 221 | 1x4 | 18x18 | 108 | 1.0x4.0 | 18-AFC | 40 | 0.190 | 0.292 | 0.325 | -2.4 |
| 222 | 1x3 | 18x18 | 108 | 1.0x3.0 | 18-AFC | 40 | 0.392 | 0.479 | 0.377 | +3.3 |
| 223 | 4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 77 | 0.312 | 0.338 | 0.435 | -4.8 |
| 224 | 3x3 | 18x18 | 36 | 3.0x3.0 | 18-AFC | 36 | 0.426 | 0.491 | 0.486 | -0.3 |
| 225 | 2x2 | 18x18 | 54 | 2.0x2.0 | 18-AFC | 19 | 0.787 | 0.624 | 0.560 | -4.6 |
| 301 | 2x2 5x 24AFC | 24x24 | 60 | 2.0x2.0 | 24-AFC | 46 | 0.514 | 0.581 | 0.497 | +1.9 |
| filt0E_3L | filt0E_3L | 12x12 | 18 | 2.0x2.0 | 11-AFC | 31 | 0.372 | 0.504 | 0.513 | -0.6 |
| filt3E_3L | filt3E_3L | 12x12 | 27 | 2.0x2.0 | 11-AFC | 30 | 0.449 | 0.461 | 0.470 | -30.3 |
| filt6E_3L | filt6E_3L | 12x12 | 36 | 2.0x2.0 | 11-AFC | 30 | 0.424 | 0.454 | 0.459 | -30.3 |
| filt9E_3L | filt9E_3L | 12x12 | 45 | 2.0x2.0 | 11-AFC | 31 | 0.414 | 0.454 | 0.454 | -19.7 |
| filt0E_6L | filt0E_6L | 12x12 | 36 | 2.0x2.0 | 11-AFC | 23 | 0.487 | 0.554 | 0.629 | -2.7 |
| filt3E_6L | filt3E_6L | 12x12 | 45 | 2.0x2.0 | 11-AFC | 23 | 0.482 | 0.482 | 0.565 | +6.1 |
| filt6E_6L | filt6E_6L | 12x12 | 54 | 2.0x2.0 | 11-AFC | 23 | 0.553 | 0.473 | 0.538 | -8.1 |
| filt9E_6L | filt9E_6L | 12x12 | 63 | 2.0x2.0 | 11-AFC | 23 | 0.513 | 0.472 | 0.525 | +2.3 |
| filt0E_9L | filt0E_9L | 12x12 | 54 | 2.0x2.0 | 11-AFC | 31 | 0.586 | 0.577 | 0.694 | -2.8 |
| filt3E_9L | filt3E_9L | 12x12 | 63 | 2.0x2.0 | 11-AFC | 31 | 0.575 | 0.505 | 0.624 | -23.5 |
| filt6E_9L | filt6E_9L | 12x12 | 72 | 2.0x2.0 | 11-AFC | 31 | 0.590 | 0.495 | 0.589 | -3.1 |
| filt9E_9L | filt9E_9L | 12x12 | 81 | 2.0x2.0 | 11-AFC | 31 | 0.601 | 0.495 | 0.570 | +3.0 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.411 | 0.354 | 0.413 | +4.7 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.391 | 0.344 | 0.411 | +4.1 |
| freq369-3x3loCD | freq369-3x3loCD | 18x18 | 36 | 3.0x3.0 | 18-AFC | 34 | 0.485 | 0.430 | 0.428 | -16.9 |
| freq369-3x3hiCD | freq369-3x3hiCD | 18x18 | 36 | 3.0x3.0 | 18-AFC | 104 | 0.555 | 0.516 | 0.483 | -1.0 |
| freq369_36mx | freq369_36mx | 18x18 | 36 | 3.0x3.0 | 18-AFC | 66 | 0.454 | 0.467 | 0.456 | -16.1 |
| freq369_39mx | freq369_39mx | 18x18 | 36 | 3.0x3.0 | 18-AFC | 40 | 0.622 | 0.457 | 0.451 | -2.1 |
| cont_div6-12 | cont_div6-12 | 18x18 | 36 | 3.0x3.0 | 18-AFC | 62 | 0.435 | 0.479 | 0.481 | -3.7 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.489 | 0.417 | 0.435 | -1.0 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 18x18 | 27 | 4.0x4.0 | 18-AFC | 32 | 0.405 | 0.417 | 0.435 | +0.4 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.317 | 0.366 | 0.433 | -7.3 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.280 | 0.357 | 0.436 | -6.1 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.414 | 0.354 | 0.437 | -1.0 |

## Response

{"interpretation":"xsl_14 is clearly ahead on held-out fit (NLL 16880 vs 17154; wSSE 0.40 vs 0.50). It beats inc_fazly in most conditions, especially the sparse and low-accuracy ones: 207 and 223 in this round, and 221/222, 215, 225 earlier. Its failures are systematic. In several designs it under-predicts and flattens: the 4x4 conditions with higher human accuracy (2_x8_39_4x4, 3_x8_369_4x4, temp_cont_2olap2tr_contr, ~0.36 vs human ~0.41), the filt*E_9L and filt*E_6L conditions (~0.47-0.50 vs ~0.55-0.60), and 220 and freq369_39mx (~0.45-0.49 vs 0.62-0.69). In all of these, human accuracy keeps rising with repetition or with structure such as contiguity or frequency. xsl_14 saturates near 0.47-0.50 because ambiguity-scaled throttling keeps dense or partly resolved words near floor, and because commitment-gated exclusion stalls once words are only partly learned. inc_fazly captures the slow additive separation of the correct referent with repetition, but it overpredicts the plain dense and hard designs (223, 207, 201-206 at about 0.42 vs human 0.2-0.3) and cannot show the sharp gains that people get from clear structure (215, 225). The 223 query shows xsl_14's load and encoding cost is right: the 4x4 floor is matched within 0.9 SE. The inc_fazly query shows the cost of the ambiguity throttle: when 4x4 designs contain some disambiguating structure, such as overlap or repeated pairs, xsl_14 does not exploit it gradually. The failure is in the mechanism of uncommitted accumulation, not in the central claim of competitive, load-limited learning. So xsl_14 should be revised, not replaced.","target":1,"verdict":"new_model","recommendation":"Keep the competitive, load-limited core, the contiguity trace and the commitment gating. Replace the hard commit-gated, ambiguity-throttled exponent (kappa*amb*(1-commit)) with a soft, evidence-proportional accumulation channel that is always on, as in Fazly. Specifically, add to each trial's increment a term proportional to the current normalized association, rsb = M/(rowsum+c). This gives a rich-get-richer alignment, roughly lam2*enc*(rsb+eps)/(sum over present objects of (rsb+eps)). It should not be multiplied by (1-xi*cl) exclusion or by the (1+cnt)^(kappa*amb) discount, so partly learned words keep gaining gradually from repeated pairs. Mix it with the existing sharp commit-gated path using one new weight parameter. Make the throttle depend on the within-trial competition margin, meaning how much the pair's alignment exceeds the alternatives, rather than on the word's global row entropy. Then dense 4x4 designs with no consistent pair stay near floor (223 and 207 preserved), while designs with repeated or overlapping pairs (2_x8_39_4x4, 3_x8_369_4x4, temp_cont_2olap2tr, filt*E_9L, 220, freq369_39mx) climb toward 0.4-0.6 instead of plateauing near 0.35-0.50. Refit lam, kappa and kpos after the change."}

## Usage

```json
{
  "input_tokens": 8140,
  "output_tokens": 1171,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
