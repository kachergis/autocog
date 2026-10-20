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
- proposed by xsl_14 on 225, 207: human 0.6840 (SE 0.0549); xsl_14 0.4836 (3.7 SE off), inc_fazly 0.3038 (6.9 SE off); winner: xsl_14
  claim: Compare the sparse 2x2 design (225) with the dense 4x4 design that includes words without referents (207). Humans show a very large advantage for the sparse design, about 0.68. xsl_14 predicts a large gap, about 0.48. inc_fazly predicts only about 0.30, because its normalised alignment lets dense designs accumulate too much cross-situational evidence (207 predicted at 0.256 against a human 0.103) and gives sparse designs too little (225 predicted at 0.56 against a human 0.79). xsl_14's competition and load-limited learning throttles words with many candidate objects, so dense designs stay near floor. Its uncommitted returns scale with row ambiguity, and committed pairs keep gaining from repeated co-occurrence, so sparse designs profit strongly. This should put xsl_14 clearly closer to the human gap than inc_fazly.
- proposed by inc_fazly on filt6E_9L, filt9E_9L, filt6E_3L, filt9E_3L: human 0.1766 (SE 0.0377); xsl_14 0.0414 (3.6 SE off), inc_fazly 0.1231 (1.4 SE off); winner: inc_fazly
  claim: inc_fazly should match people better. This is the benefit of 9 versus 3 late-phase repetitions in the 12x12 2x2 filtering designs, with 6 or 9 early filler trials. Humans gain about 0.18 in accuracy. inc_fazly's incremental, normalised alignment keeps accumulating association with each repetition, so it predicts a sizeable gain of roughly 0.12. xsl_14's returns saturate once a word is committed, and its load-limited interference leaves the 9L conditions near 0.50, so it predicts a gain of only about 0.04. That is well short of the human value.

## EARLIER ROUNDS
- proposed by xsl_12 on 221, 222: human 0.2014 (SE 0.0497); xsl_12 0.1936 (0.2 SE off), inc_fazly 0.0525 (3.0 SE off); winner: xsl_12
- proposed by inc_fazly on filt3E_9L, filt6E_9L, filt9E_9L: human 0.5887 (SE 0.0222); xsl_12 0.4911 (4.4 SE off), inc_fazly 0.5941 (0.2 SE off); winner: inc_fazly
- proposed by xsl_13 on 216, 219: human 0.5633 (SE 0.0431); xsl_13 0.5527 (0.2 SE off), inc_fazly 0.4549 (2.5 SE off); winner: xsl_13
- proposed by inc_fazly on filt6E_6L, filt9E_6L, filt9E_9L, 2_x8_39_4x4, 3_x8_369_4x4: human 0.4936 (SE 0.0205); xsl_13 0.4237 (3.4 SE off), inc_fazly 0.4914 (0.1 SE off); winner: inc_fazly
- proposed by xsl_13 on 207: human 0.1026 (SE 0.0096); xsl_13 0.1498 (4.9 SE off), inc_fazly 0.2562 (16.1 SE off); winner: xsl_13
- proposed by inc_fazly on filt0E_3L, filt3E_3L, filt6E_3L, filt9E_3L, filt0E_9L, filt3E_9L, filt6E_9L, filt9E_9L: human 0.1734 (SE 0.0258); xsl_13 0.0425 (5.1 SE off), inc_fazly 0.1450 (1.1 SE off); winner: inc_fazly

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

{"interpretation":"Theory 1 (xsl_14) wins the round on the sparse-versus-dense contrast (225 vs 207). It gets the direction and rough size right, though it is still 3.7 SE short (0.48 vs 0.68). Its competition and load limits correctly keep dense designs near floor. It undershoots sparse designs: 225 comes out at 0.62 against a human 0.79, and 1x3 (215) at 0.50 against 0.74. It also undershoots 220 (0.49 vs 0.69), the 9/12 condition. The 3.6 SE miss on the filtering benefit shows the same weakness. Humans gain about 0.18 from 9 versus 3 late repetitions, but xsl_14 gains only 0.04. Its 9L filtering conditions sit near 0.50 against a human 0.58–0.60. So the mechanism throttles late repetition: committed pairs saturate, and interference from the filler or early trials caps them. Repeated co-occurrence should keep paying off in sparse, consistent designs. Theory 2 (inc_fazly) is the opposite. It captures cumulative repetition gains and the filtering results, but it fails on dense and ambiguous designs. It overpredicts 207 (0.26 vs 0.10) and 206, and gives roughly 0.42 on the 3x4 designs where humans are near 0.2. Its held-out NLL is also worse (17153.6 vs 16880.2), and xsl_14 has the lower NLL in most conditions. Theory 1 is the better base. Its fault is a fixable mechanism: weak accumulation for committed words in the late phase. This calls for a mechanism change, not a new theory.","target":1,"verdict":"new_model","recommendation":"Keep the competitive, load-limited, commitment-gated core, but change how committed pairs accumulate and how they are protected. (1) Make late gain for committed pairs non-saturating and stronger. Raise the kpos cap above 1, or replace (1+n_pair)^kpos with a gain that scales with the word's evidence for that object, such as lam*enc*(1+n_pair)^kpos*inc. The saturation term sat = 1 - blk/Mmax should then be made soft or Mmax raised for committed pairs, so a word with a clear referent keeps growing across 9 versus 3 late repetitions. (2) Protect committed associations from decay. Apply the deff multiplicative interference to committed pair cells only weakly: scale it by (1 - commit*protect), with protect fitted. Early filler trials and other late trials then stop eroding the committed row, which is what holds the 9L filtering conditions near 0.50. (3) Keep the ambiguity-scaled throttling for uncommitted words. This preserves the near-floor result in dense designs (207, 206, 3x4) while sparse designs (225, 215, 220) rise toward 0.7–0.8."}

## Usage

```json
{
  "input_tokens": 8035,
  "output_tokens": 973,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
