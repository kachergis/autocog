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


## THIS ROUND'S QUERIES
- proposed by xsl_13 on 216, 219: human 0.5633 (SE 0.0431); xsl_13 0.5527 (0.2 SE off), inc_fazly 0.4549 (2.5 SE off); winner: xsl_13
  claim: On the 2-words x 3-objects designs with an 18-AFC test (216 and 219), people average about 0.56. xsl_13 predicts about 0.553 and inc_fazly about 0.455, so inc_fazly should be well outside the 2 SE threshold. xsl_13's commitment-gated, increasing-returns encoding lets repetition pay off in low-ambiguity trials. Consistent word-object pairs also get shielded from interference, and claimed objects are excluded from other words. Together these let learning build up in small, easy designs. inc_fazly's incremental alignment normalises across co-present objects and has no commitment or exclusion mechanism, so it consolidates consistent pairs too slowly and underpredicts accuracy in these conditions.
- proposed by inc_fazly on filt6E_6L, filt9E_6L, filt9E_9L, 2_x8_39_4x4, 3_x8_369_4x4: human 0.4936 (SE 0.0205); xsl_13 0.4237 (3.4 SE off), inc_fazly 0.4914 (0.1 SE off); winner: inc_fazly
  claim: inc_fazly should match people better. Averaged over these five conditions, humans score about 0.49, inc_fazly about 0.49 and xsl_13 about 0.42. The five are the high-exposure filtered 2x2 conditions (6E_6L, 9E_6L, 9E_9L) and the two 4x4 mixed-frequency conditions. In all of them, repeated consistent word-object co-occurrences keep adding to the probability-normalised association. Fazly's alignment is competitive, so each word's meaning distribution sharpens steadily as evidence accumulates, and filler trials do not erode it. xsl_13 adds interference and exclusion penalties and a diminishing-returns exposure term (negative kappa) for words that have not committed. It therefore under-learns these conditions by about 0.06 to 0.1. Fazly's predictions sit within about 0.01 of the human values on the individual conditions.

## EARLIER ROUNDS
- proposed by xsl_12 on 221, 222: human 0.2014 (SE 0.0497); xsl_12 0.1936 (0.2 SE off), inc_fazly 0.0525 (3.0 SE off); winner: xsl_12
- proposed by inc_fazly on filt3E_9L, filt6E_9L, filt9E_9L: human 0.5887 (SE 0.0222); xsl_12 0.4911 (4.4 SE off), inc_fazly 0.5941 (0.2 SE off); winner: inc_fazly

## CONDITIONS
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

## Response

{"interpretation":"The round splits. xsl_13 wins on 216/219 (2x3 designs, 18-AFC): 0.553 vs human 0.563, while inc_fazly sits 2.5 SE low. inc_fazly wins on the high-exposure filtered 2x2 conditions and the 4x4 mixed-frequency conditions: 0.491 vs human 0.494, while xsl_13 sits 3.4 SE low at 0.424. Earlier rounds show the same pattern. xsl_12 lost on filt*_9L by 4.4 SE, and the lineage's one clear win was on the sparse 221/222 conditions. The recurring xsl failure is under-learning where repetition of consistent pairs should keep paying off: filt6E/9E_6L, filt*_9L (0.47-0.50 vs human 0.51-0.60), 220 (0.48 vs 0.69), and 225 (0.62 vs 0.79). In these designs filler trials and extra exposure do not raise accuracy at all, and xsl's accuracy stays flat or falls as exposure grows. That points to the mechanisms that cap or erode learning with exposure: negative kappa (diminishing returns for uncommitted words), per-trial interference erosion (deff), and the commitment threshold. In 2x2 designs a word is often not yet committed (th=0.76), so it is stuck in the diminishing-returns regime and filler erosion eats its gains. inc_fazly, in contrast, accumulates without decay and so misses the cases where people use exclusion or mutual exclusivity: 215 (0.38 vs 0.74), 225, 220, 212, 2x3. It also over-predicts the hard, high-ambiguity conditions: 201-207 are all 0.1-0.15 too high. Held-out NLL favors xsl_13 (16887 vs 17154) and the per-condition NLL differences are mostly negative, so xsl_13 is the better base. Revise theory 1, keeping its core of competition, exclusion and a contiguity trace, and fix the exposure-dependence of its learning rate and erosion.","target":1,"verdict":"new_model","recommendation":"Keep xsl_13's central claim and replace the commitment-binary exposure and interference machinery with a cumulative-evidence gate. (1) Remove the negative-kappa diminishing returns, which are a main cause of the flat or declining accuracy with exposure in filt*_6L/9L and the 4x4 mixed-frequency conditions. Set the exposure exponent to a smooth non-negative function of the word's evidence concentration, for example e = kpos*conc, where conc = max_j rs_j (or 1 - normalized entropy of the word's row), instead of the sigmoid commit at th=0.76. Even partially-concentrated words should then get increasing returns. (2) Make interference erosion shielding graded by the same concentration: M[w,o] erodes by deff*(1 - rs_wo/mx)^k, not only for committed words. Also make deff for absent-partner erosion scale down for pairs with a long trace, so fillers cannot wash out consistent pairs in 2x2 designs. (3) Keep the exclusion term (xi) and the share^b test discount, since these produce the wins on 216/219/215/225, but apply the discount only after a word's own max row share exceeds a small threshold, so it does not suppress weak words in the high-ambiguity conditions 201-207, where xsl_13 is still too high. Refit d, kpos, and th (or drop th) after these changes, and check that filt6E_6L, filt9E_9L, 2_x8_39_4x4 and 3_x8_369_4x4 rise to about 0.5 without lowering 216/219."}

## Usage

```json
{
  "input_tokens": 7573,
  "output_tokens": 1246,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
