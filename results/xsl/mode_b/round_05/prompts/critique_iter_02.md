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
`parameters` maps each name to a string interval "[lo, hi]" (fitted to the data) or a bare number (held fixed). Every free parameter costs log(N) BIC (see below), so add one only if it buys a real improvement in fit. Every theory automatically gets one extra fitted parameter, `lapse`: the rate of pure guessing (uniform over the choices) at test. Do not declare `lapse` yourself, and do not model random guessing inside `learn`.

## HOW THEORIES ARE SCORED
Each theory is FITTED to all 47 conditions at once (one parameter set for everyone) by differential evolution, minimising the negative log-likelihood (NLL) of every human test choice: multinomial over objects for n-AFC tests (so WHICH wrong object people choose matters, not just accuracy), binomial correct/incorrect for m-AFC tests. The score is BIC = 2*NLL + k*ln(N) with k free parameters (yours plus `lapse`) and N = 31266 choices. Lower is better. The n_subj-weighted SSE on per-word accuracy (the xsl-bakeoff objective) is shown for reference but is not the score.

## OUTPUT
Return one JSON object with these fields:
- interpretation: Where the candidate fits the human data well or badly, and which mechanism is responsible.
- verdict: 'continue' to accept the candidate as this loop's answer, or 'regenerate' to ask for a revision.
- rationale: If 'regenerate': the concrete change to the mechanism or parameters to try next. If 'continue': why it is good enough.


## User Prompt

## CANDIDATE — BIC 32521.2, NLL 16219.2, wSSE 0.4959 — REJECTED (not better than this loop's best)
Fitted parameters: d=0.5804, q=1.224, lam=0.5702, fl=0.01957, b=0, r=5.129e-06, ch_dec=1.005, lapse=0.2495 — at bound: b

**Name:** Decaying-trace boosted interference learning (no saturation)

**Description:** Learning is competitive and interference-limited. Each pair gains the product of its word-side and object-side responsibilities, which gives soft mutual exclusivity and explaining away. Retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words, at a rate that is a power law of trial load (words x objects). Gain is unsaturated: evidence simply accumulates, and the interference process alone limits strength. Working memory holds a graded, decaying trace of recent co-occurrences instead of a single one-trial trace. A pair seen on the previous trial has trace 1, one seen two trials back has 0.4, and so on. A pair's credit is amplified by 1 + r*trace, so recently repeated pairs are boosted before interference erodes them. This is a general recency-of-repeat benefit, and it explains the temporal-contiguity and overlap designs. At test, an object already claimed by other words is a poorer referent: association is multiplied by the tested word's share of that object's total association, raised to a power b, plus a retrieval floor. Choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.05, 6]
- fl: [0.0, 3.0]
- b: [0.0, 3.0]
- r: [0.0, 30]
- ch_dec: [0.3, 6]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    d = params['d']
    q = params['q']
    lam = params['lam']
    fl = params['fl']
    b = params['b']
    r = params['r']
    s = 4.0
    c = 0.5
    g = 0.5
    eps = 0.1
    rho = 0.4
    M = np.zeros((n_words, n_objects))
    trace = np.zeros((n_words, n_objects))
    for words, objs in trials:
        words = np.asarray(words)
        objs = np.asarray(objs)
        nw = len(words)
        no = len(objs)
        enc = 1.0 / (1.0 + (no / c) ** g)
        deff = min(0.95, d * (nw * no / 12.0) ** q)
        ix = np.ix_(words, objs)
        blk = M[ix].copy()
        M[words, :] *= (1.0 - deff)
        M[:, objs] *= (1.0 - deff)
        st = (blk + eps) ** s
        rw = st / st.sum(1, keepdims=True)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        bonus = 1.0 + r * trace[ix]
        M[ix] = blk + lam * enc * inc * bonus
        trace = trace * rho
        trace[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc | NLL | NLL minus xsl_05's |
|---|---|---|---|---|---|
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.440 | 484.0 | +83.6 |
| filt6E_9L | filt6E_9L | 0.590 | 0.398 | 173.5 | +83.2 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.505 | 627.7 | +79.4 |
| filt3E_9L | filt3E_9L | 0.575 | 0.395 | 153.1 | +73.3 |
| freq369_36mx | freq369_36mx | 0.454 | 0.456 | 656.0 | +68.1 |
| 223 | 4x4 | 0.312 | 0.416 | 698.4 | +55.5 |
| filt6E_6L | filt6E_6L | 0.553 | 0.384 | 112.7 | +47.0 |
| filt3E_6L | filt3E_6L | 0.482 | 0.378 | 105.2 | +41.6 |
| filt9E_9L | filt9E_9L | 0.601 | 0.400 | 166.5 | +39.0 |
| 215 | 1x3 | 0.744 | 0.443 | 316.1 | +33.1 |
| 225 | 2x2 | 0.787 | 0.563 | 226.7 | +30.0 |
| freq369_39mx | freq369_39mx | 0.622 | 0.452 | 470.6 | +29.0 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.419 | 497.3 | +25.7 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.348 | 460.7 | +24.3 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.416 | 476.9 | +23.2 |
| 202 | 3x4 1/.5 | 0.224 | 0.341 | 448.6 | +22.0 |
| filt9E_6L | filt9E_6L | 0.513 | 0.387 | 99.3 | +21.5 |
| 220 | 3x4 9/12 | 0.685 | 0.418 | 414.0 | +18.6 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.366 | 449.8 | +15.1 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.405 | 399.4 | +13.6 |
| 219 | 2x3 6/9 | 0.547 | 0.474 | 394.0 | +13.2 |
| 205 | 2x4 | 0.303 | 0.301 | 482.1 | +11.3 |
| 218 | 3x4 6/8 | 0.419 | 0.304 | 277.3 | +9.2 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.459 | 746.7 | +9.0 |
| 203 | 3x4 1/.66 | 0.211 | 0.310 | 449.8 | +7.7 |
| 224 | 3x3 | 0.426 | 0.486 | 463.5 | +5.7 |
| 216 | 2x3 | 0.580 | 0.468 | 316.3 | +5.1 |
| filt9E_3L | filt9E_3L | 0.414 | 0.352 | 75.0 | +4.8 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.705 | 58.4 | +4.7 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.405 | 433.3 | +2.3 |
| filt6E_3L | filt6E_3L | 0.424 | 0.346 | 75.7 | +2.2 |
| 201 | 3x4 | 0.191 | 0.306 | 452.9 | +0.7 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.734 | 57.1 | +0.4 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.632 | 47.1 | -0.1 |
| 217 | 2x4 6/12 | 0.325 | 0.298 | 305.5 | -2.1 |
| 222 | 1x3 | 0.392 | 0.445 | 481.6 | -2.3 |
| 204 | 3x4 +6o | 0.272 | 0.399 | 458.7 | -3.0 |
| filt0E_9L | filt0E_9L | 0.586 | 0.545 | 66.7 | -3.2 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.419 | 424.0 | -3.2 |
| filt3E_3L | filt3E_3L | 0.449 | 0.334 | 92.3 | -11.4 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.644 | 51.3 | -12.4 |
| filt0E_6L | filt0E_6L | 0.487 | 0.480 | 53.7 | -14.0 |
| 207 | 4x4 +2w/o | 0.103 | 0.062 | 565.0 | -15.1 |
| 221 | 1x4 | 0.190 | 0.305 | 559.8 | -17.3 |
| filt0E_3L | filt0E_3L | 0.372 | 0.363 | 64.9 | -20.7 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.480 | 781.7 | -33.5 |
| 206 | 3x3 +1w/o | 0.167 | 0.275 | 548.3 | -37.4 |

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
| rank | label | name | k | BIC | NLL | wSSE | fitted parameters |
|---|---|---|---|---|---|---|---|
| 1 | xsl_05 | Consecutive-repetition boosted interference learning | 9 | 31077.1 | 15492.0 | 0.4177 | d=0.2088, q=0.3556, lam=0.06911, Mmax=3.588, fl=0.0007805, b=0.1948, r=1.482, ch_dec=1.332, lapse=0.2629 |
| 2 | xsl_04 | Interference learning with test-time object-claim competition | 8 | 31165.9 | 15541.6 | 0.4280 | d=0.229, q=0.4687, lam=0.07155, Mmax=1.767, fl=0.0007025, b=0.1984, ch_dec=1.325, lapse=0.2755 |
| 3 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 6 | 31863.2 | 15900.6 | 0.5101 | d=0.2836, q=0.03972, lam=0.381, Mmax=9.816, ch_dec=0.6064, lapse=0.4524 |
| 4 | xsl_02 | Two-stage verify-and-consolidate with object-driven load | 6 | 32002.0 | 15969.9 | 0.4980 | a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g |
| 5 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 32045.4 | 15996.8 | 0.5529 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 6 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 32391.7 | 16175.2 | 0.5668 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 7 | baseline | Co-occurrence counting | 1 | 39486.6 | 19738.1 | 1.3660 | lapse=0.3881 |

## LOOP HISTORY
- iter 0: Consolidation-protected boosted interference learning — BIC 31093.9 — ACCEPTED
  critic: Drop h and the protection term, which returns to xsl_05's 9 parameters. Then fix how load is computed and how strongly it hits encoding. (a) Split load into separate word and object exponents: deff = min(0.95, d * (nw/3)^q_w * (no/3)^q_o), with q_o free in [0,3] and q_w free in [0,2]. Alternatively keep a single q but make load depend mainly on no. This should push 1x4 and 3x4 down toward 0.19 without hurting 1x3. (b) Make the encoding exponent g in enc = 1/(1+(no/c)^g) a fitted parameter instead of the fixed 0.5, so encoding falls steeply with objects per trial. (c) To help high-frequency words, let the sharpness exponent s (currently fixed at 4) be fitted, or make sat depend on the word's own evidence, so that words reaching a few co-occurrences lock in. Fit the added parameters only if the NLL gain is more than about 5 per parameter, since each costs 10.3 in BIC.
- iter 1: Object-driven interference with steep encoding loss — BIC 31729.3 — REJECTED
  critic: Revert to the xsl_05 backbone and make one small change, instead of the steep-encoding variant. (a) Remove the enc factor (g, c); set it to 1, or a mild fixed 1/sqrt(no/3) at most. Restore the modest bonus r (about 1-3, e.g. bound [0,4]) so the consecutive-repeat boost cannot dominate. This restores accumulation of evidence over repeated co-occurrences, which is what drives the rising 3L->9L accuracy in the filt conditions. (b) Keep only the split interference exponents, deff = min(0.95, d*(nw/3)^qw*(no/3)^qo), as the single new structure. Make qo free and qw free or tied to a fraction of qo. (c) To stop pruning from overwhelming a correct pair on low-load 2x2 trials, scale the pruning of a pair's row and column links by (1 - sat), where sat = M/Mmax. Strongly established pairs then resist interference, while weak spurious pairs still decay, and accuracy rises with trial count as in the filtering data. This replaces the g and r flexibility with one mechanism. Adopt it only if the NLL gain over xsl_05 exceeds roughly 5 per added parameter.

## Response

{"interpretation":"The candidate is rejected (BIC 32521 vs. best 31077, about 1440 worse). It removed the saturation cap Mmax and fixed the encoding and sharpness settings, and it fits worst where accumulation and saturation matter.\n\n1. The filtering conditions are badly underpredicted. The model gives ~0.40 where humans give 0.55-0.60 (filt6E_9L, filt3E_9L, filt6E_6L, filt9E_9L). Model accuracy is flat across 3L to 9L and across added exposure. Without saturation, the strong interference decay (d=0.58, q=1.2, lam=0.57) erodes repeated evidence as fast as it accumulates. Humans improve steadily with trial count and frequency, which the model does not reproduce. The 1x3 and 2x2 conditions (215, 225) are also underpredicted (0.44 vs 0.74, 0.56 vs 0.79), which points to the same problem: pruning is too strong at low load, and nothing protects established pairs.\n\n2. The high-load, low-accuracy conditions are overpredicted. Examples are 201, 202, 203, 204, 206, 221, 223 and the contiguity-control conditions (1olap2tr_contr 0.42 vs 0.28). The model's accuracy is too flat across load. The fixed enc factor (c=0.5, g=0.5) has only a weak effect and cannot drive accuracy down to 0.19-0.25.\n\n3. Frequency and contextual-diversity effects are underfit. The freq369 and cont_div conditions are among the worst NLL gaps (+68 to +84). Words at frequency 3, 6 and 9 should differ strongly, but the model cannot lock in high-frequency words. Accuracy is 0.44-0.46 across them, with no strong benefit at frequency 9.\n\n4. The mechanism parameters are degenerate. r=5e-6 means the decaying-trace boost is unused, so this is a plain xsl_05 variant without a working recency mechanism. b=0 at its bound means the test-time claim competition is also unused. fl=0.02 is a flat floor, and the large lapse (0.25) absorbs the misfit. Two of the seven free parameters contribute nothing, so they only cost BIC.\n\nThe main gain on the leaderboard comes from Mmax-style saturation and the competitive claim, and this candidate has neither.","verdict":"regenerate","rationale":"Revert to the xsl_05 backbone: restore the saturation cap Mmax (about [0.5,10]) and a modest consecutive-repeat boost, with r bounded to [0,4]. Delete the unused decaying trace (rho) and r=5e-6, and keep the claim exponent b only if it fits above 0.1. Then make one structural change: scale the pruning of a pair's row and column links by (1 - M/Mmax) for established pairs, so strongly learned pairs resist interference. Apply the load penalty deff through separate word and object exponents, with a stronger object exponent: deff = min(0.95, d*(nw/3)^qw*(no/3)^qo). That should pull 1x4, 3x4 and 4x4 down toward 0.1-0.3 while letting 1x3, 2x2 and the filt conditions grow with frequency. Cap the interference rate (d about 0.2-0.3, q about 0.4) instead of letting d=0.58 and q=1.2 wipe out accumulation. Keep the new parameters only if the NLL gain over xsl_05 exceeds about 5 per parameter."}

## Usage

```json
{
  "input_tokens": 9230,
  "output_tokens": 1209,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
