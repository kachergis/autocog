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
`parameters` maps each name to a string interval "[lo, hi]" (fitted to the data) or a bare number (held fixed). Every free parameter costs log(N) BIC (see below), so add one only if it buys a real improvement in fit. Every theory automatically gets one extra fitted parameter, `lapse`: the rate of pure guessing (uniform over the choices) at test. Do not declare `lapse` yourself, and do not model random guessing inside `learn`.

## HOW THEORIES ARE SCORED
Each theory is FITTED to all 47 conditions at once (one parameter set for everyone) by differential evolution, minimising the negative log-likelihood (NLL) of every human test choice: multinomial over objects for n-AFC tests (so WHICH wrong object people choose matters, not just accuracy), binomial correct/incorrect for m-AFC tests. The score is BIC = 2*NLL + k*ln(N) with k free parameters (yours plus `lapse`) and N = 31266 choices. Lower is better. The n_subj-weighted SSE on per-word accuracy (the xsl-bakeoff objective) is shown for reference but is not the score.

## OUTPUT
Return one JSON object with these fields:
- interpretation: Where the candidate fits the human data well or badly, and which mechanism is responsible.
- verdict: 'continue' to accept the candidate as this loop's answer, or 'regenerate' to ask for a revision.
- rationale: If 'regenerate': the concrete change to the mechanism or parameters to try next. If 'continue': why it is good enough.


## User Prompt

## CANDIDATE — BIC 31093.9, NLL 15495.2, wSSE 0.4175 — ACCEPTED
Fitted parameters: d=0.2258, q=0.4937, lam=0.06846, Mmax=9.398, fl=0.0007459, b=0.1973, r=0.8932, h=28.48, ch_dec=1.331, lapse=0.2648

**Name:** Consolidation-protected boosted interference learning

**Description:** Learning is competitive and interference-limited. Each pair gains the product of its word-side and object-side responsibilities (soft mutual exclusivity and explaining away), and the gain saturates. Retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words, at a rate that is a power law of trial load. A one-trial working-memory trace boosts pairs that repeat on consecutive trials (temporal contiguity). The new mechanism is consolidation: interference is not equally damaging to every link. Weak, tentative associations are fully exposed to it, while associations that have already been built up become progressively protected from it. Protection falls as 1/(1+(M/h)^2), so strength acts as a soft consolidation threshold. Words seen often enough to cross the threshold become durable and snowball, as with 9-repetition words and sparse 1-word trials. Words that never cross it keep being eroded under dense load. At test, an object claimed by other words is a poorer referent, with a share^b competition term plus a retrieval floor, and choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.05, 4]
- Mmax: [0.3, 10]
- fl: [0.0, 3.0]
- b: [0.0, 3.0]
- r: [0.0, 30]
- h: [0.05, 30]
- ch_dec: [0.3, 6]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    d = params['d']
    q = params['q']
    lam = params['lam']
    Mmax = params['Mmax']
    fl = params['fl']
    b = params['b']
    r = params['r']
    h = params['h']
    s = 4.0
    c = 0.5
    g = 0.5
    eps = 0.1
    M = np.zeros((n_words, n_objects))
    prev = np.zeros((n_words, n_objects))
    for words, objs in trials:
        words = np.asarray(words)
        objs = np.asarray(objs)
        nw = len(words)
        no = len(objs)
        enc = 1.0 / (1.0 + (no / c) ** g)
        deff = min(0.95, d * (nw * no / 12.0) ** q)
        ix = np.ix_(words, objs)
        blk = M[ix].copy()
        prot = 1.0 / (1.0 + (M / h) ** 2)
        M[words, :] *= (1.0 - deff * prot[words, :])
        M[:, objs] *= (1.0 - deff * prot[:, objs])
        st = (blk + eps) ** s
        rw = st / st.sum(1, keepdims=True)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        bonus = 1.0 + r * prev[ix]
        M[ix] = blk + lam * enc * inc * sat * bonus
        prev = np.zeros((n_words, n_objects))
        prev[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc | NLL | NLL minus xsl_05's |
|---|---|---|---|---|---|
| 221 | 1x4 | 0.190 | 0.307 | 582.8 | +5.7 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.421 | 390.7 | +4.9 |
| 220 | 3x4 9/12 | 0.685 | 0.437 | 397.6 | +2.3 |
| 222 | 1x3 | 0.392 | 0.509 | 486.1 | +2.2 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.586 | 65.3 | +1.6 |
| filt0E_3L | filt0E_3L | 0.372 | 0.494 | 87.1 | +1.5 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.476 | 549.7 | +1.4 |
| 218 | 3x4 6/8 | 0.419 | 0.344 | 269.1 | +1.0 |
| filt3E_9L | filt3E_9L | 0.575 | 0.566 | 80.5 | +0.8 |
| filt0E_6L | filt0E_6L | 0.487 | 0.602 | 68.5 | +0.8 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.585 | 47.9 | +0.7 |
| 203 | 3x4 1/.66 | 0.211 | 0.319 | 442.7 | +0.7 |
| filt3E_6L | filt3E_6L | 0.482 | 0.543 | 64.1 | +0.6 |
| 204 | 3x4 +6o | 0.272 | 0.340 | 462.3 | +0.6 |
| 223 | 4x4 | 0.312 | 0.308 | 643.4 | +0.5 |
| 224 | 3x3 | 0.426 | 0.475 | 458.3 | +0.5 |
| 202 | 3x4 1/.5 | 0.224 | 0.334 | 427.1 | +0.4 |
| filt0E_9L | filt0E_9L | 0.586 | 0.648 | 70.4 | +0.4 |
| 205 | 2x4 | 0.303 | 0.318 | 471.2 | +0.3 |
| filt3E_3L | filt3E_3L | 0.449 | 0.461 | 104.0 | +0.3 |
| 225 | 2x2 | 0.787 | 0.688 | 197.0 | +0.3 |
| 206 | 3x3 +1w/o | 0.167 | 0.294 | 585.9 | +0.2 |
| freq369_36mx | freq369_36mx | 0.454 | 0.457 | 587.9 | +0.1 |
| 201 | 3x4 | 0.191 | 0.318 | 452.2 | +0.0 |
| 217 | 2x4 6/12 | 0.325 | 0.342 | 307.6 | +0.0 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.343 | 453.6 | -0.0 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.673 | 53.7 | -0.1 |
| 219 | 2x3 6/9 | 0.547 | 0.523 | 380.6 | -0.2 |
| filt9E_6L | filt9E_6L | 0.513 | 0.415 | 77.6 | -0.3 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.427 | 400.0 | -0.4 |
| filt9E_3L | filt9E_3L | 0.414 | 0.375 | 69.8 | -0.4 |
| filt6E_3L | filt6E_3L | 0.424 | 0.404 | 73.1 | -0.4 |
| filt6E_6L | filt6E_6L | 0.553 | 0.456 | 65.3 | -0.5 |
| 216 | 2x3 | 0.580 | 0.541 | 310.7 | -0.5 |
| filt9E_9L | filt9E_9L | 0.601 | 0.428 | 127.0 | -0.5 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.693 | 56.1 | -0.6 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.331 | 471.0 | -0.6 |
| 207 | 4x4 +2w/o | 0.103 | 0.095 | 579.3 | -0.8 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.645 | 814.4 | -0.9 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.309 | 433.9 | -0.9 |
| filt6E_9L | filt6E_9L | 0.590 | 0.471 | 89.3 | -1.0 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.302 | 434.9 | -1.6 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.328 | 425.5 | -1.8 |
| freq369_39mx | freq369_39mx | 0.622 | 0.449 | 439.2 | -2.4 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.472 | 734.5 | -3.2 |
| 215 | 1x3 | 0.744 | 0.524 | 279.3 | -3.7 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.421 | 427.1 | -3.9 |

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
(first iteration)

## Response

```json
{
  "interpretation": "The candidate is not a real improvement. It adds h, a 10th parameter, but its NLL (15495.2) is slightly worse than xsl_05's (15492.0). Its BIC of 31093.9 is therefore 17 points worse than xsl_05's 31077.1. The consolidation mechanism buys nothing, and it is probably counterproductive. True pairs are never decayed, because both the word and object are present whenever they co-occur and `blk` is restored after the decay step. Protection from interference therefore only shields strong false links, such as persistent competitors. The remaining misfits are the same as xsl_05's. (1) Object count per trial matters more to humans than the model allows: 1x4 is 0.19 human vs 0.31 model, and 1x3 is 0.39 vs 0.51. Humans find 1x4 as hard as 3x4 (0.19), which suggests that objects per trial, not words x objects, drive difficulty. The model's load term (nw*no/12)^q cannot produce that. (2) High-frequency or low-diversity conditions are underpredicted: 220 (3x4 9/12) is 0.685 vs 0.437, freq369_39mx is 0.622 vs 0.449, filt9E_9L is 0.601 vs 0.428, and the 4x4 conditions are 0.41 vs 0.30. Words with many repetitions do not snowball enough. (3) Conditions 215 and 222 are identical on paper (1x3, 108 trials) but differ in human accuracy (0.744 vs 0.392). The model gives about 0.51 to both, and this between-study noise cannot be fixed.",
  "verdict": "regenerate",
  "rationale": "Drop h and the protection term, which returns to xsl_05's 9 parameters. Then fix how load is computed and how strongly it hits encoding. (a) Split load into separate word and object exponents: deff = min(0.95, d * (nw/3)^q_w * (no/3)^q_o), with q_o free in [0,3] and q_w free in [0,2]. Alternatively keep a single q but make load depend mainly on no. This should push 1x4 and 3x4 down toward 0.19 without hurting 1x3. (b) Make the encoding exponent g in enc = 1/(1+(no/c)^g) a fitted parameter instead of the fixed 0.5, so encoding falls steeply with objects per trial. (c) To help high-frequency words, let the sharpness exponent s (currently fixed at 4) be fitted, or make sat depend on the word's own evidence, so that words reaching a few co-occurrences lock in. Fit the added parameters only if the NLL gain is more than about 5 per parameter, since each costs 10.3 in BIC."
}
```

## Usage

```json
{
  "input_tokens": 8509,
  "output_tokens": 2549,
  "reasoning_tokens": 1621,
  "provider": "Claude Platform on AWS"
}
```
