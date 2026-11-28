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
`parameters` maps each name to a string interval "[lo, hi]" (fitted to the data) or a bare number (held fixed). Add a parameter only if it captures something that generalises across conditions (see scoring below). Every theory automatically gets one extra fitted parameter, `lapse`: the rate of pure guessing (uniform over the choices) at test. Do not declare `lapse` yourself, and do not model random guessing inside `learn`.

## HOW THEORIES ARE SCORED
Each theory is FITTED to all 47 conditions at once (one parameter set for everyone) by differential evolution, minimising the negative log-likelihood (NLL) of every human test choice: multinomial over objects for n-AFC tests (so WHICH wrong object people choose matters, not just accuracy), binomial correct/incorrect for m-AFC tests.

The SCORE is the held-out NLL from 5-fold cross-validation over conditions: the theory is fitted to four fifths of the conditions and must predict the human choices in the remaining fifth, for each fifth in turn (always the same split), and the five held-out NLLs are summed. Lower is better. Extra parameters that only fit the training conditions do not help here and usually hurt. Also shown, for reference only: BIC = 2*NLL + k*ln(N) of the full-corpus fit (k free parameters incl. `lapse`, N = 31266 choices) and the n_subj-weighted SSE on per-word accuracy (the xsl-bakeoff objective).

## OUTPUT
You have no tools and cannot run code: do not write tool calls, XML tags or `<invoke>` blocks. Reply with the JSON object only — no prose before or after it.
Return one JSON object with these fields:
- interpretation: Where the candidate fits the human data well or badly, and which mechanism is responsible.
- verdict: 'continue' to accept the candidate as this loop's answer, or 'regenerate' to ask for a revision.
- rationale: If 'regenerate': the concrete change to the mechanism or parameters to try next. If 'continue': why it is good enough.


## User Prompt

## CANDIDATE — held-out NLL 16421.2, BIC 32292.1, wSSE (full fit) 0.4542 — ACCEPTED
Fitted parameters: d=0.2905, q=0.76, lam=0.8416, Mmax=10, fl=0.2012, b=0, g=2.039, ch_dec=1.084, lapse=0.09592 — at bound: Mmax, b

**Name:** Object-ambiguity-gated interference learning with claim competition

**Description:** Learning is competitive and interference-limited. Each pair gains the product of its word-side and object-side responsibilities, with soft mutual exclusivity and explaining away. The gain saturates, and retrieval-induced interference prunes links to absent partners at a rate that is a power law of trial load. The new claim is that the efficiency of encoding falls as a power law of the number of candidate objects on the trial. This reflects referential ambiguity and limited visual attention. The exponent is fitted rather than fixed to a weak dependence. Adding a fourth object to a display therefore sharply cuts what is learned from that trial, as in the 1x4, 2x4 and 3x4 designs, while 2-object trials are encoded far more efficiently. At test, an object already claimed by other words is penalised through the tested word's share of that object's association, raised to a power b. A retrieval floor is added, and choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.005, 4]
- Mmax: [0.3, 10]
- fl: [0.0, 3.0]
- b: [0.0, 3.0]
- g: [0.0, 4.0]
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
    g = params['g']
    s = 4.0
    eps = 0.1
    M = np.zeros((n_words, n_objects))
    for words, objs in trials:
        words = np.asarray(words)
        objs = np.asarray(objs)
        nw = len(words)
        no = len(objs)
        enc = (3.0 / no) ** g
        deff = min(0.95, d * (nw * no / 12.0) ** q)
        ix = np.ix_(words, objs)
        blk = M[ix].copy()
        M[words, :] *= (1.0 - deff)
        M[:, objs] *= (1.0 - deff)
        st = (blk + eps) ** s
        rw = st / st.sum(1, keepdims=True)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        M[ix] = blk + lam * enc * inc * sat
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc (full fit) | held-out NLL | held-out NLL minus xsl_04's |
|---|---|---|---|---|---|
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.468 | 493.6 | +93.7 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.517 | 646.6 | +88.8 |
| 220 | 3x4 9/12 | 0.685 | 0.410 | 482.8 | +72.5 |
| 215 | 1x3 | 0.744 | 0.419 | 372.3 | +68.6 |
| freq369_36mx | freq369_36mx | 0.454 | 0.489 | 658.6 | +68.0 |
| 206 | 3x3 +1w/o | 0.167 | 0.395 | 641.6 | +46.1 |
| filt3E_6L | filt3E_6L | 0.482 | 0.406 | 109.6 | +44.5 |
| 223 | 4x4 | 0.312 | 0.373 | 690.5 | +43.0 |
| filt3E_9L | filt3E_9L | 0.575 | 0.418 | 144.2 | +41.4 |
| filt6E_9L | filt6E_9L | 0.590 | 0.419 | 170.4 | +33.8 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.324 | 471.0 | +33.8 |
| filt9E_6L | filt9E_6L | 0.513 | 0.412 | 98.5 | +24.6 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.723 | 73.9 | +19.9 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.337 | 456.2 | +19.9 |
| filt6E_6L | filt6E_6L | 0.553 | 0.410 | 105.5 | +16.7 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.374 | 491.1 | +16.5 |
| 202 | 3x4 1/.5 | 0.224 | 0.344 | 445.7 | +14.9 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.372 | 472.3 | +14.2 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.773 | 69.9 | +13.9 |
| 222 | 1x3 | 0.392 | 0.422 | 489.5 | +13.1 |
| freq369_39mx | freq369_39mx | 0.622 | 0.485 | 449.5 | +11.8 |
| 205 | 2x4 | 0.303 | 0.299 | 485.8 | +11.4 |
| 219 | 2x3 6/9 | 0.547 | 0.483 | 389.4 | +10.0 |
| 225 | 2x2 | 0.787 | 0.656 | 209.2 | +9.6 |
| filt9E_9L | filt9E_9L | 0.601 | 0.420 | 158.3 | +9.3 |
| 201 | 3x4 | 0.191 | 0.337 | 468.4 | +7.3 |
| 203 | 3x4 1/.66 | 0.211 | 0.336 | 453.9 | +5.7 |
| 216 | 2x3 | 0.580 | 0.473 | 316.6 | +5.3 |
| 224 | 3x3 | 0.426 | 0.508 | 464.9 | +4.2 |
| 218 | 3x4 6/8 | 0.419 | 0.332 | 273.4 | +2.6 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.374 | 427.3 | +1.9 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.368 | 435.8 | -2.4 |
| filt0E_9L | filt0E_9L | 0.586 | 0.599 | 67.7 | -2.8 |
| 217 | 2x4 6/12 | 0.325 | 0.288 | 302.9 | -5.4 |
| filt9E_3L | filt9E_3L | 0.414 | 0.385 | 69.2 | -5.5 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.643 | 47.4 | -6.1 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.368 | 401.3 | -11.0 |
| 207 | 4x4 +2w/o | 0.103 | 0.093 | 563.8 | -12.5 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.490 | 724.3 | -13.7 |
| 204 | 3x4 +6o | 0.272 | 0.322 | 446.9 | -14.6 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.683 | 56.7 | -18.0 |
| filt6E_3L | filt6E_3L | 0.424 | 0.381 | 67.8 | -21.0 |
| filt0E_6L | filt0E_6L | 0.487 | 0.532 | 59.3 | -22.8 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.577 | 801.3 | -30.8 |
| filt3E_3L | filt3E_3L | 0.449 | 0.372 | 77.6 | -36.8 |
| filt0E_3L | filt0E_3L | 0.372 | 0.406 | 71.0 | -48.9 |
| 221 | 1x4 | 0.190 | 0.245 | 548.0 | -72.8 |

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
| 1 | xsl_04 | Interference learning with test-time object-claim competition | 8 | 15879.4 | 0.5113 | 31165.9 | d=0.229, q=0.4687, lam=0.07155, Mmax=1.767, fl=0.0007025, b=0.1984, ch_dec=1.325, lapse=0.2755 |
| 2 | xsl_05 | Consecutive-repetition boosted interference learning | 9 | 16040.9 | 0.5465 | 31077.1 | d=0.2088, q=0.3556, lam=0.06911, Mmax=3.588, fl=0.0007805, b=0.1948, r=1.482, ch_dec=1.332, lapse=0.2629 |
| 3 | xsl_06 | Consolidation-protected boosted interference learning | 10 | 16095.6 | 0.5452 | 31093.9 | d=0.2258, q=0.4937, lam=0.06846, Mmax=9.398, fl=0.0007459, b=0.1973, r=0.8932, h=28.48, ch_dec=1.331, lapse=0.2648 |
| 4 | xsl_02 | Two-stage verify-and-consolidate with object-driven load | 6 | 16104.9 | 0.5384 | 32002.0 | a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g |
| 5 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 6 | 16128.6 | 0.5626 | 31863.2 | d=0.2836, q=0.03972, lam=0.381, Mmax=9.816, ch_dec=0.6064, lapse=0.4524 |
| 6 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 16146.9 | 0.5900 | 32045.4 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 7 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 16274.8 | 0.5976 | 32391.7 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 8 | baseline | Co-occurrence counting | 1 | 19786.1 | 1.3811 | 39486.6 | lapse=0.3881 |

## LOOP HISTORY
(first iteration)

## Response

{"interpretation":"The candidate is accepted but ranks below the leaderboard leader xsl_04 (held-out NLL 16421 vs 15879, +542). It was also worse than xsl_05, xsl_06, xsl_02, xsl_03, xsl_01 and Kachergis, so it is mid-to-low on the board. The new mechanism, a power-law encoding penalty (3/no)**g with g≈2.04, overfits the 1x4/2x4/3x4 designs. It gains on 221 (1x4, -73), 301 and the filt low-load conditions. It loses badly on conditions with 3 or fewer objects per trial, which get no penalty, and on high-frequency or high-contextual-diversity items.\n\nThe worst misfits are:\n- 220 (3x4 9/12): human 0.685, model 0.41. The model cannot produce high accuracy with 4 objects per trial, because the encoding penalty is applied at every frequency.\n- 215 (1x3): human 0.744, model 0.42.\n- freq369/cont_div: human 0.49 and 0.435, model near 0.47-0.52. The model misses the frequency and diversity structure.\n- filt3E_6L, filt3E_9L, filt6E_9L: human 0.48-0.59, model about 0.41-0.42. The filt conditions are nearly flat in the model.\n\nThe fit has pathological parameters. Mmax is at the bound of 10, so the saturation is effectively off. b=0 means the test-time claim competition is inactive. Accuracy stays compressed at about 0.3-0.5 for almost all conditions, so the model fails to separate easy conditions (1x3, 2x2, 220) from hard ones. The lapse of 0.096 is low and compensated by a large floor fl=0.2. That floor flattens everything, so extra learning is poorly expressed. The fitted values for lam, g and fl are high and trade off against each other. This suggests that encoding scaling and the floor are absorbing variance that should come from frequency-dependent learning and strengthening.\n\nThe strong, parameter-hungry g exponent generalises poorly. Per-trial candidate count is already partly captured by the interference load (nw*no)**q. xsl_04 does better with a simpler structure, so the extra gating adds little.","verdict":"regenerate","rationale":"Drop the (3/no)**g encoding gate, which duplicates the load term, and return to the xsl_04 backbone (load-scaled interference d*(nw*no/12)^q, saturation Mmax, test-time claim competition with b>0). Fix the failure to separate easy from hard conditions by removing the large additive floor fl (keep it near 0, as in xsl_04) and letting lapse absorb guessing. Add one mechanistic term for what is missing: a frequency-dependent strengthening of the winner. Make the responsibility sharpening exponent s a fitted parameter (about 1-6) instead of fixing s=4, so that repeated co-occurrence of a consistent pair is amplified and high-frequency or low-diversity conditions (220, freq369, filt*L) reach high accuracy. Keep b in about 0.2-1 so the model prefers unclaimed objects. Require Mmax to be an interior value, with an upper bound of about 5, not pinned at 10."}

## Usage

```json
{
  "input_tokens": 8634,
  "output_tokens": 1115,
  "reasoning_tokens": 0,
  "provider": "Claude Platform on AWS"
}
```
