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

## CANDIDATE — BIC 32062.0, NLL 16005.1, wSSE 0.5614 — REJECTED (not better than this loop's best)
Fitted parameters: s=1.006, g=2.703, C=0.8414, ch_dec=0.9162, lapse=0.4657

**Name:** Competitor-erosion mutual-responsibility learning

**Description:** On each trial the learner splits credit for each word among the co-present objects, and for each object among the co-present words, in proportion to sharpened current strength plus a small prior. Each pair is strengthened by the product of its word-side and object-side responsibilities, which gives soft mutual exclusivity and explaining away. Forgetting is competitor-specific. Only the rival associations of the words and objects on the trial decay: the other objects linked to a present word, and the other words linked to a present object. The co-present pairs being reinforced are not decayed, so repeated co-occurrence accumulates. Associations of items not on the trial persist untouched. Test choice is a power-law Luce competition.

**Parameters:**
- s: [0.01, 5]
- g: [0.5, 5]
- C: [0.3, 1]
- ch_dec: [0.3, 6]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    s = params['s']
    g = params['g']
    C = params['C']
    m = np.zeros((n_words, n_objects))
    for words, objects in trials:
        ix = np.ix_(words, objects)
        A = (m[ix] + s) ** g
        pw = A / A.sum(axis=1, keepdims=True)
        po = A / A.sum(axis=0, keepdims=True)
        dec = np.ones((n_words, n_objects))
        dec[words, :] = C
        dec[:, objects] = C
        dec[ix] = 1.0
        m *= dec
        m[ix] += pw * po
    return m + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc | NLL | NLL minus kachergis's |
|---|---|---|---|---|---|
| filt9E_3L | filt9E_3L | 0.414 | 0.459 | 154.9 | +83.5 |
| filt6E_3L | filt6E_3L | 0.424 | 0.467 | 133.3 | +63.7 |
| 215 | 1x3 | 0.744 | 0.424 | 322.9 | +61.4 |
| filt3E_3L | filt3E_3L | 0.449 | 0.501 | 145.6 | +57.0 |
| filt0E_3L | filt0E_3L | 0.372 | 0.577 | 127.5 | +47.2 |
| filt3E_9L | filt3E_9L | 0.575 | 0.559 | 150.9 | +40.2 |
| filt9E_9L | filt9E_9L | 0.601 | 0.515 | 134.1 | +39.8 |
| filt6E_9L | filt6E_9L | 0.590 | 0.530 | 128.9 | +36.5 |
| filt9E_6L | filt9E_6L | 0.513 | 0.482 | 93.6 | +32.8 |
| filt6E_6L | filt6E_6L | 0.553 | 0.497 | 106.4 | +32.3 |
| filt3E_6L | filt3E_6L | 0.482 | 0.537 | 87.3 | +24.3 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.613 | 55.2 | +6.9 |
| 204 | 3x4 +6o | 0.272 | 0.384 | 480.9 | +2.3 |
| 224 | 3x3 | 0.426 | 0.462 | 449.5 | +2.3 |
| 225 | 2x2 | 0.787 | 0.536 | 235.0 | +1.3 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.360 | 425.1 | +0.7 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.639 | 56.7 | +0.6 |
| 218 | 3x4 6/8 | 0.419 | 0.362 | 269.0 | +0.6 |
| filt0E_6L | filt0E_6L | 0.487 | 0.577 | 62.5 | +0.1 |
| 216 | 2x3 | 0.580 | 0.477 | 315.2 | -0.1 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.613 | 56.2 | -0.6 |
| 219 | 2x3 6/9 | 0.547 | 0.479 | 385.8 | -0.7 |
| filt0E_9L | filt0E_9L | 0.586 | 0.577 | 64.3 | -1.7 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.351 | 471.0 | -1.7 |
| 203 | 3x4 1/.66 | 0.211 | 0.367 | 454.7 | -2.0 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.523 | 775.4 | -2.4 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.348 | 456.4 | -4.9 |
| 223 | 4x4 | 0.312 | 0.345 | 647.4 | -5.0 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.613 | 48.5 | -5.0 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.463 | 552.3 | -7.6 |
| 220 | 3x4 9/12 | 0.685 | 0.497 | 362.6 | -8.8 |
| 202 | 3x4 1/.5 | 0.224 | 0.373 | 439.7 | -9.9 |
| 217 | 2x4 6/12 | 0.325 | 0.368 | 306.3 | -10.0 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.350 | 430.0 | -11.9 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.342 | 432.4 | -12.1 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.373 | 419.4 | -14.2 |
| 201 | 3x4 | 0.191 | 0.367 | 469.2 | -16.2 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.342 | 410.3 | -21.5 |
| freq369_36mx | freq369_36mx | 0.454 | 0.441 | 598.1 | -37.1 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.410 | 414.8 | -38.4 |
| 205 | 2x4 | 0.303 | 0.370 | 477.9 | -42.5 |
| freq369_39mx | freq369_39mx | 0.622 | 0.445 | 433.6 | -42.8 |
| 207 | 4x4 +2w/o | 0.103 | 0.129 | 603.8 | -47.1 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.471 | 713.8 | -59.7 |
| 222 | 1x3 | 0.392 | 0.424 | 465.4 | -62.3 |
| 206 | 3x3 +1w/o | 0.167 | 0.329 | 593.9 | -65.5 |
| 221 | 1x4 | 0.190 | 0.318 | 587.4 | -171.8 |

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
| 1 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 32391.7 | 16175.2 | 0.5668 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 2 | baseline | Co-occurrence counting | 1 | 39486.6 | 19738.1 | 1.3660 | lapse=0.3881 |

## LOOP HISTORY
- iter 0: Mutual-responsibility competitive learning — BIC 32279.8 — ACCEPTED
  critic: Replace global per-trial decay with interference-based forgetting, so that only the rows and columns of the words and objects present on the trial decay. Concretely: m[words,:] *= C and m[:,objects] *= C, or decay only m[ix]. Then add the responsibility update. Keep ch_dec. This keeps the parameter count the same. Strength then persists for items that are not currently competing, so more trials and more repetitions help, which should lift 215, 225, 220 and the filt*_9L conditions. Heavy competition, as in 3x4 and 4x4, still erodes strength, which should lower the over-predicted hard conditions. If that still leaves the accuracy range compressed, also let the sharpening exponent g act on the final matrix at test, or add a mild familiarity bonus. Both are alternatives to raising ch_dec, and both would let the lapse drop below 0.46.
- iter 1: Interference-limited mutual-responsibility learning — BIC 32045.4 — ACCEPTED
  critic: Keep the mutual-responsibility update, the power-law sharpening and ch_dec. Change the forgetting so that it hits only competitors, not the pairs that are co-present and being reinforced. On each trial, decay by C only the cells in the rows of the present words and the columns of the present objects that lie outside the present word-by-object block. Concretely: dec = ones; dec[words,:] = C; dec[:,objects] = C; dec[ix] = 1; m *= dec; then m[ix] += pw*po. Repeated co-occurrence then accumulates, because a pair is not eroded by its own reinforcement, while the competing associations of those words and objects (the wrong pairings) erode. That should let accuracy rise with frequency and trial count (215, 225, 220, filt*_6L and filt*_9L, the filt0E_3L-to-9L trend). Dense 3x4 and 4x4 trials would still have many competitors eroding each other, which should lower the over-predicted hard conditions. This keeps the parameter count at four plus lapse. With a wider accuracy range the fit should be able to lower the lapse from 0.46. If it is still compressed, optionally let C vary with the current row mass, or widen the g range.

## Response

```json
{
  "interpretation": "The candidate is rejected: BIC 32062 is worse than this loop's best (32045), and it is only the critic's iter-1 suggestion carried out. Its main fault is compressed dynamic range, and the fitted lapse of 0.47 shows it. With a lapse that high, the best possible n-AFC accuracy is about 0.56. The model therefore cannot reach the easy conditions: 215 (1x3) human 0.74 vs model 0.42, 225 (2x2) 0.79 vs 0.54, 220 (3x4 9/12) 0.69 vs 0.50, and filt*_9L about 0.59 vs 0.52. At the same time it overpredicts the dense, hard conditions: 201 0.19 vs 0.37, 202 0.22 vs 0.37, 203 0.21 vs 0.37, 206 0.17 vs 0.33, 221 0.19 vs 0.32, 205 0.30 vs 0.37. The fit settles on a high lapse, a small decay per trial (C=0.84) and a mild ch_dec as a compromise. Human accuracy falls steadily with ambiguity per trial (1x3 and 2x2 about 0.75-0.79, 2x3 0.58, 3x3 0.43, 3x4 0.19-0.22). The model's forgetting is the same constant C whether a trial has 3 cells or 16, so it does not capture this. The model also ignores the order structure in the filt* family. All filt0E conditions come out at exactly 0.577 against human 0.37 to 0.59, and the filt*_3L conditions are the worst misfits, up to +84 NLL against Kachergis. The model learns almost nothing from trial order or repetition there. Because of this it loses to Kachergis on filt*, 215 and 225 by roughly 600 NLL in total, and wins only by 170 on 221 and by smaller amounts on several other conditions.",
  "verdict": "regenerate",
  "rationale": "Keep the mutual-responsibility update, the sharpening (s, g), the competitor-only decay and ch_dec. Change the forgetting so it scales with the ambiguity of the trial instead of using a fixed C. Use exponent = len(words) * len(objects) and set dec[words,:] = dec[:,objects] = C ** exponent (still leaving dec[ix] = 1), or use C ** (len(objects) * len(words) / 4). Re-range C to about [0.9, 1.0], and keep k at four parameters plus lapse. Small trials (1x3, 2x2, 2x3) then keep their strengths and approach ceiling, so the lapse can fall well below 0.47 and accuracy can reach 0.75-0.8. Dense trials (3x4, 4x4, 3x3+1w/o) erode competing associations much more, which should pull the 0.19-0.22 conditions down from about 0.37. If it still fits poorly, also scale the credit added per trial, m[ix] += pw*po / sqrt(cells), so that dense trials do not distribute more total mass."
}
```

## Usage

```json
{
  "input_tokens": 8085,
  "output_tokens": 4208,
  "reasoning_tokens": 3174,
  "provider": "Claude Platform on AWS"
}
```
