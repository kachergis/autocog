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

## CANDIDATE — BIC 32279.8, NLL 16114.0, wSSE 0.5803 — ACCEPTED
Fitted parameters: s=0.3004, g=1.363, C=0.9573, ch_dec=1.115, lapse=0.4626

**Name:** Mutual-responsibility competitive learning

**Description:** On each trial the learner divides credit for each word among the co-present objects, and for each object among the co-present words, in proportion to current (sharpened) association strength plus a small prior. A pair is strengthened by the product of its word-side and object-side responsibilities, so pairs that are already explained by other partners get little credit (soft mutual exclusivity / explaining away) while an initially ambiguous trial spreads a small amount of credit over all pairs. Strength decays on every trial, and test choice is a power-law (Luce) competition over candidates.

**Parameters:**
- s: [0.01, 5]
- g: [0.5, 5]
- C: [0.5, 1]
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
        m *= C
        m[ix] += pw * po
    return m + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc | NLL | NLL minus kachergis's |
|---|---|---|---|---|---|
| 215 | 1x3 | 0.744 | 0.387 | 349.5 | +88.1 |
| filt9E_3L | filt9E_3L | 0.414 | 0.472 | 154.6 | +83.2 |
| filt6E_3L | filt6E_3L | 0.424 | 0.472 | 136.0 | +66.5 |
| filt3E_3L | filt3E_3L | 0.449 | 0.486 | 145.3 | +56.7 |
| filt0E_3L | filt0E_3L | 0.372 | 0.579 | 129.2 | +49.0 |
| filt9E_9L | filt9E_9L | 0.601 | 0.503 | 134.0 | +39.7 |
| filt6E_9L | filt6E_9L | 0.590 | 0.500 | 130.7 | +38.2 |
| filt6E_6L | filt6E_6L | 0.553 | 0.488 | 109.2 | +35.1 |
| filt9E_6L | filt9E_6L | 0.513 | 0.489 | 92.7 | +31.9 |
| filt3E_9L | filt3E_9L | 0.575 | 0.514 | 139.4 | +28.7 |
| filt3E_6L | filt3E_6L | 0.482 | 0.504 | 84.3 | +21.3 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.599 | 57.4 | +9.1 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.349 | 433.0 | +8.5 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.589 | 61.7 | +4.9 |
| 224 | 3x3 | 0.426 | 0.464 | 451.4 | +4.1 |
| 204 | 3x4 +6o | 0.272 | 0.383 | 481.4 | +2.9 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.629 | 58.8 | +2.7 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.360 | 475.4 | +2.7 |
| 216 | 2x3 | 0.580 | 0.474 | 316.7 | +1.3 |
| filt0E_6L | filt0E_6L | 0.487 | 0.579 | 63.1 | +0.7 |
| 203 | 3x4 1/.66 | 0.211 | 0.376 | 456.8 | +0.0 |
| 225 | 2x2 | 0.787 | 0.541 | 233.2 | -0.5 |
| 220 | 3x4 9/12 | 0.685 | 0.481 | 370.9 | -0.6 |
| 218 | 3x4 6/8 | 0.419 | 0.376 | 267.6 | -0.8 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.527 | 776.3 | -1.5 |
| 219 | 2x3 6/9 | 0.547 | 0.478 | 384.9 | -1.6 |
| filt0E_9L | filt0E_9L | 0.586 | 0.579 | 64.2 | -1.8 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.358 | 431.8 | -1.8 |
| 223 | 4x4 | 0.312 | 0.354 | 649.6 | -2.8 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.356 | 457.6 | -3.7 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.602 | 48.2 | -5.4 |
| 202 | 3x4 1/.5 | 0.224 | 0.370 | 442.9 | -6.7 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.464 | 552.1 | -7.9 |
| 201 | 3x4 | 0.191 | 0.380 | 475.4 | -9.9 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.359 | 431.1 | -10.8 |
| 217 | 2x4 6/12 | 0.325 | 0.379 | 304.3 | -12.0 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.357 | 431.6 | -12.9 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.357 | 406.4 | -25.4 |
| freq369_39mx | freq369_39mx | 0.622 | 0.437 | 442.7 | -33.7 |
| freq369_36mx | freq369_36mx | 0.454 | 0.443 | 598.3 | -37.0 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.408 | 414.1 | -39.1 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.459 | 734.3 | -39.3 |
| 205 | 2x4 | 0.303 | 0.383 | 478.8 | -41.5 |
| 207 | 4x4 +2w/o | 0.103 | 0.139 | 599.8 | -51.1 |
| 206 | 3x3 +1w/o | 0.167 | 0.341 | 602.1 | -57.3 |
| 222 | 1x3 | 0.392 | 0.390 | 466.6 | -61.1 |
| 221 | 1x4 | 0.190 | 0.316 | 588.7 | -170.4 |

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
(first iteration)

## Response

```json
{
  "interpretation": "The candidate is accepted: it beats the best leaderboard theory (BIC 32279.8 vs 32391.7). Its per-condition NLL is mixed. It wins big on 221 (1x4), 222, 206, 207, 205 and the freq369 conditions. It loses clearly on 215 (1x3, +88), all the 12x12 filtering conditions (+20 to +83) and 225 (2x2). The model's accuracy range is compressed (about 0.14 to 0.63) while human accuracy spans 0.10 to 0.79. It under-predicts the easy, low-ambiguity conditions: 215 at 0.39 vs 0.74, 225 at 0.54 vs 0.79, and 220 at 0.48 vs 0.69. It over-predicts the hard, high-ambiguity ones: 201 at 0.38 vs 0.19, 206 at 0.34 vs 0.17, 221 at 0.32 vs 0.19, and 202/203/204/205. The high lapse (0.46) and the global per-trial decay (C=0.957 applied to the whole matrix on every trial) are the likely cause. Under global decay, conditions with more trials, such as 108-trial 1x3 or 81-trial filt9E_9L, lose as much old evidence as they gain. This hides the benefit of extra exposure and of low ambiguity. The 11-AFC filtering conditions are also under-fit. In the human data, accuracy rises with more late-phase and early-phase repetitions (filt9E_9L 0.60 vs filt9E_3L 0.41), but the model is flat at about 0.47 to 0.58 and even falls with more early trials.",
  "verdict": "regenerate",
  "rationale": "Replace global per-trial decay with interference-based forgetting, so that only the rows and columns of the words and objects present on the trial decay. Concretely: m[words,:] *= C and m[:,objects] *= C, or decay only m[ix]. Then add the responsibility update. Keep ch_dec. This keeps the parameter count the same. Strength then persists for items that are not currently competing, so more trials and more repetitions help, which should lift 215, 225, 220 and the filt*_9L conditions. Heavy competition, as in 3x4 and 4x4, still erodes strength, which should lower the over-predicted hard conditions. If that still leaves the accuracy range compressed, also let the sharpening exponent g act on the final matrix at test, or add a mild familiarity bonus. Both are alternatives to raising ch_dec, and both would let the lapse drop below 0.46."
}
```

## Usage

```json
{
  "input_tokens": 7202,
  "output_tokens": 1482,
  "reasoning_tokens": 601,
  "provider": "Claude Platform on AWS"
}
```
