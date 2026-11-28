# critique_iter_01

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

## CANDIDATE — BIC 32045.4, NLL 15996.8, wSSE 0.5529 — ACCEPTED
Fitted parameters: s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636

**Name:** Interference-limited mutual-responsibility learning

**Description:** On each trial the learner divides credit for each word among the co-present objects, and for each object among the co-present words, in proportion to sharpened current strength plus a small prior. Each pair is strengthened by the product of its word-side and object-side responsibilities, which gives soft mutual exclusivity and explaining away. Forgetting is interference-based rather than global: only associations involving a word or object that is currently on the trial decay. Knowledge of items not currently competing persists, so extra exposure and long training help, while dense competition (many words and objects per trial) erodes strength. Test choice is a power-law Luce competition.

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
        dec = np.ones((n_words, n_objects))
        dec[words, :] = C
        dec[:, objects] = C
        m *= dec
        m[ix] += pw * po
    return m + 1e-6
```


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc | NLL | NLL minus kachergis's |
|---|---|---|---|---|---|
| filt9E_3L | filt9E_3L | 0.414 | 0.476 | 154.4 | +83.0 |
| filt6E_3L | filt6E_3L | 0.424 | 0.472 | 134.1 | +64.5 |
| 215 | 1x3 | 0.744 | 0.427 | 322.1 | +60.6 |
| filt3E_3L | filt3E_3L | 0.449 | 0.492 | 143.6 | +55.0 |
| filt0E_3L | filt0E_3L | 0.372 | 0.579 | 128.7 | +48.4 |
| filt9E_9L | filt9E_9L | 0.601 | 0.515 | 133.2 | +38.9 |
| filt6E_9L | filt6E_9L | 0.590 | 0.508 | 128.1 | +35.6 |
| filt6E_6L | filt6E_6L | 0.553 | 0.490 | 108.1 | +34.0 |
| filt3E_9L | filt3E_9L | 0.575 | 0.532 | 140.5 | +29.8 |
| filt9E_6L | filt9E_6L | 0.513 | 0.496 | 90.1 | +29.3 |
| filt3E_6L | filt3E_6L | 0.482 | 0.514 | 83.6 | +20.6 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.607 | 56.8 | +8.5 |
| 224 | 3x3 | 0.426 | 0.471 | 451.3 | +4.0 |
| 204 | 3x4 +6o | 0.272 | 0.383 | 481.6 | +3.0 |
| 218 | 3x4 6/8 | 0.419 | 0.358 | 269.4 | +0.9 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.348 | 425.2 | +0.7 |
| filt0E_6L | filt0E_6L | 0.487 | 0.579 | 62.9 | +0.5 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.640 | 56.4 | +0.4 |
| 225 | 2x2 | 0.787 | 0.541 | 233.2 | -0.5 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.612 | 56.2 | -0.6 |
| filt0E_9L | filt0E_9L | 0.586 | 0.579 | 64.2 | -1.7 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.529 | 776.1 | -1.7 |
| 219 | 2x3 6/9 | 0.547 | 0.489 | 384.7 | -1.8 |
| 216 | 2x3 | 0.580 | 0.486 | 313.5 | -1.9 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.474 | 557.8 | -2.1 |
| 203 | 3x4 1/.66 | 0.211 | 0.363 | 454.1 | -2.6 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.339 | 469.2 | -3.5 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.610 | 48.6 | -5.0 |
| 223 | 4x4 | 0.312 | 0.333 | 647.4 | -5.0 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.336 | 455.8 | -5.4 |
| 220 | 3x4 9/12 | 0.685 | 0.495 | 363.6 | -7.8 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.329 | 435.5 | -9.0 |
| 217 | 2x4 6/12 | 0.325 | 0.371 | 306.7 | -9.6 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.338 | 432.2 | -9.7 |
| 202 | 3x4 1/.5 | 0.224 | 0.370 | 438.4 | -11.1 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.366 | 419.4 | -14.2 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.329 | 415.6 | -16.2 |
| 201 | 3x4 | 0.191 | 0.362 | 467.4 | -17.9 |
| freq369_36mx | freq369_36mx | 0.454 | 0.451 | 600.0 | -35.3 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.419 | 416.2 | -37.0 |
| 205 | 2x4 | 0.303 | 0.372 | 479.2 | -41.1 |
| 207 | 4x4 +2w/o | 0.103 | 0.122 | 605.9 | -45.0 |
| freq369_39mx | freq369_39mx | 0.622 | 0.456 | 428.2 | -48.2 |
| 222 | 1x3 | 0.392 | 0.429 | 466.1 | -61.5 |
| 206 | 3x3 +1w/o | 0.167 | 0.327 | 595.5 | -63.9 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.480 | 705.6 | -67.9 |
| 221 | 1x4 | 0.190 | 0.319 | 590.3 | -168.9 |

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

## Response

```json
{
  "interpretation": "The candidate is accepted: BIC 32045 beats the best leaderboard entry (kachergis, 32392) with the same number of parameters. It still has a systematic misfit, a compressed accuracy range. The fitted lapse is very high at 0.46, and the model accuracy is nearly flat across conditions (0.33 to 0.64). The errors fall into three groups. (1) Easy, high-exposure conditions are badly under-predicted: 215 (1x3, 108 trials) is 0.43 against 0.74 human, 225 (2x2) is 0.54 against 0.79, and 220 is 0.50 against 0.69. The filt*_9L conditions are under-predicted too. (2) Hard conditions are over-predicted: 221 is 0.32 against 0.19, 201 and 202 are about 0.36 against 0.2, 206 is 0.33 against 0.17, and filt0E_3L is 0.58 against 0.37. These are also the worst NLL gaps against kachergis (221 is -169 in the candidate's favor, but the filt and 215 conditions are +30 to +83 against it). (3) All filt0E_* conditions get the same model accuracy of 0.579 whether the learning phase is 3, 6 or 9 blocks. The human data rise with training length. The cause is that every co-present cell is decayed by C (about 0.83) on every trial, even the pair being strengthened. Strength therefore hits a fixed equilibrium of roughly 1/(1-C) within a few repetitions, so extra repetitions and extra trials add nothing. That is why the model cannot show human learning growth with frequency or trial count, and why the fit needs a large lapse and a flat ch_dec to compensate. The responsibility step handles mutual exclusivity and explaining away reasonably well (the 3x3, 3x4 and 4x4 NLLs are near kachergis's).",
  "verdict": "regenerate",
  "rationale": "Keep the mutual-responsibility update, the power-law sharpening and ch_dec. Change the forgetting so that it hits only competitors, not the pairs that are co-present and being reinforced. On each trial, decay by C only the cells in the rows of the present words and the columns of the present objects that lie outside the present word-by-object block. Concretely: dec = ones; dec[words,:] = C; dec[:,objects] = C; dec[ix] = 1; m *= dec; then m[ix] += pw*po. Repeated co-occurrence then accumulates, because a pair is not eroded by its own reinforcement, while the competing associations of those words and objects (the wrong pairings) erode. That should let accuracy rise with frequency and trial count (215, 225, 220, filt*_6L and filt*_9L, the filt0E_3L-to-9L trend). Dense 3x4 and 4x4 trials would still have many competitors eroding each other, which should lower the over-predicted hard conditions. This keeps the parameter count at four plus lapse. With a wider accuracy range the fit should be able to lower the lapse from 0.46. If it is still compressed, optionally let C vary with the current row mass, or widen the g range."
}
```

## Usage

```json
{
  "input_tokens": 7594,
  "output_tokens": 1798,
  "reasoning_tokens": 733,
  "provider": "Claude Platform on AWS"
}
```
