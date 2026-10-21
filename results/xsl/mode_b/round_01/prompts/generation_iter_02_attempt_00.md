# generation_iter_02_attempt_00

## System Prompt

You are a cognitive scientist and expert Python programmer working on theories of cross-situational word learning. Your job is to propose a NEW theory, with an executable learner, that explains the human data better (lower BIC) than every theory on the leaderboard.

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

## ACCEPT GATE
You work in a loop. Each proposal is fitted and scored; it is ACCEPTED only if its BIC is strictly lower than the best accepted candidate so far in this loop. `## CURRENT BEST CANDIDATE` always shows the best accepted candidate, never a rejected one, so build on it. A critic comments on each candidate; the `## LOOP HISTORY` block shows which of its suggestions helped (ACCEPTED) and which did not (REJECTED).

Leaderboard theories are for inspiration only: do not resubmit one of them or a cosmetic variant. A new mechanism that fits better is the goal.

## OUTPUT
Return one JSON object with these fields:
- name: Short name for the theory (a few words).
- theory: The theoretical claim this theory makes about how people learn words across situations.
- learn: Python source defining def learn(params, trials, n_words, n_objects) -> np.ndarray, returning the final (n_words, n_objects) association matrix. `np` is available.
- parameters: Free parameters as intervals, e.g. {"X": "[0.001, 5]"}; a bare number pins a value.
- rationale: Why this theory should fit human cross-situational learning better than the alternatives.


## User Prompt

## GOAL
Propose a theory of cross-situational word learning that achieves a lower BIC than every theory on the leaderboard. Study which conditions the best theory fits worst and what those designs have in common, and target that mechanism.

## CORPUS (47 conditions)
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

## LEADERBOARD (fitted to the full corpus; lower BIC is better)
| rank | label | name | k | BIC | NLL | wSSE | fitted parameters |
|---|---|---|---|---|---|---|---|
| 1 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 32045.4 | 15996.8 | 0.5529 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 2 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 32391.7 | 16175.2 | 0.5668 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 3 | baseline | Co-occurrence counting | 1 | 39486.6 | 19738.1 | 1.3660 | lapse=0.3881 |

## LEADERBOARD THEORIES
### xsl_01: Interference-limited mutual-responsibility learning
On each trial the learner divides credit for each word among the co-present objects, and for each object among the co-present words, in proportion to sharpened current strength plus a small prior. Each pair is strengthened by the product of its word-side and object-side responsibilities, which gives soft mutual exclusivity and explaining away. Forgetting is interference-based rather than global: only associations involving a word or object that is currently on the trial decay. Knowledge of items not currently competing persists, so extra exposure and long training help, while dense competition (many words and objects per trial) erodes strength. Test choice is a power-law Luce competition.

### kachergis: Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012)
Learners distribute a limited amount of attention over the word-object pairs on each trial. Attention is drawn both to familiar pairs (already strongly associated) and to uncertain stimuli (words or objects whose associations are still spread over many partners, measured by entropy). The attended share of a fixed associative weight is added to each pair, and all associations decay slightly on every trial. Familiarity produces a rich-get-richer bias; uncertainty pulls attention toward stimuli that have no strong partner yet, which together implement a soft mutual-exclusivity bias.

### baseline: Co-occurrence counting
Learners simply count how often each word and object occur together across trials, with no attention, competition or forgetting.

## BEST LEADERBOARD THEORY: xsl_01
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

Per-condition fit, worst first:
| label | condition | human acc | model acc | NLL | NLL |
|---|---|---|---|---|---|
| 301 | 2x2 5x 24AFC | 0.514 | 0.529 | 776.1 | +776.1 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.480 | 705.6 | +705.6 |
| 223 | 4x4 | 0.312 | 0.333 | 647.4 | +647.4 |
| 207 | 4x4 +2w/o | 0.103 | 0.122 | 605.9 | +605.9 |
| freq369_36mx | freq369_36mx | 0.454 | 0.451 | 600.0 | +600.0 |
| 206 | 3x3 +1w/o | 0.167 | 0.327 | 595.5 | +595.5 |
| 221 | 1x4 | 0.190 | 0.319 | 590.3 | +590.3 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.474 | 557.8 | +557.8 |
| 204 | 3x4 +6o | 0.272 | 0.383 | 481.6 | +481.6 |
| 205 | 2x4 | 0.303 | 0.372 | 479.2 | +479.2 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.339 | 469.2 | +469.2 |
| 201 | 3x4 | 0.191 | 0.362 | 467.4 | +467.4 |
| 222 | 1x3 | 0.392 | 0.429 | 466.1 | +466.1 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.336 | 455.8 | +455.8 |
| 203 | 3x4 1/.66 | 0.211 | 0.363 | 454.1 | +454.1 |
| 224 | 3x3 | 0.426 | 0.471 | 451.3 | +451.3 |
| 202 | 3x4 1/.5 | 0.224 | 0.370 | 438.4 | +438.4 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.329 | 435.5 | +435.5 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.338 | 432.2 | +432.2 |
| freq369_39mx | freq369_39mx | 0.622 | 0.456 | 428.2 | +428.2 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.348 | 425.2 | +425.2 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.366 | 419.4 | +419.4 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.419 | 416.2 | +416.2 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.329 | 415.6 | +415.6 |
| 219 | 2x3 6/9 | 0.547 | 0.489 | 384.7 | +384.7 |
| 220 | 3x4 9/12 | 0.685 | 0.495 | 363.6 | +363.6 |
| 215 | 1x3 | 0.744 | 0.427 | 322.1 | +322.1 |
| 216 | 2x3 | 0.580 | 0.486 | 313.5 | +313.5 |
| 217 | 2x4 6/12 | 0.325 | 0.371 | 306.7 | +306.7 |
| 218 | 3x4 6/8 | 0.419 | 0.358 | 269.4 | +269.4 |
| 225 | 2x2 | 0.787 | 0.541 | 233.2 | +233.2 |
| filt9E_3L | filt9E_3L | 0.414 | 0.476 | 154.4 | +154.4 |
| filt3E_3L | filt3E_3L | 0.449 | 0.492 | 143.6 | +143.6 |
| filt3E_9L | filt3E_9L | 0.575 | 0.532 | 140.5 | +140.5 |
| filt6E_3L | filt6E_3L | 0.424 | 0.472 | 134.1 | +134.1 |
| filt9E_9L | filt9E_9L | 0.601 | 0.515 | 133.2 | +133.2 |
| filt0E_3L | filt0E_3L | 0.372 | 0.579 | 128.7 | +128.7 |
| filt6E_9L | filt6E_9L | 0.590 | 0.508 | 128.1 | +128.1 |
| filt6E_6L | filt6E_6L | 0.553 | 0.490 | 108.1 | +108.1 |
| filt9E_6L | filt9E_6L | 0.513 | 0.496 | 90.1 | +90.1 |
| filt3E_6L | filt3E_6L | 0.482 | 0.514 | 83.6 | +83.6 |
| filt0E_9L | filt0E_9L | 0.586 | 0.579 | 64.2 | +64.2 |
| filt0E_6L | filt0E_6L | 0.487 | 0.579 | 62.9 | +62.9 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.607 | 56.8 | +56.8 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.640 | 56.4 | +56.4 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.612 | 56.2 | +56.2 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.610 | 48.6 | +48.6 |

## CURRENT BEST CANDIDATE (this loop) — BIC 32070.1, params a=0.3417, f=0.02406, k=0.1585, ch_dec=1.298, lapse=0.4024
**Name:** Load-scaled propose-but-verify with consolidation

**Description:** Each word holds one tentative referent hypothesis (propose-but-verify), tracked as an exact expected distribution over learners. When a word is heard its unconsolidated hypothesis may be forgotten, and the forgetting probability grows with the trial's processing load (words x objects present). A hypothesis whose object is present is confirmed and may be consolidated into a durable, replacement-immune association. If the hypothesis is missing or absent, the learner re-proposes among the present objects with probability a, strongly preferring objects not already claimed by other words (mutual exclusivity). Unlearned words are guessed uniformly at test; choices follow a power-law Luce rule.

**Parameters:**
- a: [0.05, 1]
- f: [0, 0.5]
- k: [0, 1]
- ch_dec: [0.3, 6]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    a = params['a']
    f = params['f']
    k = params['k']
    me = 20.0
    h = np.zeros((n_words, n_objects))
    S = np.zeros((n_words, n_objects))
    for words, objects in trials:
        words = np.asarray(words)
        objects = np.asarray(objects)
        load = len(words) * len(objects) / 4.0
        fe = 1.0 - (1.0 - f) ** load
        hw = h[words] * (1.0 - fe)
        Sw = S[words]
        hp = hw[:, objects]
        none = np.clip(1.0 - Sw.sum(1) - hw.sum(1), 0.0, 1.0)
        habs = hw.sum(1) - hp.sum(1)
        moved = a * (none + habs)
        hall = hw * (1.0 - a)
        hall[:, objects] = hp
        col = (h + S).sum(0)
        own = (h[words] + Sw)[:, objects]
        other = np.maximum(col[objects][None, :] - own, 0.0)
        wts = 1.0 / (1.0 + me * other)
        wts = wts / wts.sum(1, keepdims=True)
        newh = hp + moved[:, None] * wts
        t = k * hp
        newh = newh - t
        hall[:, objects] = newh
        h[words] = hall
        Sw[:, objects] += t
        S[words] = Sw
    none_f = np.clip(1.0 - S.sum(1) - h.sum(1), 0.0, 1.0)
    return h + S + none_f[:, None] / n_objects + 1e-6
```

Per-condition fit, worst first:
| label | condition | human acc | model acc | NLL | NLL minus xsl_01's |
|---|---|---|---|---|---|
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.445 | 764.2 | +58.6 |
| 221 | 1x4 | 0.190 | 0.397 | 625.5 | +35.3 |
| 207 | 4x4 +2w/o | 0.103 | 0.142 | 635.8 | +29.9 |
| freq369_39mx | freq369_39mx | 0.622 | 0.423 | 452.7 | +24.5 |
| 206 | 3x3 +1w/o | 0.167 | 0.309 | 618.6 | +23.1 |
| 223 | 4x4 | 0.312 | 0.363 | 668.5 | +21.1 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.368 | 485.8 | +16.6 |
| 201 | 3x4 | 0.191 | 0.389 | 482.9 | +15.5 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.362 | 468.9 | +13.1 |
| 202 | 3x4 1/.5 | 0.224 | 0.399 | 450.7 | +12.2 |
| 222 | 1x3 | 0.392 | 0.490 | 477.4 | +11.3 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.346 | 445.6 | +10.0 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.334 | 429.4 | +10.0 |
| 220 | 3x4 9/12 | 0.685 | 0.477 | 373.1 | +9.5 |
| 205 | 2x4 | 0.303 | 0.399 | 488.1 | +8.9 |
| 203 | 3x4 1/.66 | 0.211 | 0.394 | 461.6 | +7.5 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.346 | 423.0 | +7.4 |
| 204 | 3x4 +6o | 0.272 | 0.401 | 488.9 | +7.3 |
| 224 | 3x3 | 0.426 | 0.452 | 457.7 | +6.5 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.627 | 62.4 | +5.9 |
| 216 | 2x3 | 0.580 | 0.474 | 318.1 | +4.6 |
| freq369_36mx | freq369_36mx | 0.454 | 0.424 | 604.5 | +4.5 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.594 | 61.1 | +4.3 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.593 | 52.3 | +3.7 |
| 217 | 2x4 6/12 | 0.325 | 0.367 | 310.0 | +3.3 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.615 | 59.2 | +3.0 |
| 218 | 3x4 6/8 | 0.419 | 0.376 | 271.6 | +2.2 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.340 | 426.2 | +1.0 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.381 | 417.2 | +1.0 |
| filt0E_9L | filt0E_9L | 0.586 | 0.592 | 64.2 | -0.0 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.545 | 775.9 | -0.2 |
| filt3E_9L | filt3E_9L | 0.575 | 0.582 | 140.2 | -0.3 |
| 219 | 2x3 6/9 | 0.547 | 0.492 | 382.7 | -2.0 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.366 | 429.5 | -2.8 |
| filt3E_6L | filt3E_6L | 0.482 | 0.528 | 78.5 | -5.1 |
| filt0E_6L | filt0E_6L | 0.487 | 0.539 | 56.6 | -6.3 |
| 225 | 2x2 | 0.787 | 0.565 | 225.9 | -7.3 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.448 | 544.1 | -13.7 |
| filt6E_9L | filt6E_9L | 0.590 | 0.541 | 110.5 | -17.5 |
| filt6E_6L | filt6E_6L | 0.553 | 0.488 | 83.0 | -25.1 |
| filt9E_9L | filt9E_9L | 0.601 | 0.496 | 107.5 | -25.7 |
| 215 | 1x3 | 0.744 | 0.487 | 294.5 | -27.6 |
| filt9E_6L | filt9E_6L | 0.513 | 0.449 | 61.8 | -28.3 |
| filt3E_3L | filt3E_3L | 0.449 | 0.406 | 115.3 | -28.4 |
| filt6E_3L | filt6E_3L | 0.424 | 0.391 | 102.9 | -31.2 |
| filt0E_3L | filt0E_3L | 0.372 | 0.397 | 65.7 | -62.9 |
| filt9E_3L | filt9E_3L | 0.414 | 0.371 | 89.5 | -64.9 |

## LOOP HISTORY
- iter 0: Expected propose-but-verify with consolidation — BIC 32080.0 — ACCEPTED
  critic: Make forgetting and consolidation depend on trial load and on repeated confirmation, so the model gains dynamic range and the lapse can fall. (1) Replace the constant forgetting f with a load-dependent rate, f_eff = 1 - (1-f)^(n_objs_in_trial * n_words_in_trial / c), or simply f * (n_objs_in_trial - 1). A hypothesis is then much more likely to be lost in 1x4, 3x4 and 4x4 trials than in 1x3, 2x2 and 3x3 trials. (2) Make consolidation accelerate with confirmation. For example, use a probability k * (confirmed mass)^g with an exponent g > 1, or let an already-confirmed hypothesis consolidate at a higher rate. Words confirmed on several occasions (high frequency, low contextual diversity, 2x2) would then be locked in, and the model could reach 0.6-0.8 accuracy in easy conditions. Keep the number of free parameters about the same: drop me, since it is stuck at its bound and a fixed value of about 10 would do, and add one load-scaling parameter. Expect the fitted lapse to drop well below 0.4.
- iter 1: Load-scaled propose-but-verify with consolidation — BIC 32070.1 — ACCEPTED
  critic: Keep the expected-distribution propose-but-verify core and mutual exclusivity. Change two things, with a net of at most one extra parameter (drop the fixed `me` scaling if needed). (1) Make forgetting steeply and super-linearly dependent on the number of objects in the trial: fe = 1 - (1-f)^(n_obj**g * n_words**0.5), with a fitted exponent g in [1,4] and f allowed up to about 0.5. This should push 1x4, 3x4 and 4x4 down toward 0.2 while leaving 2x2 and 3x3 nearly unaffected. (2) Make consolidation accelerate with repeated confirmation. Track a per-word confirmation count (expected, or mass-weighted). Use a two-stage hypothesis state, fresh then confirmed once, where only a second confirmation consolidates, with probability k. Alternatively, use consolidation probability k*(confirmed mass)**2 in place of the linear k*hp. Confirmed-once hypotheses should also be forgotten at a reduced rate. This should lift high-frequency, low-diversity and 2x2 conditions toward 0.7–0.8 and let the fitted lapse fall well below 0.4.

## Response


<invoke name="Bash">
<parameter name="command">echo skip</parameter>
</invoke>

{"name":"Two-stage verify-and-consolidate with object-driven load","theory":"Each word holds a tentative referent hypothesis (propose-but-verify), tracked as an exact expected distribution over learners, passing through three stages: freshly proposed, confirmed once, and consolidated (durable). A hypothesis is only consolidated after a second confirmation. Unconsolidated hypotheses are forgotten whenever the word is heard, with a probability that rises steeply (power-law) with the number of objects on the trial and with the square root of the number of words, and fresh hypotheses are more fragile than confirmed ones. A word with no surviving hypothesis, or whose hypothesis object is absent, re-proposes among the present objects with probability a, strongly preferring objects already claimed by no other word (mutual exclusivity). Unlearned words are guessed uniformly; choice is a power-law Luce rule.","learn":"def learn(params, trials, n_words, n_objects):\n    a = params['a']\n    f = params['f']\n    k = params['k']\n    g = params['g']\n    me = 20.0\n    h1 = np.zeros((n_words, n_objects))\n    h2 = np.zeros((n_words, n_objects))\n    S = np.zeros((n_words, n_objects))\n    for words, objects in trials:\n        words = np.asarray(words)\n        objects = np.asarray(objects)\n        nw = len(words)\n        no = len(objects)\n        load = (nw * 0.5) ** 0.5 * (no / 2.0) ** g\n        fe = 1.0 - (1.0 - f) ** load\n        h1w = h1[words] * (1.0 - fe)\n        h2w = h2[words] * (1.0 - 0.5 * fe)\n        Sw = S[words]\n        p1 = h1w[:, objects]\n        p2 = h2w[:, objects]\n        tot = h1w.sum(1) + h2w.sum(1) + Sw.sum(1)\n        none = np.clip(1.0 - tot, 0.0, 1.0)\n        habs = (h1w.sum(1) - p1.sum(1)) + (h2w.sum(1) - p2.sum(1))\n        moved = a * (none + habs)\n        col = (h1 + h2 + S).sum(0)\n        own = (h1[words] + h2[words] + Sw)[:, objects]\n        other = np.maximum(col[objects][None, :] - own, 0.0)\n        wts = 1.0 / (1.0 + me * other)\n        wts = wts / wts.sum(1, keepdims=True)\n        h1all = h1w * (1.0 - a)\n        h1all[:, objects] = moved[:, None] * wts\n        h2all = h2w * (1.0 - a)\n        h2all[:, objects] = p2 * (1.0 - k) + p1\n        Sw = Sw.copy()\n        Sw[:, objects] += k * p2\n        h1[words] = h1all\n        h2[words] = h2all\n        S[words] = Sw\n    tot = h1.sum(1) + h2.sum(1) + S.sum(1)\n    none_f = np.clip(1.0 - tot, 0.0, 1.0)\n    return h1 + h2 + S + none_f[:, None] / n_objects + 1e-6\n","parameters":{"a":"[0.05, 1]","f":"[0, 0.5]","k":"[0, 1]","g":"[0.5, 3]","ch_dec":"[0.3, 6]"},"rationale":"The current best propose-but-verify model overpredicts accuracy for 1x4 (0.40 vs 0.19) and underpredicts it for easy, repeated-confirmation designs such as 2x2 and 1x3. Making forgetting depend steeply on the number of objects (fitted exponent g) separates 1x4, 3x4 and 4x4 from 1x3, 2x2 and 3x3, and requiring a second confirmation before consolidation, with confirmed-once hypotheses forgotten at half the rate, lets repeated consistent evidence lock words in while a single chance co-occurrence stays fragile. This should raise dynamic range across designs and lower the needed lapse."}

## Usage

```json
{
  "input_tokens": 12010,
  "output_tokens": 4121,
  "reasoning_tokens": 2628,
  "provider": "Claude Platform on AWS"
}
```
