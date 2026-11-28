# generation_iter_00_attempt_00

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
| 1 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 6 | 31863.2 | 15900.6 | 0.5101 | d=0.2836, q=0.03972, lam=0.381, Mmax=9.816, ch_dec=0.6064, lapse=0.4524 |
| 2 | xsl_02 | Two-stage verify-and-consolidate with object-driven load | 6 | 32002.0 | 15969.9 | 0.4980 | a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g |
| 3 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 32045.4 | 15996.8 | 0.5529 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 4 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 32391.7 | 16175.2 | 0.5668 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 5 | baseline | Co-occurrence counting | 1 | 39486.6 | 19738.1 | 1.3660 | lapse=0.3881 |

## LEADERBOARD THEORIES
### xsl_03: Load-scaled absent-alternative interference with saturating competitive encoding
Each word splits credit over the co-present objects and each object over the co-present words (sharpened strength plus a prior). A pair gains the product of its two responsibilities, which gives soft mutual exclusivity and explaining away. The gain saturates as a pair's strength approaches a ceiling, and it is mildly gated down when many objects are present. Forgetting is retrieval-induced interference. When a word is heard, its links to absent objects decay, and when an object is seen, its links to absent words decay. Co-present pairs are protected from this decay. The decay rate grows as a power law in trial load, the number of words times the number of objects. Sparse trials (1x3, 2x2) therefore barely erode earlier learning, so accuracy is high and snowballs with repetition, while dense trials (3x4, 4x4) prune strongly. Three, six and nine repetitions give graded rather than all-or-none learning. Test choice is a power-law Luce rule.

### xsl_02: Two-stage verify-and-consolidate with object-driven load
Each word holds a tentative referent hypothesis (propose-but-verify), tracked as an exact expected distribution over learners, passing through three stages: freshly proposed, confirmed once, and consolidated (durable). A hypothesis is only consolidated after a second confirmation. Unconsolidated hypotheses are forgotten whenever the word is heard, with a probability that rises steeply (power-law) with the number of objects on the trial and with the square root of the number of words, and fresh hypotheses are more fragile than confirmed ones. A word with no surviving hypothesis, or whose hypothesis object is absent, re-proposes among the present objects with probability a, strongly preferring objects already claimed by no other word (mutual exclusivity). Unlearned words are guessed uniformly; choice is a power-law Luce rule.

### xsl_01: Interference-limited mutual-responsibility learning
On each trial the learner divides credit for each word among the co-present objects, and for each object among the co-present words, in proportion to sharpened current strength plus a small prior. Each pair is strengthened by the product of its word-side and object-side responsibilities, which gives soft mutual exclusivity and explaining away. Forgetting is interference-based rather than global: only associations involving a word or object that is currently on the trial decay. Knowledge of items not currently competing persists, so extra exposure and long training help, while dense competition (many words and objects per trial) erodes strength. Test choice is a power-law Luce competition.

### kachergis: Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012)
Learners distribute a limited amount of attention over the word-object pairs on each trial. Attention is drawn both to familiar pairs (already strongly associated) and to uncertain stimuli (words or objects whose associations are still spread over many partners, measured by entropy). The attended share of a fixed associative weight is added to each pair, and all associations decay slightly on every trial. Familiarity produces a rich-get-richer bias; uncertainty pulls attention toward stimuli that have no strong partner yet, which together implement a soft mutual-exclusivity bias.

### baseline: Co-occurrence counting
Learners simply count how often each word and object occur together across trials, with no attention, competition or forgetting.

## BEST LEADERBOARD THEORY: xsl_03
**Name:** Load-scaled absent-alternative interference with saturating competitive encoding

**Description:** Each word splits credit over the co-present objects and each object over the co-present words (sharpened strength plus a prior). A pair gains the product of its two responsibilities, which gives soft mutual exclusivity and explaining away. The gain saturates as a pair's strength approaches a ceiling, and it is mildly gated down when many objects are present. Forgetting is retrieval-induced interference. When a word is heard, its links to absent objects decay, and when an object is seen, its links to absent words decay. Co-present pairs are protected from this decay. The decay rate grows as a power law in trial load, the number of words times the number of objects. Sparse trials (1x3, 2x2) therefore barely erode earlier learning, so accuracy is high and snowballs with repetition, while dense trials (3x4, 4x4) prune strongly. Three, six and nine repetitions give graded rather than all-or-none learning. Test choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.05, 4]
- Mmax: [0.3, 10]
- ch_dec: [0.3, 6]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    d = params['d']
    q = params['q']
    lam = params['lam']
    Mmax = params['Mmax']
    s = 4.0
    c = 0.5
    g = 0.5
    eps = 0.1
    M = np.zeros((n_words, n_objects))
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
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        M[ix] = blk + lam * enc * inc * sat
    return M + 1e-6
```

Per-condition fit, worst first:
| label | condition | human acc | model acc | NLL | NLL |
|---|---|---|---|---|---|
| 301 | 2x2 5x 24AFC | 0.514 | 0.527 | 778.1 | +778.1 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.486 | 702.7 | +702.7 |
| 223 | 4x4 | 0.312 | 0.364 | 652.9 | +652.9 |
| 207 | 4x4 +2w/o | 0.103 | 0.107 | 608.2 | +608.2 |
| freq369_36mx | freq369_36mx | 0.454 | 0.451 | 602.5 | +602.5 |
| 206 | 3x3 +1w/o | 0.167 | 0.310 | 594.4 | +594.4 |
| 221 | 1x4 | 0.190 | 0.254 | 587.3 | +587.3 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.480 | 559.0 | +559.0 |
| 204 | 3x4 +6o | 0.272 | 0.392 | 484.3 | +484.3 |
| 205 | 2x4 | 0.303 | 0.330 | 478.7 | +478.7 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.369 | 474.7 | +474.7 |
| 222 | 1x3 | 0.392 | 0.378 | 471.8 | +471.8 |
| 201 | 3x4 | 0.191 | 0.362 | 467.5 | +467.5 |
| 203 | 3x4 1/.66 | 0.211 | 0.366 | 457.9 | +457.9 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.367 | 457.1 | +457.1 |
| 224 | 3x3 | 0.426 | 0.475 | 451.9 | +451.9 |
| 202 | 3x4 1/.5 | 0.224 | 0.376 | 443.7 | +443.7 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.357 | 429.2 | +429.2 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.369 | 428.9 | +428.9 |
| freq369_39mx | freq369_39mx | 0.622 | 0.463 | 423.5 | +423.5 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.378 | 422.6 | +422.6 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.389 | 414.7 | +414.7 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.421 | 414.6 | +414.6 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.357 | 404.1 | +404.1 |
| 219 | 2x3 6/9 | 0.547 | 0.475 | 386.9 | +386.9 |
| 215 | 1x3 | 0.744 | 0.363 | 366.5 | +366.5 |
| 220 | 3x4 9/12 | 0.685 | 0.505 | 359.7 | +359.7 |
| 216 | 2x3 | 0.580 | 0.472 | 317.6 | +317.6 |
| 217 | 2x4 6/12 | 0.325 | 0.339 | 308.2 | +308.2 |
| 218 | 3x4 6/8 | 0.419 | 0.362 | 270.0 | +270.0 |
| 225 | 2x2 | 0.787 | 0.541 | 234.2 | +234.2 |
| filt3E_9L | filt3E_9L | 0.575 | 0.578 | 153.7 | +153.7 |
| filt3E_3L | filt3E_3L | 0.449 | 0.488 | 140.9 | +140.9 |
| filt0E_3L | filt0E_3L | 0.372 | 0.589 | 135.2 | +135.2 |
| filt9E_9L | filt9E_9L | 0.601 | 0.433 | 118.9 | +118.9 |
| filt6E_9L | filt6E_9L | 0.590 | 0.512 | 113.6 | +113.6 |
| filt6E_3L | filt6E_3L | 0.424 | 0.434 | 106.3 | +106.3 |
| filt6E_6L | filt6E_6L | 0.553 | 0.466 | 93.8 | +93.8 |
| filt9E_3L | filt9E_3L | 0.414 | 0.386 | 89.6 | +89.6 |
| filt3E_6L | filt3E_6L | 0.482 | 0.548 | 87.6 | +87.6 |
| filt9E_6L | filt9E_6L | 0.513 | 0.408 | 73.0 | +73.0 |
| filt0E_6L | filt0E_6L | 0.487 | 0.589 | 65.1 | +65.1 |
| filt0E_9L | filt0E_9L | 0.586 | 0.589 | 64.2 | +64.2 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.659 | 54.3 | +54.3 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.631 | 53.7 | +53.7 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.640 | 49.1 | +49.1 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.649 | 48.5 | +48.5 |

## LOOP HISTORY
(first iteration)

## Response

**Name:** Load-scaled interference learning with a strength-dependent retrieval floor

**Theory:** Learning follows the leaderboard-best mechanism, xsl_03:
- Each word splits credit over the co-present objects, and each object over the co-present words.
- Pairs gain the product of those two responsibilities, saturating toward a ceiling.
- Gain is gated down when many objects are present.
- Retrieval-induced interference prunes links to absent partners at a rate that grows with trial load.

The new mechanism is at retrieval. A word's memory strength has to rise above a fixed background level of noise before it can drive choice. Test strength is therefore the learned association plus a constant retrieval floor `fl`. A word with little accumulated strength (few repetitions, or heavy interference) is chosen almost at random among the candidates. A word with large accumulated strength is chosen almost deterministically. This makes accuracy graded in the effective strength of the memory trace, rather than switching abruptly from guessing to a ceiling. The global `lapse` parameter alone cannot do this.

**Rationale:** The best model saturates in the 12x12 filtering conditions. It predicts accuracy of about 0.589, which is the ceiling set by `lapse`, for nearly every filt cell. Humans rise gradually from 0.37 at 3 late repetitions to 0.60 at 9. The same all-or-none behaviour shows in the sparse 2x2 and 2x3 conditions: the model overshoots when there are few repetitions, and in some designs the opposite error appears. A floor that competes with learned strength makes accuracy a smooth function of repetition count, since strength grows with repetitions. It also scales with load, since interference lowers strength. This costs one extra free parameter.

```json
{
  "name": "Load-scaled interference learning with strength-dependent retrieval floor",
  "theory": "Learning follows the leaderboard-best mechanism (xsl_03): mutual-responsibility credit assignment (word-side times object-side), saturating gain, and retrieval-induced interference that prunes links to absent partners at a rate growing as a power law of trial load. The new claim is a retrieval-noise floor: at test, each candidate's strength is its learned association plus a constant background noise level, so weakly encoded words are chosen near randomly and strongly encoded words nearly deterministically. Accuracy is then graded in accumulated strength, i.e. in repetitions and interference.",
  "learn": "def learn(params, trials, n_words, n_objects):\n    d = params['d']\n    q = params['q']\n    lam = params['lam']\n    Mmax = params['Mmax']\n    fl = params['fl']\n    s = 4.0\n    c = 0.5\n    g = 0.5\n    eps = 0.1\n    M = np.zeros((n_words, n_objects))\n    for words, objs in trials:\n        words = np.asarray(words)\n        objs = np.asarray(objs)\n        nw = len(words)\n        no = len(objs)\n        enc = 1.0 / (1.0 + (no / c) ** g)\n        deff = min(0.95, d * (nw * no / 12.0) ** q)\n        ix = np.ix_(words, objs)\n        blk = M[ix].copy()\n        M[words, :] *= (1.0 - deff)\n        M[:, objs] *= (1.0 - deff)\n        st = (blk + eps) ** s\n        rw = st / st.sum(1, keepdims=True)\n        ro = st / st.sum(0, keepdims=True)\n        inc = rw * ro\n        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)\n        M[ix] = blk + lam * enc * inc * sat\n    return M + fl + 1e-6",
  "parameters": {
    "d": "[0.0, 1.0]",
    "q": "[0.0, 2.0]",
    "lam": "[0.05, 4]",
    "Mmax": "[0.3, 10]",
    "fl": "[0.0, 3.0]",
    "ch_dec": "[0.3, 6]"
  },
  "rationale": "The best leaderboard model predicts a lapse-limited ceiling of about 0.589 for nearly all filtering conditions, while humans increase gradually with repetitions (0.37 to 0.60). A strength-dependent retrieval floor turns the all-or-none learned/unlearned outcome into a graded one. Weak traces, whether from few repetitions or heavy interference in dense trials, yield near-chance choice. Strong traces yield near-deterministic choice. It adds one parameter and keeps the competitive-credit and load-scaled interference mechanisms that explain the sparse-versus-dense contrast."
}
```

## Usage

```json
{
  "input_tokens": 9297,
  "output_tokens": 5744,
  "reasoning_tokens": 4129,
  "provider": "Claude Platform on AWS"
}
```
