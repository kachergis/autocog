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
| 1 | xsl_05 | Consecutive-repetition boosted interference learning | 9 | 31077.1 | 15492.0 | 0.4177 | d=0.2088, q=0.3556, lam=0.06911, Mmax=3.588, fl=0.0007805, b=0.1948, r=1.482, ch_dec=1.332, lapse=0.2629 |
| 2 | xsl_04 | Interference learning with test-time object-claim competition | 8 | 31165.9 | 15541.6 | 0.4280 | d=0.229, q=0.4687, lam=0.07155, Mmax=1.767, fl=0.0007025, b=0.1984, ch_dec=1.325, lapse=0.2755 |
| 3 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 6 | 31863.2 | 15900.6 | 0.5101 | d=0.2836, q=0.03972, lam=0.381, Mmax=9.816, ch_dec=0.6064, lapse=0.4524 |
| 4 | xsl_02 | Two-stage verify-and-consolidate with object-driven load | 6 | 32002.0 | 15969.9 | 0.4980 | a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g |
| 5 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 32045.4 | 15996.8 | 0.5529 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 6 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 32391.7 | 16175.2 | 0.5668 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 7 | baseline | Co-occurrence counting | 1 | 39486.6 | 19738.1 | 1.3660 | lapse=0.3881 |

## LEADERBOARD THEORIES
### xsl_05: Consecutive-repetition boosted interference learning
Learning is competitive and interference-limited: each pair gains the product of its word-side and object-side responsibilities (soft mutual exclusivity and explaining away), the gain saturates, and retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words at a rate that is a power law of trial load (words x objects). In addition, learners hold a one-trial working-memory trace of the previous trial's co-occurrences. A word-object pair that is also present on the immediately preceding trial is recognised as a repeat, so it receives amplified credit (1 + r). This explains the benefit of temporal contiguity: when consecutive trials share items, repeated pairs are boosted before interference erodes them, whereas in randomly ordered designs consecutive repeats are rare and the boost is negligible. At test, an object already claimed by other words is a poorer referent: association is multiplied by the tested word's share of that object's total association, raised to a power b, plus a retrieval floor. Choice is a power-law Luce rule.

### xsl_04: Interference learning with test-time object-claim competition
Learning follows the accepted account. Credit for each pair is the product of the word-side and object-side responsibilities, gain saturates, and retrieval-induced interference prunes links to absent partners at a rate that grows as a power law of trial load. At test, people also use mutual exclusivity. An object already strongly claimed by other words is a poor referent for the word being tested. The effective strength of a candidate is therefore its learned association times the share of that object's total association that belongs to the tested word, raised to a fitted power b. A retrieval-noise floor is then added. This changes which wrong objects are chosen: wrong choices are drawn to objects that no word has claimed, and are less likely to be objects that other words own.

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

## BEST LEADERBOARD THEORY: xsl_05
**Name:** Consecutive-repetition boosted interference learning

**Description:** Learning is competitive and interference-limited: each pair gains the product of its word-side and object-side responsibilities (soft mutual exclusivity and explaining away), the gain saturates, and retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words at a rate that is a power law of trial load (words x objects). In addition, learners hold a one-trial working-memory trace of the previous trial's co-occurrences. A word-object pair that is also present on the immediately preceding trial is recognised as a repeat, so it receives amplified credit (1 + r). This explains the benefit of temporal contiguity: when consecutive trials share items, repeated pairs are boosted before interference erodes them, whereas in randomly ordered designs consecutive repeats are rare and the boost is negligible. At test, an object already claimed by other words is a poorer referent: association is multiplied by the tested word's share of that object's total association, raised to a power b, plus a retrieval floor. Choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.05, 4]
- Mmax: [0.3, 10]
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
    Mmax = params['Mmax']
    fl = params['fl']
    b = params['b']
    r = params['r']
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
        M[words, :] *= (1.0 - deff)
        M[:, objs] *= (1.0 - deff)
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

Per-condition fit, worst first:
| label | condition | human acc | model acc | NLL | NLL |
|---|---|---|---|---|---|
| 301 | 2x2 5x 24AFC | 0.514 | 0.646 | 815.3 | +815.3 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.468 | 737.7 | +737.7 |
| 223 | 4x4 | 0.312 | 0.300 | 642.9 | +642.9 |
| freq369_36mx | freq369_36mx | 0.454 | 0.453 | 587.8 | +587.8 |
| 206 | 3x3 +1w/o | 0.167 | 0.294 | 585.7 | +585.7 |
| 207 | 4x4 +2w/o | 0.103 | 0.098 | 580.0 | +580.0 |
| 221 | 1x4 | 0.190 | 0.294 | 577.1 | +577.1 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.471 | 548.3 | +548.3 |
| 222 | 1x3 | 0.392 | 0.500 | 483.9 | +483.9 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.331 | 471.6 | +471.6 |
| 205 | 2x4 | 0.303 | 0.317 | 470.9 | +470.9 |
| 204 | 3x4 +6o | 0.272 | 0.335 | 461.7 | +461.7 |
| 224 | 3x3 | 0.426 | 0.472 | 457.8 | +457.8 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.349 | 453.6 | +453.6 |
| 201 | 3x4 | 0.191 | 0.317 | 452.2 | +452.2 |
| 203 | 3x4 1/.66 | 0.211 | 0.318 | 442.1 | +442.1 |
| freq369_39mx | freq369_39mx | 0.622 | 0.445 | 441.6 | +441.6 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.295 | 436.4 | +436.4 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.301 | 434.7 | +434.7 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.465 | 431.0 | +431.0 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.326 | 427.3 | +427.3 |
| 202 | 3x4 1/.5 | 0.224 | 0.331 | 426.7 | +426.7 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.424 | 400.4 | +400.4 |
| 220 | 3x4 9/12 | 0.685 | 0.442 | 395.4 | +395.4 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.465 | 385.7 | +385.7 |
| 219 | 2x3 6/9 | 0.547 | 0.520 | 380.8 | +380.8 |
| 216 | 2x3 | 0.580 | 0.548 | 311.2 | +311.2 |
| 217 | 2x4 6/12 | 0.325 | 0.355 | 307.6 | +307.6 |
| 215 | 1x3 | 0.744 | 0.518 | 283.0 | +283.0 |
| 218 | 3x4 6/8 | 0.419 | 0.359 | 268.1 | +268.1 |
| 225 | 2x2 | 0.787 | 0.689 | 196.7 | +196.7 |
| filt9E_9L | filt9E_9L | 0.601 | 0.428 | 127.5 | +127.5 |
| filt3E_3L | filt3E_3L | 0.449 | 0.459 | 103.7 | +103.7 |
| filt6E_9L | filt6E_9L | 0.590 | 0.469 | 90.3 | +90.3 |
| filt0E_3L | filt0E_3L | 0.372 | 0.490 | 85.6 | +85.6 |
| filt3E_9L | filt3E_9L | 0.575 | 0.563 | 79.7 | +79.7 |
| filt9E_6L | filt9E_6L | 0.513 | 0.415 | 77.8 | +77.8 |
| filt6E_3L | filt6E_3L | 0.424 | 0.402 | 73.5 | +73.5 |
| filt9E_3L | filt9E_3L | 0.414 | 0.375 | 70.2 | +70.2 |
| filt0E_9L | filt0E_9L | 0.586 | 0.646 | 70.0 | +70.0 |
| filt0E_6L | filt0E_6L | 0.487 | 0.599 | 67.8 | +67.8 |
| filt6E_6L | filt6E_6L | 0.553 | 0.455 | 65.8 | +65.8 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.595 | 63.7 | +63.7 |
| filt3E_6L | filt3E_6L | 0.482 | 0.540 | 63.5 | +63.5 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.697 | 56.7 | +56.7 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.671 | 53.7 | +53.7 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.592 | 47.2 | +47.2 |

## CURRENT BEST CANDIDATE (this loop) — BIC 31093.9, params d=0.2258, q=0.4937, lam=0.06846, Mmax=9.398, fl=0.0007459, b=0.1973, r=0.8932, h=28.48, ch_dec=1.331, lapse=0.2648
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

Per-condition fit, worst first:
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

## LOOP HISTORY
- iter 0: Consolidation-protected boosted interference learning — BIC 31093.9 — ACCEPTED
  critic: Drop h and the protection term, which returns to xsl_05's 9 parameters. Then fix how load is computed and how strongly it hits encoding. (a) Split load into separate word and object exponents: deff = min(0.95, d * (nw/3)^q_w * (no/3)^q_o), with q_o free in [0,3] and q_w free in [0,2]. Alternatively keep a single q but make load depend mainly on no. This should push 1x4 and 3x4 down toward 0.19 without hurting 1x3. (b) Make the encoding exponent g in enc = 1/(1+(no/c)^g) a fitted parameter instead of the fixed 0.5, so encoding falls steeply with objects per trial. (c) To help high-frequency words, let the sharpness exponent s (currently fixed at 4) be fitted, or make sat depend on the word's own evidence, so that words reaching a few co-occurrences lock in. Fit the added parameters only if the NLL gain is more than about 5 per parameter, since each costs 10.3 in BIC.
- iter 1: Object-driven interference with steep encoding loss — BIC 31729.3 — REJECTED
  critic: Revert to the xsl_05 backbone and make one small change, instead of the steep-encoding variant. (a) Remove the enc factor (g, c); set it to 1, or a mild fixed 1/sqrt(no/3) at most. Restore the modest bonus r (about 1-3, e.g. bound [0,4]) so the consecutive-repeat boost cannot dominate. This restores accumulation of evidence over repeated co-occurrences, which is what drives the rising 3L->9L accuracy in the filt conditions. (b) Keep only the split interference exponents, deff = min(0.95, d*(nw/3)^qw*(no/3)^qo), as the single new structure. Make qo free and qw free or tied to a fraction of qo. (c) To stop pruning from overwhelming a correct pair on low-load 2x2 trials, scale the pruning of a pair's row and column links by (1 - sat), where sat = M/Mmax. Strongly established pairs then resist interference, while weak spurious pairs still decay, and accuracy rises with trial count as in the filtering data. This replaces the g and r flexibility with one mechanism. Adopt it only if the NLL gain over xsl_05 exceeds roughly 5 per added parameter.

## Response


<invoke name="Python">
<parameter name="code">print(1)</parameter>
</invoke>
does not exist; just output JSON.


<invoke name="Python">
<parameter name="code">print(2)</parameter>
</invoke>
```json
{
  "name": "Decaying-trace boosted interference learning (no saturation)",
  "theory": "Learning is competitive and interference-limited. Each pair gains the product of its word-side and object-side responsibilities, which gives soft mutual exclusivity and explaining away. Retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words, at a rate that is a power law of trial load (words x objects). Gain is unsaturated: evidence simply accumulates, and the interference process alone limits strength. Working memory holds a graded, decaying trace of recent co-occurrences instead of a single one-trial trace. A pair seen on the previous trial has trace 1, one seen two trials back has 0.4, and so on. A pair's credit is amplified by 1 + r*trace, so recently repeated pairs are boosted before interference erodes them. This is a general recency-of-repeat benefit, and it explains the temporal-contiguity and overlap designs. At test, an object already claimed by other words is a poorer referent: association is multiplied by the tested word's share of that object's total association, raised to a power b, plus a retrieval floor. Choice is a power-law Luce rule.",
  "learn": "def learn(params, trials, n_words, n_objects):\n    d = params['d']\n    q = params['q']\n    lam = params['lam']\n    fl = params['fl']\n    b = params['b']\n    r = params['r']\n    s = 4.0\n    c = 0.5\n    g = 0.5\n    eps = 0.1\n    rho = 0.4\n    M = np.zeros((n_words, n_objects))\n    trace = np.zeros((n_words, n_objects))\n    for words, objs in trials:\n        words = np.asarray(words)\n        objs = np.asarray(objs)\n        nw = len(words)\n        no = len(objs)\n        enc = 1.0 / (1.0 + (no / c) ** g)\n        deff = min(0.95, d * (nw * no / 12.0) ** q)\n        ix = np.ix_(words, objs)\n        blk = M[ix].copy()\n        M[words, :] *= (1.0 - deff)\n        M[:, objs] *= (1.0 - deff)\n        st = (blk + eps) ** s\n        rw = st / st.sum(1, keepdims=True)\n        ro = st / st.sum(0, keepdims=True)\n        inc = rw * ro\n        bonus = 1.0 + r * trace[ix]\n        M[ix] = blk + lam * enc * inc * bonus\n        trace = trace * rho\n        trace[ix] = 1.0\n    col = M.sum(0, keepdims=True)\n    share = (M + 1e-9) / (col + 0.05)\n    out = M * share ** b\n    return out + fl + 1e-6\n",
  "parameters": {
    "d": "[0.0, 1.0]",
    "q": "[0.0, 2.0]",
    "lam": "[0.05, 6]",
    "fl": "[0.0, 3.0]",
    "b": "[0.0, 3.0]",
    "r": "[0.0, 30]",
    "ch_dec": "[0.3, 6]"
  },
  "rationale": "In the fits of xsl_05 and the current best, the saturation ceiling (Mmax) and the extra consolidation threshold (h) barely bind, because association strengths stay far below the ceiling. Dropping them saves free parameters (about 10 BIC each) at little cost in fit. The one-trial repeat trace is replaced with a graded decaying trace, so the contiguity and overlap conditions (1 overlap per 2 or 3 trials, temporal contiguity) get a smoother boost that reaches two or more trials back, with no extra free parameter. Interference, competitive responsibility, the test-time claim competition and the Luce power rule keep the structure that fitted the load, frequency and filtering conditions best."
}
```

## Usage

```json
{
  "input_tokens": 14546,
  "output_tokens": 3454,
  "reasoning_tokens": 1998,
  "provider": "Claude Platform on AWS"
}
```
