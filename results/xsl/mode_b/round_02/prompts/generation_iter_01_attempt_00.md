# generation_iter_01_attempt_00

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
| 1 | xsl_02 | Two-stage verify-and-consolidate with object-driven load | 6 | 32002.0 | 15969.9 | 0.4980 | a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g |
| 2 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 32045.4 | 15996.8 | 0.5529 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 3 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 32391.7 | 16175.2 | 0.5668 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 4 | baseline | Co-occurrence counting | 1 | 39486.6 | 19738.1 | 1.3660 | lapse=0.3881 |

## LEADERBOARD THEORIES
### xsl_02: Two-stage verify-and-consolidate with object-driven load
Each word holds a tentative referent hypothesis (propose-but-verify), tracked as an exact expected distribution over learners, passing through three stages: freshly proposed, confirmed once, and consolidated (durable). A hypothesis is only consolidated after a second confirmation. Unconsolidated hypotheses are forgotten whenever the word is heard, with a probability that rises steeply (power-law) with the number of objects on the trial and with the square root of the number of words, and fresh hypotheses are more fragile than confirmed ones. A word with no surviving hypothesis, or whose hypothesis object is absent, re-proposes among the present objects with probability a, strongly preferring objects already claimed by no other word (mutual exclusivity). Unlearned words are guessed uniformly; choice is a power-law Luce rule.

### xsl_01: Interference-limited mutual-responsibility learning
On each trial the learner divides credit for each word among the co-present objects, and for each object among the co-present words, in proportion to sharpened current strength plus a small prior. Each pair is strengthened by the product of its word-side and object-side responsibilities, which gives soft mutual exclusivity and explaining away. Forgetting is interference-based rather than global: only associations involving a word or object that is currently on the trial decay. Knowledge of items not currently competing persists, so extra exposure and long training help, while dense competition (many words and objects per trial) erodes strength. Test choice is a power-law Luce competition.

### kachergis: Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012)
Learners distribute a limited amount of attention over the word-object pairs on each trial. Attention is drawn both to familiar pairs (already strongly associated) and to uncertain stimuli (words or objects whose associations are still spread over many partners, measured by entropy). The attended share of a fixed associative weight is added to each pair, and all associations decay slightly on every trial. Familiarity produces a rich-get-richer bias; uncertainty pulls attention toward stimuli that have no strong partner yet, which together implement a soft mutual-exclusivity bias.

### baseline: Co-occurrence counting
Learners simply count how often each word and object occur together across trials, with no attention, competition or forgetting.

## BEST LEADERBOARD THEORY: xsl_02
**Name:** Two-stage verify-and-consolidate with object-driven load

**Description:** Each word holds a tentative referent hypothesis (propose-but-verify), tracked as an exact expected distribution over learners, passing through three stages: freshly proposed, confirmed once, and consolidated (durable). A hypothesis is only consolidated after a second confirmation. Unconsolidated hypotheses are forgotten whenever the word is heard, with a probability that rises steeply (power-law) with the number of objects on the trial and with the square root of the number of words, and fresh hypotheses are more fragile than confirmed ones. A word with no surviving hypothesis, or whose hypothesis object is absent, re-proposes among the present objects with probability a, strongly preferring objects already claimed by no other word (mutual exclusivity). Unlearned words are guessed uniformly; choice is a power-law Luce rule.

**Parameters:**
- a: [0.05, 1]
- f: [0, 0.5]
- k: [0, 1]
- g: [0.5, 3]
- ch_dec: [0.3, 6]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    a = params['a']
    f = params['f']
    k = params['k']
    g = params['g']
    me = 20.0
    h1 = np.zeros((n_words, n_objects))
    h2 = np.zeros((n_words, n_objects))
    S = np.zeros((n_words, n_objects))
    for words, objects in trials:
        words = np.asarray(words)
        objects = np.asarray(objects)
        nw = len(words)
        no = len(objects)
        load = (nw * 0.5) ** 0.5 * (no / 2.0) ** g
        fe = 1.0 - (1.0 - f) ** load
        h1w = h1[words] * (1.0 - fe)
        h2w = h2[words] * (1.0 - 0.5 * fe)
        Sw = S[words]
        p1 = h1w[:, objects]
        p2 = h2w[:, objects]
        tot = h1w.sum(1) + h2w.sum(1) + Sw.sum(1)
        none = np.clip(1.0 - tot, 0.0, 1.0)
        habs = (h1w.sum(1) - p1.sum(1)) + (h2w.sum(1) - p2.sum(1))
        moved = a * (none + habs)
        col = (h1 + h2 + S).sum(0)
        own = (h1[words] + h2[words] + Sw)[:, objects]
        other = np.maximum(col[objects][None, :] - own, 0.0)
        wts = 1.0 / (1.0 + me * other)
        wts = wts / wts.sum(1, keepdims=True)
        h1all = h1w * (1.0 - a)
        h1all[:, objects] = moved[:, None] * wts
        h2all = h2w * (1.0 - a)
        h2all[:, objects] = p2 * (1.0 - k) + p1
        Sw = Sw.copy()
        Sw[:, objects] += k * p2
        h1[words] = h1all
        h2[words] = h2all
        S[words] = Sw
    tot = h1.sum(1) + h2.sum(1) + S.sum(1)
    none_f = np.clip(1.0 - tot, 0.0, 1.0)
    return h1 + h2 + S + none_f[:, None] / n_objects + 1e-6
```

Per-condition fit, worst first:
| label | condition | human acc | model acc | NLL | NLL |
|---|---|---|---|---|---|
| 301 | 2x2 5x 24AFC | 0.514 | 0.579 | 785.3 | +785.3 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.455 | 762.7 | +762.7 |
| 223 | 4x4 | 0.312 | 0.332 | 659.5 | +659.5 |
| 206 | 3x3 +1w/o | 0.167 | 0.350 | 637.5 | +637.5 |
| 207 | 4x4 +2w/o | 0.103 | 0.140 | 630.6 | +630.6 |
| 221 | 1x4 | 0.190 | 0.377 | 615.1 | +615.1 |
| freq369_36mx | freq369_36mx | 0.454 | 0.432 | 597.0 | +597.0 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.462 | 546.4 | +546.4 |
| 205 | 2x4 | 0.303 | 0.368 | 485.3 | +485.3 |
| 204 | 3x4 +6o | 0.272 | 0.374 | 484.2 | +484.2 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.334 | 481.5 | +481.5 |
| 222 | 1x3 | 0.392 | 0.503 | 480.4 | +480.4 |
| 201 | 3x4 | 0.191 | 0.357 | 470.8 | +470.8 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.329 | 464.2 | +464.2 |
| 224 | 3x3 | 0.426 | 0.467 | 459.6 | +459.6 |
| 203 | 3x4 1/.66 | 0.211 | 0.363 | 453.4 | +453.4 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.312 | 452.1 | +452.1 |
| freq369_39mx | freq369_39mx | 0.622 | 0.436 | 444.0 | +444.0 |
| 202 | 3x4 1/.5 | 0.224 | 0.372 | 440.9 | +440.9 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.307 | 436.6 | +436.6 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.312 | 435.1 | +435.1 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.317 | 431.6 | +431.6 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.334 | 431.3 | +431.3 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.393 | 409.6 | +409.6 |
| 219 | 2x3 6/9 | 0.547 | 0.503 | 382.0 | +382.0 |
| 220 | 3x4 9/12 | 0.685 | 0.468 | 378.9 | +378.9 |
| 216 | 2x3 | 0.580 | 0.485 | 317.9 | +317.9 |
| 217 | 2x4 6/12 | 0.325 | 0.339 | 308.9 | +308.9 |
| 215 | 1x3 | 0.744 | 0.500 | 289.8 | +289.8 |
| 218 | 3x4 6/8 | 0.419 | 0.345 | 273.1 | +273.1 |
| 225 | 2x2 | 0.787 | 0.604 | 215.1 | +215.1 |
| filt3E_9L | filt3E_9L | 0.575 | 0.602 | 133.3 | +133.3 |
| filt3E_3L | filt3E_3L | 0.449 | 0.372 | 120.7 | +120.7 |
| filt6E_9L | filt6E_9L | 0.590 | 0.587 | 111.1 | +111.1 |
| filt6E_3L | filt6E_3L | 0.424 | 0.382 | 110.3 | +110.3 |
| filt9E_9L | filt9E_9L | 0.601 | 0.508 | 94.4 | +94.4 |
| filt9E_3L | filt9E_3L | 0.414 | 0.360 | 91.9 | +91.9 |
| filt6E_6L | filt6E_6L | 0.553 | 0.504 | 82.1 | +82.1 |
| filt3E_6L | filt3E_6L | 0.482 | 0.515 | 73.7 | +73.7 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.563 | 69.4 | +69.4 |
| filt0E_3L | filt0E_3L | 0.372 | 0.349 | 65.6 | +65.6 |
| filt0E_9L | filt0E_9L | 0.586 | 0.607 | 64.9 | +64.9 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.628 | 62.9 | +62.9 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.618 | 59.6 | +59.6 |
| filt9E_6L | filt9E_6L | 0.513 | 0.446 | 59.4 | +59.4 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.564 | 55.6 | +55.6 |
| filt0E_6L | filt0E_6L | 0.487 | 0.516 | 54.5 | +54.5 |

## CURRENT BEST CANDIDATE (this loop) — BIC 32138.6, params c=7.671, g=0.5, s=2.083, d=0.1394, lam=0.3695, ch_dec=1.074, lapse=0.4622 — at bound: g
**Name:** Load-gated encoding with competitive responsibility and interference decay

**Description:** Learning on a trial is limited by how many objects must be encoded. The share of a trial's word-object evidence that is actually encoded falls steeply as the number of candidate objects rises beyond a capacity c, following a logistic-in-power law with exponent g. The evidence that is encoded is divided competitively. Each word splits credit over the co-present objects, and each object splits credit over the co-present words, both in proportion to sharpened current strength plus a prior. Each pair gains the product of the two responsibilities, which gives soft mutual exclusivity and explaining away. Forgetting is interference-based: only associations of words or objects on the current trial decay. Test choice is a power-law Luce rule.

**Parameters:**
- c: [0.5, 8]
- g: [0.5, 5]
- s: [0, 4]
- d: [0, 0.3]
- lam: [0.1, 4]
- ch_dec: [0.3, 6]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    c = params['c']
    g = params['g']
    s = params['s']
    d = params['d']
    lam = params['lam']
    eps = 0.1
    M = np.zeros((n_words, n_objects))
    for words, objs in trials:
        words = np.asarray(words)
        objs = np.asarray(objs)
        no = len(objs)
        enc = 1.0 / (1.0 + (no / c) ** g)
        M[words, :] *= (1.0 - d)
        M[:, objs] *= (1.0 - d)
        sub = M[np.ix_(words, objs)]
        st = (sub + eps) ** s
        rw = st / st.sum(1, keepdims=True)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        M[np.ix_(words, objs)] = sub + lam * enc * inc
    return M + 1e-6
```

Per-condition fit, worst first:
| label | condition | human acc | model acc | NLL | NLL minus xsl_02's |
|---|---|---|---|---|---|
| filt9E_3L | filt9E_3L | 0.414 | 0.498 | 160.9 | +69.0 |
| filt0E_3L | filt0E_3L | 0.372 | 0.580 | 129.5 | +63.8 |
| filt9E_9L | filt9E_9L | 0.601 | 0.516 | 132.5 | +38.1 |
| filt9E_6L | filt9E_6L | 0.513 | 0.509 | 88.6 | +29.2 |
| filt6E_3L | filt6E_3L | 0.424 | 0.481 | 136.8 | +26.5 |
| filt6E_6L | filt6E_6L | 0.553 | 0.491 | 107.8 | +25.6 |
| 215 | 1x3 | 0.744 | 0.440 | 314.7 | +24.9 |
| filt3E_3L | filt3E_3L | 0.449 | 0.485 | 143.4 | +22.7 |
| filt6E_9L | filt6E_9L | 0.590 | 0.496 | 130.2 | +19.1 |
| 225 | 2x2 | 0.787 | 0.545 | 231.9 | +16.8 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.479 | 560.5 | +14.1 |
| filt3E_6L | filt3E_6L | 0.482 | 0.494 | 84.9 | +11.3 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.424 | 419.3 | +9.7 |
| filt0E_6L | filt0E_6L | 0.487 | 0.580 | 63.2 | +8.6 |
| filt3E_9L | filt3E_9L | 0.575 | 0.498 | 141.4 | +8.1 |
| freq369_36mx | freq369_36mx | 0.454 | 0.459 | 603.3 | +6.3 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.330 | 434.4 | +3.2 |
| 219 | 2x3 6/9 | 0.547 | 0.496 | 384.0 | +2.0 |
| 203 | 3x4 1/.66 | 0.211 | 0.361 | 453.1 | -0.3 |
| filt0E_9L | filt0E_9L | 0.586 | 0.580 | 64.2 | -0.6 |
| 217 | 2x4 6/12 | 0.325 | 0.373 | 306.6 | -2.3 |
| 204 | 3x4 +6o | 0.272 | 0.377 | 481.7 | -2.6 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.607 | 56.9 | -2.7 |
| 201 | 3x4 | 0.191 | 0.361 | 467.3 | -3.6 |
| 202 | 3x4 1/.5 | 0.224 | 0.367 | 437.2 | -3.7 |
| 218 | 3x4 6/8 | 0.419 | 0.355 | 269.4 | -3.7 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.332 | 426.1 | -5.5 |
| 216 | 2x3 | 0.580 | 0.492 | 312.3 | -5.6 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.636 | 57.2 | -5.7 |
| 205 | 2x4 | 0.303 | 0.377 | 479.4 | -5.9 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.602 | 48.8 | -6.8 |
| 220 | 3x4 9/12 | 0.685 | 0.477 | 372.1 | -6.8 |
| 224 | 3x3 | 0.426 | 0.478 | 452.3 | -7.3 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.328 | 456.3 | -8.0 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.534 | 777.3 | -8.1 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.596 | 59.8 | -9.7 |
| 223 | 4x4 | 0.312 | 0.326 | 647.6 | -11.8 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.331 | 468.4 | -13.1 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.322 | 437.7 | -14.4 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.347 | 421.9 | -14.7 |
| 222 | 1x3 | 0.392 | 0.438 | 465.5 | -14.9 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.322 | 419.2 | -15.9 |
| freq369_39mx | freq369_39mx | 0.622 | 0.460 | 427.7 | -16.3 |
| 221 | 1x4 | 0.190 | 0.329 | 589.7 | -25.4 |
| 207 | 4x4 +2w/o | 0.103 | 0.123 | 604.4 | -26.2 |
| 206 | 3x3 +1w/o | 0.167 | 0.346 | 602.4 | -35.1 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.484 | 703.5 | -59.2 |

## LOOP HISTORY
- iter 0: Load-gated encoding with competitive responsibility and interference decay — BIC 32138.6 — ACCEPTED
  critic: Replace the stateless additive update with a two-stage, hypothesis-driven consolidation mechanism: (a) a fast, weak, noisy trace with a per-trial gain, and (b) a nonlinear consolidation step. Whenever a word's current top-strength object is co-present on a trial, add a strong boost proportional to the margin (or sharpened share) of that pair. This makes strength grow superlinearly and gives the snowballing that the model lacks in easy conditions (1x3, 2x2, 3x4 9/12) and in repeated-pair conditions. Make forgetting a function of intervening trials, such as decay applied per trial to non-present pairs at a small rate, and not only decay on the current trial's items. This lets early filler trials fade while late trials accumulate, which fixes the filt*E_*L pattern. Keep the load gate on object count, but make it multiplicative on the new-evidence term only (not on consolidation). Fix g at its bound (e.g. 1) to drop one parameter. Keep ch_dec free, and consider fixing eps and the sharpening s to save parameters. The target is the xsl_02 structure with a cleaner parameterisation.

## Response


<invoke name="bash">
<parameter name="command">echo skip</parameter>
</invoke>

<invoke name="bash">
<parameter name="command">echo skip</parameter>
</invoke>

{"name":"Absent-alternative interference with protected co-present block","theory":"Learners encode each trial's word-object evidence competitively: each word splits credit over co-present objects and each object over co-present words (sharpened strength plus prior), and each pair gains the product of the two responsibilities, giving soft mutual exclusivity and explaining away. Encoding is mildly gated by the number of objects on the trial. Forgetting is retrieval-induced interference: when a word is heard, its associations to objects that are NOT present decay, and when an object is seen, its associations to words NOT present decay. Pairs that are co-present are protected from this competitive forgetting and suffer only a separate, weaker within-trial decay (d2). Repeated consistent pairings therefore accumulate and snowball with repetition (late-filler and frequency effects) while spurious associations from other contexts are pruned. Test choice is a power-law Luce rule.","learn":"def learn(params, trials, n_words, n_objects):\n    c = params['c']\n    s = params['s']\n    d = params['d']\n    d2 = params['d2']\n    lam = params['lam']\n    g = 0.5\n    eps = 0.1\n    M = np.zeros((n_words, n_objects))\n    for words, objs in trials:\n        words = np.asarray(words)\n        objs = np.asarray(objs)\n        no = len(objs)\n        enc = 1.0 / (1.0 + (no / c) ** g)\n        ix = np.ix_(words, objs)\n        blk = M[ix].copy()\n        M[words, :] *= (1.0 - d)\n        M[:, objs] *= (1.0 - d)\n        sub = blk * (1.0 - d2)\n        st = (sub + eps) ** s\n        rw = st / st.sum(1, keepdims=True)\n        ro = st / st.sum(0, keepdims=True)\n        inc = rw * ro\n        M[ix] = sub + lam * enc * inc\n    return M + 1e-6\n","parameters":{"c":"[0.5, 8]","s":"[0, 4]","d":"[0, 0.5]","d2":"[0, 0.5]","lam":"[0.05, 4]","ch_dec":"[0.3, 6]"},"rationale":"The current best saturates at ~0.58 in all filt*_*L conditions because decay applied to every item on every appearance balances gain at a low equilibrium, whereas humans improve steadily with repetitions (0.37 -> 0.49 -> 0.59) and are very good in easy designs (1x3, 2x2). Separating decay of absent alternatives (competitive interference, parameter d) from decay of the co-present block (d2) nests the previous model (d2 about 1-(1-d)^2) but lets repeated co-occurrences accumulate, giving snowballing with frequency and allowing the late-vs-early filler pattern, while dense-trial erosion is still captured through competition and d2. Fixing g at its bound saves a parameter."}

## Usage

```json
{
  "input_tokens": 12429,
  "output_tokens": 3675,
  "reasoning_tokens": 2559,
  "provider": "Claude Platform on AWS"
}
```
