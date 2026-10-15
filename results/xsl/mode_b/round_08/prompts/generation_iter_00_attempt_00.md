# generation_iter_00_attempt_00

## System Prompt

You are a cognitive scientist and expert Python programmer working on theories of cross-situational word learning. Your job is to propose a NEW theory, with an executable learner, that predicts the human data better (lower held-out NLL) than every theory on the leaderboard.

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

## ACCEPT GATE
You work in a loop. Each proposal is fitted and cross-validated; it is ACCEPTED only if its held-out NLL is strictly lower than that of the best accepted candidate so far in this loop. `## CURRENT BEST CANDIDATE` always shows the best accepted candidate, never a rejected one, so build on it. A critic comments on each candidate; the `## LOOP HISTORY` block shows which of its suggestions helped (ACCEPTED) and which did not (REJECTED).

Leaderboard theories are for inspiration only: do not resubmit one of them or a cosmetic variant. A new mechanism that fits better is the goal.

## OUTPUT
You have no tools and cannot run code: do not write tool calls, XML tags or `<invoke>` blocks. Reply with the JSON object only — no prose before or after it.
Return one JSON object with these fields:
- name: Short name for the theory (a few words).
- theory: The theoretical claim this theory makes about how people learn words across situations.
- learn: Python source defining def learn(params, trials, n_words, n_objects) -> np.ndarray, returning the final (n_words, n_objects) association matrix. `np` is available.
- parameters: Free parameters as intervals, e.g. {"X": "[0.001, 5]"}; a bare number pins a value.
- rationale: Why this theory should fit human cross-situational learning better than the alternatives.


## User Prompt

## GOAL
Propose a theory of cross-situational word learning that achieves a lower held-out NLL than every theory on the leaderboard. Study which conditions the best theory predicts worst and what those designs have in common, and target that mechanism.

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

## LEADERBOARD (ranked by held-out NLL; lower is better)
| rank | label | name | k | held-out NLL (score) | held-out wSSE | BIC (full fit) | fitted parameters (full fit) |
|---|---|---|---|---|---|---|---|
| 1 | xsl_08 | Decaying-trace contiguity interference learning | 12 | 15496.1 | 0.4334 | 30691.9 | d=0.1936, q=0.05846, lam=1.136, Mmax=4.123, fl=0.01979, b=0.1702, s=0.7385, r=1.17, ch_dec=1.162, p=1.272, rho=0.3678, lapse=0.2 |
| 2 | xsl_07 | Soft-sharpness interference learning with contiguity boost | 10 | 15529.5 | 0.4394 | 30721.2 | d=0.2157, q=0.3407, lam=1.53, Mmax=7.773, fl=0.03068, b=0.04131, s=0.7391, r=0.9038, ch_dec=1.165, lapse=0.1924 |
| 3 | xsl_04 | Interference learning with test-time object-claim competition | 8 | 15879.4 | 0.5113 | 31165.9 | d=0.229, q=0.4687, lam=0.07155, Mmax=1.767, fl=0.0007025, b=0.1984, ch_dec=1.325, lapse=0.2755 |
| 4 | xsl_05 | Consecutive-repetition boosted interference learning | 9 | 16040.9 | 0.5465 | 31077.1 | d=0.2088, q=0.3556, lam=0.06911, Mmax=3.588, fl=0.0007805, b=0.1948, r=1.482, ch_dec=1.332, lapse=0.2629 |
| 5 | xsl_06 | Consolidation-protected boosted interference learning | 10 | 16095.6 | 0.5452 | 31093.9 | d=0.2258, q=0.4937, lam=0.06846, Mmax=9.398, fl=0.0007459, b=0.1973, r=0.8932, h=28.48, ch_dec=1.331, lapse=0.2648 |
| 6 | xsl_02 | Two-stage verify-and-consolidate with object-driven load | 6 | 16104.9 | 0.5384 | 32002.0 | a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g |
| 7 | xsl_03 | Load-scaled absent-alternative interference with saturating competitive encoding | 6 | 16128.6 | 0.5626 | 31863.2 | d=0.2836, q=0.03972, lam=0.381, Mmax=9.816, ch_dec=0.6064, lapse=0.4524 |
| 8 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 16146.9 | 0.5900 | 32045.4 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 9 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 16274.8 | 0.5976 | 32391.7 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 10 | baseline | Co-occurrence counting | 1 | 19786.1 | 1.3811 | 39486.6 | lapse=0.3881 |

## LEADERBOARD THEORIES
### xsl_08: Decaying-trace contiguity interference learning
Learning is competitive and interference-limited. Credit for a pair is the product of word-side and object-side responsibilities (power-sharpened shares), and the gain saturates and is gated by the number of visible objects. Retrieval-induced interference prunes links to absent partners, with a rate set by the number of objects and by words x objects load. The new claim is that the memory of recent co-occurrence is a graded, fading trace rather than a one-trial flag. A pair seen on the previous trial is strongly marked, one seen two or three trials ago is marked more weakly, and the mark decays geometrically. Marked pairs receive amplified credit (1 + r*trace), which is how temporal contiguity helps. This lets designs with overlap every 2 or 3 trials differ from designs with immediate repeats, and from random orderings. At test, objects claimed by other words are discounted (share^b), a retrieval floor is added, and choice is a power-law Luce rule.

### xsl_07: Soft-sharpness interference learning with contiguity boost
Learning is competitive and interference-limited. Each word splits credit over the co-present objects, and each object over the co-present words, in proportion to current strength raised to a fitted exponent s. The exponent is allowed to fall below 1, so competition can be soft as well as winner-take-all. A pair gains the product of the two responsibilities, which gives soft mutual exclusivity and explaining away. The gain saturates and is weakly gated down as more objects are visible. Retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words, at a rate that is a power law of trial load (words x objects). A one-trial memory trace marks pairs that co-occurred on the immediately preceding trial, and such repeated pairs receive amplified credit (1+r). This is how temporal contiguity helps. At test, an object already claimed by other words is a poorer referent: association is multiplied by the tested word's share of that object's total association, raised to b. A small retrieval floor is added, and choice is a power-law Luce rule.

### xsl_04: Interference learning with test-time object-claim competition
Learning follows the accepted account. Credit for each pair is the product of the word-side and object-side responsibilities, gain saturates, and retrieval-induced interference prunes links to absent partners at a rate that grows as a power law of trial load. At test, people also use mutual exclusivity. An object already strongly claimed by other words is a poor referent for the word being tested. The effective strength of a candidate is therefore its learned association times the share of that object's total association that belongs to the tested word, raised to a fitted power b. A retrieval-noise floor is then added. This changes which wrong objects are chosen: wrong choices are drawn to objects that no word has claimed, and are less likely to be objects that other words own.

### xsl_05: Consecutive-repetition boosted interference learning
Learning is competitive and interference-limited: each pair gains the product of its word-side and object-side responsibilities (soft mutual exclusivity and explaining away), the gain saturates, and retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words at a rate that is a power law of trial load (words x objects). In addition, learners hold a one-trial working-memory trace of the previous trial's co-occurrences. A word-object pair that is also present on the immediately preceding trial is recognised as a repeat, so it receives amplified credit (1 + r). This explains the benefit of temporal contiguity: when consecutive trials share items, repeated pairs are boosted before interference erodes them, whereas in randomly ordered designs consecutive repeats are rare and the boost is negligible. At test, an object already claimed by other words is a poorer referent: association is multiplied by the tested word's share of that object's total association, raised to a power b, plus a retrieval floor. Choice is a power-law Luce rule.

### xsl_06: Consolidation-protected boosted interference learning
Learning is competitive and interference-limited. Each pair gains the product of its word-side and object-side responsibilities (soft mutual exclusivity and explaining away), and the gain saturates. Retrieval-induced interference prunes a word's links to absent objects and an object's links to absent words, at a rate that is a power law of trial load. A one-trial working-memory trace boosts pairs that repeat on consecutive trials (temporal contiguity). The new mechanism is consolidation: interference is not equally damaging to every link. Weak, tentative associations are fully exposed to it, while associations that have already been built up become progressively protected from it. Protection falls as 1/(1+(M/h)^2), so strength acts as a soft consolidation threshold. Words seen often enough to cross the threshold become durable and snowball, as with 9-repetition words and sparse 1-word trials. Words that never cross it keep being eroded under dense load. At test, an object claimed by other words is a poorer referent, with a share^b competition term plus a retrieval floor, and choice is a power-law Luce rule.

### xsl_02: Two-stage verify-and-consolidate with object-driven load
Each word holds a tentative referent hypothesis (propose-but-verify), tracked as an exact expected distribution over learners, passing through three stages: freshly proposed, confirmed once, and consolidated (durable). A hypothesis is only consolidated after a second confirmation. Unconsolidated hypotheses are forgotten whenever the word is heard, with a probability that rises steeply (power-law) with the number of objects on the trial and with the square root of the number of words, and fresh hypotheses are more fragile than confirmed ones. A word with no surviving hypothesis, or whose hypothesis object is absent, re-proposes among the present objects with probability a, strongly preferring objects already claimed by no other word (mutual exclusivity). Unlearned words are guessed uniformly; choice is a power-law Luce rule.

### xsl_03: Load-scaled absent-alternative interference with saturating competitive encoding
Each word splits credit over the co-present objects and each object over the co-present words (sharpened strength plus a prior). A pair gains the product of its two responsibilities, which gives soft mutual exclusivity and explaining away. The gain saturates as a pair's strength approaches a ceiling, and it is mildly gated down when many objects are present. Forgetting is retrieval-induced interference. When a word is heard, its links to absent objects decay, and when an object is seen, its links to absent words decay. Co-present pairs are protected from this decay. The decay rate grows as a power law in trial load, the number of words times the number of objects. Sparse trials (1x3, 2x2) therefore barely erode earlier learning, so accuracy is high and snowballs with repetition, while dense trials (3x4, 4x4) prune strongly. Three, six and nine repetitions give graded rather than all-or-none learning. Test choice is a power-law Luce rule.

### xsl_01: Interference-limited mutual-responsibility learning
On each trial the learner divides credit for each word among the co-present objects, and for each object among the co-present words, in proportion to sharpened current strength plus a small prior. Each pair is strengthened by the product of its word-side and object-side responsibilities, which gives soft mutual exclusivity and explaining away. Forgetting is interference-based rather than global: only associations involving a word or object that is currently on the trial decay. Knowledge of items not currently competing persists, so extra exposure and long training help, while dense competition (many words and objects per trial) erodes strength. Test choice is a power-law Luce competition.

### kachergis: Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012)
Learners distribute a limited amount of attention over the word-object pairs on each trial. Attention is drawn both to familiar pairs (already strongly associated) and to uncertain stimuli (words or objects whose associations are still spread over many partners, measured by entropy). The attended share of a fixed associative weight is added to each pair, and all associations decay slightly on every trial. Familiarity produces a rich-get-richer bias; uncertainty pulls attention toward stimuli that have no strong partner yet, which together implement a soft mutual-exclusivity bias.

### baseline: Co-occurrence counting
Learners simply count how often each word and object occur together across trials, with no attention, competition or forgetting.

## BEST LEADERBOARD THEORY: xsl_08
**Name:** Decaying-trace contiguity interference learning

**Description:** Learning is competitive and interference-limited. Credit for a pair is the product of word-side and object-side responsibilities (power-sharpened shares), and the gain saturates and is gated by the number of visible objects. Retrieval-induced interference prunes links to absent partners, with a rate set by the number of objects and by words x objects load. The new claim is that the memory of recent co-occurrence is a graded, fading trace rather than a one-trial flag. A pair seen on the previous trial is strongly marked, one seen two or three trials ago is marked more weakly, and the mark decays geometrically. Marked pairs receive amplified credit (1 + r*trace), which is how temporal contiguity helps. This lets designs with overlap every 2 or 3 trials differ from designs with immediate repeats, and from random orderings. At test, objects claimed by other words are discounted (share^b), a retrieval floor is added, and choice is a power-law Luce rule.

**Parameters:**
- d: [0.0, 1.0]
- q: [0.0, 2.0]
- lam: [0.05, 4]
- Mmax: [0.3, 10]
- fl: [0.0, 1.0]
- b: [0.0, 3.0]
- s: [0.3, 4.0]
- r: [0.0, 3.0]
- ch_dec: [0.3, 6]
- p: [-0.5, 1.5]
- rho: [0.0, 0.9]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    d = params['d']
    q = params['q']
    lam = params['lam']
    Mmax = params['Mmax']
    fl = params['fl']
    b = params['b']
    s = params['s']
    r = params['r']
    p = params['p']
    rho = params['rho']
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
        deff = min(0.95, d * (nw * no / 12.0) ** q * (no / 4.0) ** p)
        ix = np.ix_(words, objs)
        blk = M[ix].copy()
        boost = 1.0 + r * prev[ix]
        M[words, :] *= (1.0 - deff)
        M[:, objs] *= (1.0 - deff)
        st = (blk + eps) ** s
        rw = st / st.sum(1, keepdims=True)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        M[ix] = blk + lam * enc * boost * inc * sat
        prev *= rho
        prev[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```

Per-condition fit, worst first:
| label | condition | human acc | model acc (full fit) | held-out NLL | held-out NLL per choice |
|---|---|---|---|---|---|
| 204 | 3x4 +6o | 0.272 | 0.338 | 458.5 | 1.273 |
| 217 | 2x4 6/12 | 0.325 | 0.363 | 305.5 | 1.212 |
| 218 | 3x4 6/8 | 0.419 | 0.376 | 266.1 | 1.137 |
| 201 | 3x4 | 0.191 | 0.332 | 459.0 | 1.020 |
| 203 | 3x4 1/.66 | 0.211 | 0.333 | 447.6 | 0.995 |
| 202 | 3x4 1/.5 | 0.224 | 0.342 | 432.9 | 0.962 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.348 | 477.2 | 0.855 |
| 206 | 3x3 +1w/o | 0.167 | 0.329 | 598.4 | 0.852 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.361 | 458.2 | 0.821 |
| 207 | 4x4 +2w/o | 0.103 | 0.104 | 575.0 | 0.819 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.337 | 437.8 | 0.811 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.341 | 435.9 | 0.807 |
| 215 | 1x3 | 0.744 | 0.483 | 330.5 | 0.798 |
| 221 | 1x4 | 0.190 | 0.247 | 574.3 | 0.798 |
| 205 | 2x4 | 0.303 | 0.325 | 470.3 | 0.792 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.344 | 423.9 | 0.760 |
| 216 | 2x3 | 0.580 | 0.550 | 309.0 | 0.746 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.438 | 429.4 | 0.746 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.601 | 799.5 | 0.724 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.438 | 390.2 | 0.723 |
| 224 | 3x3 | 0.426 | 0.473 | 454.2 | 0.701 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.414 | 410.5 | 0.671 |
| 219 | 2x3 6/9 | 0.547 | 0.522 | 379.4 | 0.659 |
| 222 | 1x3 | 0.392 | 0.464 | 466.7 | 0.648 |
| 220 | 3x4 9/12 | 0.685 | 0.485 | 383.5 | 0.646 |
| freq369_39mx | freq369_39mx | 0.622 | 0.436 | 453.2 | 0.629 |
| 225 | 2x2 | 0.787 | 0.661 | 206.0 | 0.602 |
| freq369_36mx | freq369_36mx | 0.454 | 0.447 | 599.0 | 0.504 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.475 | 550.4 | 0.493 |
| 223 | 4x4 | 0.312 | 0.318 | 639.4 | 0.461 |
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.490 | 748.2 | 0.400 |
| filt0E_6L | filt0E_6L | 0.487 | 0.590 | 76.7 | 0.139 |
| filt0E_3L | filt0E_3L | 0.372 | 0.475 | 103.3 | 0.139 |
| filt6E_3L | filt6E_3L | 0.424 | 0.411 | 81.6 | 0.113 |
| filt3E_3L | filt3E_3L | 0.449 | 0.429 | 81.1 | 0.113 |
| filt6E_6L | filt6E_6L | 0.553 | 0.483 | 61.0 | 0.111 |
| filt3E_9L | filt3E_9L | 0.575 | 0.568 | 81.6 | 0.110 |
| filt9E_9L | filt9E_9L | 0.601 | 0.501 | 80.5 | 0.108 |
| filt3E_6L | filt3E_6L | 0.482 | 0.520 | 59.1 | 0.107 |
| filt9E_3L | filt9E_3L | 0.414 | 0.406 | 78.9 | 0.106 |
| filt9E_6L | filt9E_6L | 0.513 | 0.468 | 56.8 | 0.103 |
| filt6E_9L | filt6E_9L | 0.590 | 0.522 | 73.3 | 0.099 |
| filt0E_9L | filt0E_9L | 0.586 | 0.644 | 69.6 | 0.094 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.751 | 63.9 | 0.093 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.649 | 47.0 | 0.084 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.660 | 53.8 | 0.083 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.704 | 58.2 | 0.075 |

## LOOP HISTORY
(first iteration)

## Response

{"name":"Decaying-trace contiguity interference with passive time decay","theory":"Learning is competitive and interference-limited: each pair gains the product of its word-side and object-side responsibilities (power-sharpened shares). The gain saturates and is gated by the number of visible objects. Retrieval-induced interference prunes links to absent partners at a rate set by trial load. A fading trace of recent co-occurrence amplifies credit for recently repeated pairs, which is how temporal contiguity helps. The new claim is that, besides interference, all associations also lose strength passively with every elapsed trial (time-based forgetting). Pairs learned long before test, or in long, sparse training, are therefore eroded even when their items are not competing. Late-trained words are favoured over early-trained ones, and the benefit of extra repetitions is tempered. At test, objects claimed by other words are discounted (share^b), a retrieval floor is added, and choice is a power-law Luce rule.","learn":"def learn(params, trials, n_words, n_objects):\n    d = params['d']\n    q = params['q']\n    lam = params['lam']\n    Mmax = params['Mmax']\n    fl = params['fl']\n    b = params['b']\n    s = params['s']\n    r = params['r']\n    p = params['p']\n    rho = params['rho']\n    gd = params['gd']\n    c = 0.5\n    g = 0.5\n    eps = 0.1\n    M = np.zeros((n_words, n_objects))\n    prev = np.zeros((n_words, n_objects))\n    for words, objs in trials:\n        words = np.asarray(words)\n        objs = np.asarray(objs)\n        nw = len(words)\n        no = len(objs)\n        enc = 1.0 / (1.0 + (no / c) ** g)\n        deff = min(0.95, d * (nw * no / 12.0) ** q * (no / 4.0) ** p)\n        ix = np.ix_(words, objs)\n        M *= (1.0 - gd)\n        blk = M[ix].copy()\n        boost = 1.0 + r * prev[ix]\n        M[words, :] *= (1.0 - deff)\n        M[:, objs] *= (1.0 - deff)\n        st = (blk + eps) ** s\n        rw = st / st.sum(1, keepdims=True)\n        ro = st / st.sum(0, keepdims=True)\n        inc = rw * ro\n        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)\n        M[ix] = blk + lam * enc * boost * inc * sat\n        prev *= rho\n        prev[ix] = 1.0\n    col = M.sum(0, keepdims=True)\n    share = (M + 1e-9) / (col + 0.05)\n    out = M * share ** b\n    return out + fl + 1e-6\n","parameters":{"d":"[0.0, 1.0]","q":"[0.0, 2.0]","lam":"[0.05, 4]","Mmax":"[0.3, 10]","fl":"[0.0, 1.0]","b":"[0.0, 3.0]","s":"[0.3, 4.0]","r":"[0.0, 3.0]","ch_dec":"[0.3, 6]","p":"[-0.5, 1.5]","rho":"[0.0, 0.9]","gd":"[0.0, 0.06]"},"rationale":"This extends the best leaderboard model, which already captures competition, interference and graded contiguity, with one nested parameter for passive time decay (gd=0 recovers xsl_08). Interference-only forgetting leaves pairs untouched when their items are absent. That makes the model too uniform across designs with different training lengths and orderings. It also gives no recency advantage for words learned late in training, which matters for the mixed-frequency designs and the long filtering conditions. Global decay gives a principled, generalising source of extra variation across conditions while keeping the mechanisms that already work."}

## Usage

```json
{
  "input_tokens": 12105,
  "output_tokens": 5669,
  "reasoning_tokens": 4314,
  "provider": "Claude Platform on AWS"
}
```
