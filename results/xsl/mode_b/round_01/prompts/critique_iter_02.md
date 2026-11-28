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

## CANDIDATE — BIC 32002.0, NLL 15969.9, wSSE 0.4980 — ACCEPTED
Fitted parameters: a=0.2605, f=0.0335, k=0.6885, g=3, ch_dec=1.431, lapse=0.3432 — at bound: g

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


## PER-CONDITION FIT, WORST FIRST
| label | condition | human acc | model acc | NLL | NLL minus xsl_01's |
|---|---|---|---|---|---|
| freq369-3x3hiCD | freq369-3x3hiCD | 0.555 | 0.455 | 762.7 | +57.1 |
| 206 | 3x3 +1w/o | 0.167 | 0.350 | 637.5 | +42.0 |
| 221 | 1x4 | 0.190 | 0.377 | 615.1 | +24.9 |
| 207 | 4x4 +2w/o | 0.103 | 0.140 | 630.6 | +24.7 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 0.489 | 0.312 | 435.1 | +19.4 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 0.411 | 0.307 | 436.6 | +17.2 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 0.405 | 0.312 | 452.1 | +16.6 |
| freq369_39mx | freq369_39mx | 0.622 | 0.436 | 444.0 | +15.8 |
| 220 | 3x4 9/12 | 0.685 | 0.468 | 378.9 | +15.2 |
| 222 | 1x3 | 0.392 | 0.503 | 480.4 | +14.3 |
| 214 | 3x4 6/8 4AFC | 0.688 | 0.563 | 69.4 | +12.7 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 0.280 | 0.334 | 481.5 | +12.3 |
| 223 | 4x4 | 0.312 | 0.332 | 659.5 | +12.1 |
| 301 | 2x2 5x 24AFC | 0.514 | 0.579 | 785.3 | +9.2 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 0.317 | 0.329 | 464.2 | +8.4 |
| 224 | 3x3 | 0.426 | 0.467 | 459.6 | +8.3 |
| 213 | 2x4 6/12 4AFC | 0.624 | 0.564 | 55.6 | +7.0 |
| 212 | 2x3 6/9 4AFC | 0.686 | 0.628 | 62.9 | +6.4 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 0.391 | 0.317 | 431.6 | +6.4 |
| 205 | 2x4 | 0.303 | 0.368 | 485.3 | +6.1 |
| 216 | 2x3 | 0.580 | 0.485 | 317.9 | +4.4 |
| 218 | 3x4 6/8 | 0.419 | 0.345 | 273.1 | +3.7 |
| 201 | 3x4 | 0.191 | 0.357 | 470.8 | +3.4 |
| 211 | 1x3 6/18 4AFC | 0.674 | 0.618 | 59.6 | +3.4 |
| 204 | 3x4 +6o | 0.272 | 0.374 | 484.2 | +2.7 |
| 202 | 3x4 1/.5 | 0.224 | 0.372 | 440.9 | +2.5 |
| 217 | 2x4 6/12 | 0.325 | 0.339 | 308.9 | +2.2 |
| filt0E_9L | filt0E_9L | 0.586 | 0.607 | 64.9 | +0.6 |
| 203 | 3x4 1/.66 | 0.211 | 0.363 | 453.4 | -0.7 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 0.414 | 0.334 | 431.3 | -1.0 |
| 219 | 2x3 6/9 | 0.547 | 0.503 | 382.0 | -2.7 |
| freq369_36mx | freq369_36mx | 0.454 | 0.432 | 597.0 | -3.0 |
| freq369-3x3loCD | freq369-3x3loCD | 0.485 | 0.393 | 409.6 | -6.6 |
| filt3E_9L | filt3E_9L | 0.575 | 0.602 | 133.3 | -7.2 |
| filt0E_6L | filt0E_6L | 0.487 | 0.516 | 54.5 | -8.4 |
| filt3E_6L | filt3E_6L | 0.482 | 0.515 | 73.7 | -9.9 |
| cont_div6-12 | cont_div6-12 | 0.435 | 0.462 | 546.4 | -11.4 |
| filt6E_9L | filt6E_9L | 0.590 | 0.587 | 111.1 | -16.9 |
| 225 | 2x2 | 0.787 | 0.604 | 215.1 | -18.1 |
| filt3E_3L | filt3E_3L | 0.449 | 0.372 | 120.7 | -23.0 |
| filt6E_3L | filt6E_3L | 0.424 | 0.382 | 110.3 | -23.8 |
| filt6E_6L | filt6E_6L | 0.553 | 0.504 | 82.1 | -26.0 |
| filt9E_6L | filt9E_6L | 0.513 | 0.446 | 59.4 | -30.7 |
| 215 | 1x3 | 0.744 | 0.500 | 289.8 | -32.3 |
| filt9E_9L | filt9E_9L | 0.601 | 0.508 | 94.4 | -38.8 |
| filt9E_3L | filt9E_3L | 0.414 | 0.360 | 91.9 | -62.6 |
| filt0E_3L | filt0E_3L | 0.372 | 0.349 | 65.6 | -63.0 |

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
| 1 | xsl_01 | Interference-limited mutual-responsibility learning | 5 | 32045.4 | 15996.8 | 0.5529 | s=1.077, g=3.214, C=0.8347, ch_dec=0.9653, lapse=0.4636 |
| 2 | kachergis | Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012) | 4 | 32391.7 | 16175.2 | 0.5668 | X=0.2411, B=1.456, C=0.973, lapse=0.362 |
| 3 | baseline | Co-occurrence counting | 1 | 39486.6 | 19738.1 | 1.3660 | lapse=0.3881 |

## LOOP HISTORY
- iter 0: Expected propose-but-verify with consolidation — BIC 32080.0 — ACCEPTED
  critic: Make forgetting and consolidation depend on trial load and on repeated confirmation, so the model gains dynamic range and the lapse can fall. (1) Replace the constant forgetting f with a load-dependent rate, f_eff = 1 - (1-f)^(n_objs_in_trial * n_words_in_trial / c), or simply f * (n_objs_in_trial - 1). A hypothesis is then much more likely to be lost in 1x4, 3x4 and 4x4 trials than in 1x3, 2x2 and 3x3 trials. (2) Make consolidation accelerate with confirmation. For example, use a probability k * (confirmed mass)^g with an exponent g > 1, or let an already-confirmed hypothesis consolidate at a higher rate. Words confirmed on several occasions (high frequency, low contextual diversity, 2x2) would then be locked in, and the model could reach 0.6-0.8 accuracy in easy conditions. Keep the number of free parameters about the same: drop me, since it is stuck at its bound and a fixed value of about 10 would do, and add one load-scaling parameter. Expect the fitted lapse to drop well below 0.4.
- iter 1: Load-scaled propose-but-verify with consolidation — BIC 32070.1 — ACCEPTED
  critic: Keep the expected-distribution propose-but-verify core and mutual exclusivity. Change two things, with a net of at most one extra parameter (drop the fixed `me` scaling if needed). (1) Make forgetting steeply and super-linearly dependent on the number of objects in the trial: fe = 1 - (1-f)^(n_obj**g * n_words**0.5), with a fitted exponent g in [1,4] and f allowed up to about 0.5. This should push 1x4, 3x4 and 4x4 down toward 0.2 while leaving 2x2 and 3x3 nearly unaffected. (2) Make consolidation accelerate with repeated confirmation. Track a per-word confirmation count (expected, or mass-weighted). Use a two-stage hypothesis state, fresh then confirmed once, where only a second confirmation consolidates, with probability k. Alternatively, use consolidation probability k*(confirmed mass)**2 in place of the linear k*hp. Confirmed-once hypotheses should also be forgotten at a reduced rate. This should lift high-frequency, low-diversity and 2x2 conditions toward 0.7–0.8 and let the fitted lapse fall well below 0.4.

## Response

{"interpretation":"The candidate narrowly beats the leaderboard (BIC 32002 vs 32045), but it is compressed in dynamic range. The fitted lapse is 0.34 and the load exponent g sits at its upper bound of 3. Easy conditions are badly underpredicted: 215 (1x3) 0.74 human vs 0.50 model, 225 (2x2) 0.79 vs 0.60, freq369-3x3hiCD 0.555 vs 0.455, freq369_39mx 0.62 vs 0.44, and 220 (3x4 at frequency 9) 0.685 vs 0.47. Hard conditions are overpredicted: 201 (3x4) 0.19 vs 0.36, 206 (3x3 +1w/o) 0.17 vs 0.35, 221 (1x4) 0.19 vs 0.38, 202–204 and 207. The model cannot fall below about 0.35 in crowded conditions, so the fit leans on a high lapse. That lapse caps easy-condition accuracy near 0.68. The 1x3 → 1x4 drop (0.39 → 0.19 in 222 vs 221) shows a sharp cliff in object count that the capped exponent cannot produce. The worst NLL excesses are in exactly these conditions (freq369-3x3hiCD, 206, 221, 207). The mechanism is also too weak on frequency and contextual-diversity effects, since hiCD and 39mx come out too low. Under heavy load, forgetting is missing for consolidated hypotheses and the proposal rate is not load-dependent.","verdict":"regenerate","rationale":"Keep the three-stage propose-verify-consolidate core and mutual exclusivity. Make three changes. (1) Free the load nonlinearity: widen g to [1,8] and write load = (nw*0.5)**0.5 * (no/3.0)**g. This lets the cliff between 3 and 4 objects reach 1x4, 3x4, 4x4 and 3x3+1w/o, so they drop toward 0.1–0.2, while 1x3, 2x2 and 3x3 stay nearly unforgotten. (2) Make the proposal/encoding probability also fall with load, a_eff = a / (1 + load) or a * exp(-load*c), so that crowded trials seed fewer hypotheses. This removes the 0.35 floor in 3x4, 1x4 and 206/207. (3) Let consolidated stage S decay slightly when the same word occurs under high load, for example with rate 0.1*fe. This is not a new parameter, only a fixed fraction. Keep the number of free parameters about equal by fixing me=20. Expect the fitted lapse to fall below 0.25 and easy conditions (215, 225, 220, hiCD) to rise to 0.65–0.8."}

## Usage

```json
{
  "input_tokens": 8916,
  "output_tokens": 1708,
  "reasoning_tokens": 759,
  "provider": "Claude Platform on AWS"
}
```
