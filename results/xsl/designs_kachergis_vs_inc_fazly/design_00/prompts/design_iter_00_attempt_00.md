# design_iter_00_attempt_00

## System Prompt

You are an experimental psychologist designing a NEW cross-situational word learning experiment to tell two fitted theories apart. No data exist for it yet: the goal is a design on which the theories make clearly different, pre-registrable predictions that people can then be run on.

## THE PARADIGM
On each training trial a learner hears several novel words and sees several novel objects, with no indication of which word goes with which; each word has a consistent referent across trials. At test, people hear a word and choose its referent from candidate objects. Both theories are fitted to 47 existing adult conditions (between-subject ability differences modelled).

## WHAT TO PRODUCE
Python source for a design generator, `make_design(knobs, rng)`, a few knobs with candidate values, and a CONTRAST: a statistic of per-item test accuracy on which kachergis and inc_fazly disagree. The search builds the design at every knob setting (several random orders each), simulates experiments with 50 people per arm from each theory's fit (including parameter uncertainty), and keeps the setting where a pre-registered decision on the contrast (midpoint between the theories' predictions) most often picks the theory that generated the data. The design is ACCEPTED if that happens at least 80% of the time under both theories.

Aim for an effect with a clear mechanistic reading, not an artefact: a contrast between conditions or item types that differ in one factor. Arms are between-subject groups (50 people each), so prefer within-arm item contrasts where possible (more power per person) and use at most 4 arms.

## DESIGN FORMAT
`make_design` returns {"arms": {arm_name: arm, ...}}, where each arm is
  {"trials": [[[word, ...], [object, ...]], ...],          # training, in order
    "test": [{"word": w, "candidates": [o, ...], "correct": o, "group": "label"}, ...]}
Words and objects are integer (or string) labels, local to the arm. Each test item names its correct object explicitly and an optional group label the contrast can use.

## CONSTRAINTS (designs must be runnable online)
- per arm: at most 36 words, 36 objects, 108 training trials;
- no word or object repeated within a trial; any number of words and objects per trial (the corpus uses 1x2 up to 4x4; more words than objects or vice versa is allowed);
- every word is heard on at least 2 training trials;
- test items: at most 72; 2-36 candidates each, all shown during training; the correct object must have appeared with the word at least once. By default show all objects as candidates; use fewer only when the design needs it;
- `make_design` must finish in under 2 s and use only `rng` for randomness.

## EXAMPLE (a two-arm frequency manipulation, not a recommendation)
```python
def make_design(knobs, rng):
    n = 18
    def arm(reps_hi):
        pairs = [(w, w) for w in range(n)]
        hi = set(range(n // 2))
        tokens = [p for p in pairs for _ in range(reps_hi if p[0] in hi else 3)]
        rng.shuffle(tokens)
        trials = [tokens[i:i + 3] for i in range(0, len(tokens) - 2, 3)]
        trials = [t for t in trials if len({w for w, _ in t}) == 3]
        test = [{"word": w, "candidates": list(range(n)), "correct": w,
                 "group": "hi" if w in hi else "lo"} for w in range(n)]
        return {"trials": [[[w for w, _ in t], [o for _, o in t]] for t in trials], "test": test}
    return {"arms": {"x": arm(knobs["reps"])}}
```
with knobs {"reps": [6, 9]} and contrast `def contrast(acc, groups): return acc['x'][groups['x'] == 'hi'].mean() - acc['x'][groups['x'] == 'lo'].mean()`.

## OUTPUT
You have no tools and cannot run code: do not write tool calls, XML tags or `<invoke>` blocks. Reply with the JSON object only — no prose before or after it.
Return one JSON object with these fields:
- name: Short slug for the design: lowercase letters, digits and underscores.
- rationale: Which mechanism difference between the two theories the design targets, and why their predictions should diverge on the contrast.
- generator: Python source defining def make_design(knobs, rng) -> dict. `knobs` maps each knob name to one value; `rng` is a numpy Generator (use it for every random choice, so a seed fixes the design). `np` is available. See DESIGN FORMAT for the dict to return.
- knobs: Candidate values per knob, e.g. {"early": [0, 3, 6], "load": [2, 3]} (at most 12 values each). Every combination is a candidate design; the search keeps the most discriminating one. Use {} for a fixed design.
- contrast: Python source defining def contrast(acc, groups) -> float. `acc` maps each arm name to a numpy array of per-item accuracy (proportion correct, in test order); `groups` maps each arm name to a numpy array of the items' group labels (strings). Example: return acc['high'][groups['high'] == 'late'].mean() - acc['low'][groups['low'] == 'late'].mean()
- prediction: What each theory predicts for the contrast (direction and rough size), and why.


## User Prompt

## THEORIES
### THEORY 1: kachergis — held-out NLL 17135.1
Fitted parameters: X=0.0999, B=5.305, C=0.996, lapse=0.001, ability_sd=1.559

**Name:** Uncertainty- and familiarity-biased associative learning (Kachergis, Yu & Shiffrin 2012)

**Description:** Learners distribute a limited amount of attention over the word-object pairs on each trial. Attention is drawn both to familiar pairs (already strongly associated) and to uncertain stimuli (words or objects whose associations are still spread over many partners, measured by entropy). The attended share of a fixed associative weight is added to each pair, and all associations decay slightly on every trial. Familiarity produces a rich-get-richer bias; uncertainty pulls attention toward stimuli that have no strong partner yet, which together implement a soft mutual-exclusivity bias.

**Parameters:**
- X: [0.001, 5]
- B: [0.1, 15]
- C: [0.5, 1]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    X, B, C = params["X"], params["B"], params["C"]
    start = 0.01
    m = np.zeros((n_words, n_objects))

    def entropy(p):
        q = p[p > 0] / p.sum()
        return -(q * np.log2(q)).sum()

    for words, objects in trials:
        # seed never-seen pairs on this trial, and pairs between trial
        # stimuli and already-familiar stimuli, at a small start value
        block = m[np.ix_(words, objects)]
        block[block == 0] = start
        m[np.ix_(words, objects)] = block
        fam_o = m.sum(axis=0) > 0
        fam_w = m.sum(axis=1) > 0
        for w in words:
            m[w, fam_o & (m[w, :] == 0)] = start
        for o in objects:
            m[fam_w & (m[:, o] == 0), o] = start

        assocs = m[np.ix_(words, objects)]
        h_w = np.array([entropy(m[w, :]) for w in words])
        h_o = np.array([entropy(m[:, o]) for o in objects])
        weights = assocs * np.outer(np.exp(B * h_w), np.exp(B * h_o))
        m *= C
        m[np.ix_(words, objects)] += X * weights / weights.sum()
    return m
```


### THEORY 2: inc_fazly — held-out NLL 17153.6
Fitted parameters: lambda=0.1469, beta=21.84, lapse=0.001, ability_sd=1.446

**Name:** Probabilistic incremental alignment (Fazly, Alishahi & Stevenson 2010)

**Description:** On each trial every word aligns to each co-present object in proportion to its current probability of meaning that object (smoothed for never-seen pairs); alignments accumulate as associations, and each word's meaning probabilities are the smoothed, normalised associations.

**Parameters:**
- lambda: [1e-10, 2]
- beta: [2, 20000]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    lam, beta = params["lambda"], params["beta"]
    alpha, eps = 10.0, 0.001  # XSLmodels fazly() defaults
    assoc = np.zeros((n_words, n_objects))
    probs = np.zeros((n_words, n_objects))
    t_unseen = np.full(n_words, 1.0 / beta)
    for words, objects in trials:
        tr = probs[np.ix_(words, objects)].copy()
        unseen = tr == 0
        tr[unseen] = np.broadcast_to(t_unseen[words][:, None], tr.shape)[unseen]
        align = (tr + eps) / (tr.sum(axis=0, keepdims=True) + alpha * eps)
        assoc[np.ix_(words, objects)] += align
        denom = assoc[words, :].sum(axis=1) + beta * lam
        probs[words, :] = (assoc[words, :] + lam) / denom[:, None]
        t_unseen[words] = lam / denom
    return probs
```


## WHERE THEY DIFFER ON THE EXISTING CORPUS (mean accuracy per condition)
| label | condition | words x objects | trials | per trial | test | n_subj | human acc | kachergis acc | inc_fazly acc | held-out NLL kachergis minus inc_fazly |
|---|---|---|---|---|---|---|---|---|---|---|
| 201 | 3x4 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.191 | 0.360 | 0.424 | -3.6 |
| 202 | 3x4 1/.5 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.224 | 0.344 | 0.423 | -0.7 |
| 203 | 3x4 1/.66 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.211 | 0.356 | 0.422 | -2.7 |
| 204 | 3x4 +6o | 18x24 | 36 | 3.0x4.0 | 24-AFC | 20 | 0.272 | 0.298 | 0.410 | -2.4 |
| 205 | 2x4 | 18x18 | 54 | 2.0x4.0 | 18-AFC | 33 | 0.303 | 0.441 | 0.386 | -1.1 |
| 206 | 3x3 +1w/o | 18x18 | 54 | 3.0x3.0 | 18-AFC | 39 | 0.167 | 0.416 | 0.390 | +1.9 |
| 207 | 4x4 +2w/o | 18x18 | 54 | 4.0x4.0 | 18-AFC | 39 | 0.103 | 0.248 | 0.256 | -1.8 |
| 211 | 1x3 6/18 4AFC | 18x18 | 108 | 1.0x3.0 | 4-AFC | 43 | 0.674 | 0.814 | 0.701 | +17.5 |
| 212 | 2x3 6/9 4AFC | 18x18 | 54 | 2.0x3.0 | 4-AFC | 38 | 0.686 | 0.759 | 0.755 | -3.8 |
| 213 | 2x4 6/12 4AFC | 18x18 | 54 | 2.0x4.0 | 4-AFC | 31 | 0.624 | 0.726 | 0.711 | -1.2 |
| 214 | 3x4 6/8 4AFC | 18x18 | 36 | 3.0x4.0 | 4-AFC | 36 | 0.688 | 0.647 | 0.712 | -6.1 |
| 215 | 1x3 | 18x18 | 108 | 1.0x3.0 | 18-AFC | 23 | 0.744 | 0.565 | 0.382 | -11.9 |
| 216 | 2x3 | 18x18 | 54 | 2.0x3.0 | 18-AFC | 23 | 0.580 | 0.487 | 0.454 | -1.7 |
| 217 | 2x4 6/12 | 18x18 | 54 | 2.0x4.0 | 18-AFC | 14 | 0.325 | 0.414 | 0.386 | +6.2 |
| 218 | 3x4 6/8 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 13 | 0.419 | 0.362 | 0.419 | +0.3 |
| 219 | 2x3 6/9 | 18x18 | 54 | 2.0x3.0 | 18-AFC | 32 | 0.547 | 0.492 | 0.456 | -2.1 |
| 220 | 3x4 9/12 | 18x18 | 54 | 3.0x4.0 | 18-AFC | 33 | 0.685 | 0.442 | 0.479 | +2.6 |
| 221 | 1x4 | 18x18 | 108 | 1.0x4.0 | 18-AFC | 40 | 0.190 | 0.512 | 0.325 | +26.4 |
| 222 | 1x3 | 18x18 | 108 | 1.0x3.0 | 18-AFC | 40 | 0.392 | 0.559 | 0.377 | +17.2 |
| 223 | 4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 77 | 0.312 | 0.305 | 0.435 | -4.5 |
| 224 | 3x3 | 18x18 | 36 | 3.0x3.0 | 18-AFC | 36 | 0.426 | 0.401 | 0.486 | -2.2 |
| 225 | 2x2 | 18x18 | 54 | 2.0x2.0 | 18-AFC | 19 | 0.787 | 0.545 | 0.560 | -4.4 |
| 301 | 2x2 5x 24AFC | 24x24 | 60 | 2.0x2.0 | 24-AFC | 46 | 0.514 | 0.476 | 0.497 | +4.9 |
| filt0E_3L | filt0E_3L | 12x12 | 18 | 2.0x2.0 | 11-AFC | 31 | 0.372 | 0.475 | 0.513 | -0.7 |
| filt3E_3L | filt3E_3L | 12x12 | 27 | 2.0x2.0 | 11-AFC | 30 | 0.449 | 0.412 | 0.470 | -10.8 |
| filt6E_3L | filt6E_3L | 12x12 | 36 | 2.0x2.0 | 11-AFC | 30 | 0.424 | 0.401 | 0.459 | -27.5 |
| filt9E_3L | filt9E_3L | 12x12 | 45 | 2.0x2.0 | 11-AFC | 31 | 0.414 | 0.409 | 0.454 | -17.6 |
| filt0E_6L | filt0E_6L | 12x12 | 36 | 2.0x2.0 | 11-AFC | 23 | 0.487 | 0.590 | 0.629 | -1.2 |
| filt3E_6L | filt3E_6L | 12x12 | 45 | 2.0x2.0 | 11-AFC | 23 | 0.482 | 0.490 | 0.565 | -0.0 |
| filt6E_6L | filt6E_6L | 12x12 | 54 | 2.0x2.0 | 11-AFC | 23 | 0.553 | 0.462 | 0.538 | -6.3 |
| filt9E_6L | filt9E_6L | 12x12 | 63 | 2.0x2.0 | 11-AFC | 23 | 0.513 | 0.457 | 0.525 | +8.1 |
| filt0E_9L | filt0E_9L | 12x12 | 54 | 2.0x2.0 | 11-AFC | 31 | 0.586 | 0.659 | 0.694 | -0.7 |
| filt3E_9L | filt3E_9L | 12x12 | 63 | 2.0x2.0 | 11-AFC | 31 | 0.575 | 0.547 | 0.624 | -21.9 |
| filt6E_9L | filt6E_9L | 12x12 | 72 | 2.0x2.0 | 11-AFC | 31 | 0.590 | 0.507 | 0.589 | -5.4 |
| filt9E_9L | filt9E_9L | 12x12 | 81 | 2.0x2.0 | 11-AFC | 31 | 0.601 | 0.495 | 0.570 | -6.1 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.411 | 0.302 | 0.413 | +4.3 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.391 | 0.303 | 0.411 | +1.1 |
| freq369-3x3loCD | freq369-3x3loCD | 18x18 | 36 | 3.0x3.0 | 18-AFC | 34 | 0.485 | 0.356 | 0.428 | +2.5 |
| freq369-3x3hiCD | freq369-3x3hiCD | 18x18 | 36 | 3.0x3.0 | 18-AFC | 104 | 0.555 | 0.414 | 0.483 | +9.5 |
| freq369_36mx | freq369_36mx | 18x18 | 36 | 3.0x3.0 | 18-AFC | 66 | 0.454 | 0.381 | 0.456 | +17.7 |
| freq369_39mx | freq369_39mx | 18x18 | 36 | 3.0x3.0 | 18-AFC | 40 | 0.622 | 0.390 | 0.451 | +11.1 |
| cont_div6-12 | cont_div6-12 | 18x18 | 36 | 3.0x3.0 | 18-AFC | 62 | 0.435 | 0.399 | 0.481 | +3.1 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.489 | 0.309 | 0.435 | +1.6 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 18x18 | 27 | 4.0x4.0 | 18-AFC | 32 | 0.405 | 0.309 | 0.435 | +0.6 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.317 | 0.305 | 0.433 | -4.1 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.280 | 0.307 | 0.436 | -5.2 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.414 | 0.307 | 0.437 | +2.4 |

## WHAT A CORPUS DEBATE BETWEEN THEM FOUND
(no debate run)

## Response

{"name":"me_vs_cooccurrence_1word_trios","rationale":"The theories differ in where competition lies. Kachergis spreads attention across the objects within a trial, weighting each by its uncertainty (column entropy) and familiarity. Objects that already have a strong partner are therefore down-weighted, which gives a strong mutual-exclusivity bias. Fazly normalises alignments across the words competing for an object. With one word per trial there is no competition, so it reduces to co-occurrence counting. It cannot use the fact that an object already belongs to another word. Design: 6 known words are first taught with their objects in 1x1 trials. Then each of 6 'ME' words is always shown alone with the same 3 objects: one novel target and 2 known objects (the referents of other, already learned words). Each of 6 'control' words is always shown with 3 novel objects. Within each trio the objects co-occur identically with the word on every trial. Cross-trial statistics therefore cannot identify the target, and only the known/novel status of the objects can. Kachergis should pick the novel target in ME trios, because known objects have low entropy and get little attention. Fazly gives equal association to all three objects, so it sits at 1/3 on both item types. The control items are matched on trio structure and number of exposures, and they remove any generic offset. The contrast is a within-arm item-type difference.","generator":"def make_design(knobs, rng):\n    r1 = int(knobs['r1'])\n    r2 = int(knobs['r2'])\n    nk = 6\n    # known words 0-5 with objects 0-5\n    # ME words 6-11, targets objects 6-11, plus 2 known distractors each\n    # control words 12-17, objects 12+3i .. 14+3i (all novel)\n    trio = {}\n    group = {}\n    correct = {}\n    for i in range(nk):\n        w = 6 + i\n        d = rng.choice(nk, 2, replace=False)\n        trio[w] = [6 + i, int(d[0]), int(d[1])]\n        group[w] = 'me'\n        correct[w] = 6 + i\n    for i in range(nk):\n        w = 12 + i\n        objs = [12 + 3 * i, 13 + 3 * i, 14 + 3 * i]\n        trio[w] = objs\n        group[w] = 'ctrl'\n        correct[w] = int(objs[rng.integers(3)])\n    # phase 1: known words, 1x1 trials\n    p1 = [w for w in range(nk) for _ in range(r1)]\n    p1 = [p1[j] for j in rng.permutation(len(p1))]\n    trials = [[[int(w)], [int(w)]] for w in p1]\n    # phase 2: one word, three objects\n    p2 = [w for w in range(6, 18) for _ in range(r2)]\n    p2 = [p2[j] for j in rng.permutation(len(p2))]\n    for w in p2:\n        objs = [trio[w][j] for j in rng.permutation(3)]\n        trials.append([[int(w)], [int(o) for o in objs]])\n    test = []\n    for w in range(6, 18):\n        cands = [int(o) for o in trio[w]]\n        test.append({'word': int(w), 'candidates': cands, 'correct': int(correct[w]), 'group': group[w]})\n    for w in range(nk):\n        test.append({'word': int(w), 'candidates': list(range(nk)), 'correct': int(w), 'group': 'known'})\n    order = rng.permutation(len(test))\n    test = [test[j] for j in order]\n    return {'arms': {'main': {'trials': trials, 'test': test}}}\n","knobs":{"r1":[3,4,6],"r2":[4,5,6]},"contrast":"def contrast(acc, groups):\n    a = acc['main']\n    g = groups['main']\n    return a[g == 'me'].mean() - a[g == 'ctrl'].mean()\n","prediction":"Kachergis: the novel target wins the attention competition in ME trios, so ME accuracy (3-AFC) is well above chance, roughly 0.7-0.95. Control accuracy is about 1/3 because the three novel objects are symmetric. Contrast about +0.4 to +0.6. Inc_fazly: with a single word per trial, every co-present object gets nearly the same alignment, so the target and the two known distractors are tied. ME and control accuracy both sit near 1/3, and the contrast is about 0 (within about ±0.05). A threshold near +0.2 to +0.25 separates the theories."}

## Usage

```json
{
  "input_tokens": 6883,
  "output_tokens": 8971,
  "reasoning_tokens": 7409,
  "provider": "Claude Platform on AWS"
}
```
