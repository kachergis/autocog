# design_iter_00_attempt_00

## System Prompt

You are an experimental psychologist designing a NEW cross-situational word learning experiment to tell two fitted theories apart. No data exist for it yet: the goal is a design on which the theories make clearly different, pre-registrable predictions that people can then be run on.

## THE PARADIGM
On each training trial a learner hears several novel words and sees several novel objects, with no indication of which word goes with which; each word has a consistent referent across trials. At test, people hear a word and choose its referent from candidate objects. Both theories are fitted to 47 existing adult conditions (between-subject ability differences modelled).

## WHAT TO PRODUCE
Python source for a design generator, `make_design(knobs, rng)`, a few knobs with candidate values, and a CONTRAST: a statistic of per-item test accuracy on which xsl_14 and kachergis disagree. The search builds the design at every knob setting (several random orders each), simulates experiments with 50 people per arm from each theory's fit (including parameter uncertainty), and keeps the setting where a pre-registered decision on the contrast (midpoint between the theories' predictions) most often picks the theory that generated the data. The design is ACCEPTED if that happens at least 80% of the time under both theories.

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
### THEORY 1: xsl_14 — held-out NLL 16880.2
Fitted parameters: d=0.1635, lam=5.999, Mmax=9.733, fl=0.004404, b=1.194, s=0.2981, r=2.114, ch_dec=0.4588, p=1.07, rho=0.2295, th=0.7934, kc=17.16, xi=1, kappa=-0.8956, kpos=0.195, lapse=0.001, ability_sd=1.436

**Name:** Competition-scaled uncommitted returns with pair-count commitment gain

**Description:** The core is kept from xsl_13: competitive, load-limited learning with retrieval-induced interference, a fading contiguity trace, and commitment-gated exclusion with redistributed credit. What changes is how repetition is credited. (1) The diminishing returns for an uncommitted word are no longer a blanket function of its exposure count. They scale with how ambiguous that word's association row currently is (1 - 1/effective number of candidate objects, from the inverse Simpson index). A word in a sparse, consistent design keeps near-linear returns from repetition. A word spread across many candidates, as in dense 4x4 designs, is still throttled and stays near floor. (2) Once a word has committed, its gain grows with the number of consistent co-occurrences of that specific word-object pair, (1+n_pair)^kpos. kpos can be large, so late repetitions keep lifting a committed pair instead of saturating with total word exposure.

**Parameters:**
- d: [0.0, 1.0]
- lam: [0.05, 6]
- Mmax: [0.3, 30]
- fl: [0.0, 0.5]
- b: [0.0, 3.0]
- s: [0.01, 4]
- r: [0.0, 12.0]
- ch_dec: [0.3, 3]
- p: [-0.5, 3.0]
- rho: [0.0, 0.9]
- th: [0.2, 0.9]
- kc: [0, 40]
- xi: [0.0, 1.0]
- kappa: [-1.5, 0.5]
- kpos: [0.0, 1.0]

**`learn` source:**
```python
def learn(params, trials, n_words, n_objects):
    d = params['d']
    lam = params['lam']
    Mmax = params['Mmax']
    fl = params['fl']
    b = params['b']
    s = params['s']
    r = params['r']
    p = params['p']
    rho = params['rho']
    th = params['th']
    kc = params['kc']
    xi = params['xi']
    kappa = params['kappa']
    kpos = params['kpos']
    q = 0.07
    c = 0.5
    g = 0.5
    eps = 0.1
    steep = 8.0
    M = np.zeros((n_words, n_objects))
    prev = np.zeros((n_words, n_objects))
    pc = np.zeros((n_words, n_objects))
    cnt = np.zeros(n_words)
    for words, objs in trials:
        words = np.asarray(words)
        objs = np.asarray(objs)
        nw = len(words)
        no = len(objs)
        enc = 1.0 / (1.0 + (no / c) ** g)
        deff = min(0.95, d * (nw * no / 12.0) ** q * (no / 4.0) ** p)
        ix = np.ix_(words, objs)
        rsA = M / (M.sum(1, keepdims=True) + 0.3)
        mxA = rsA.max(1)
        zA = np.clip(steep * (mxA - th), -30.0, 30.0)
        cA = 1.0 / (1.0 + np.exp(-zA))
        rs = rsA[words, :]
        mx = mxA[words]
        commit = cA[words]
        rsb = rs[:, objs]
        ratio = (rsb / (mx[:, None] + 1e-9)) ** 4
        blk = M[ix].copy()
        # ambiguity of each present word's association row
        Mw = M[words, :]
        rsum = Mw.sum(1)
        rn = Mw / (rsum[:, None] + 1e-12)
        neff = 1.0 / (np.sum(rn ** 2, axis=1) + 1e-12)
        neff = np.where(rsum > 1e-6, neff, float(no))
        neff = np.maximum(neff, 1.0)
        amb = 1.0 - 1.0 / neff
        boost = 1.0 + r * prev[ix] + kc * commit[:, None] * ratio
        contrib = commit[:, None] * rsb
        tot = contrib.sum(0, keepdims=True)
        cl = np.clip(tot - contrib, 0.0, 1.0)
        M[words, :] *= (1.0 - deff)
        M[:, objs] *= (1.0 - deff)
        st = (blk + eps) ** s
        stw = st * (1.0 - xi * cl)
        rw = stw / (stw.sum(1, keepdims=True) + 1e-12)
        ro = st / st.sum(0, keepdims=True)
        inc = rw * ro
        sat = np.clip(1.0 - blk / Mmax, 0.0, 1.0)
        e_unc = kappa * amb * (1.0 - commit)
        expo_u = np.exp(np.log(1.0 + cnt[words]) * e_unc)[:, None]
        expo_c = (1.0 + pc[ix]) ** (kpos * commit[:, None])
        expo = expo_u * expo_c
        M[ix] = blk + lam * enc * expo * boost * inc * sat
        cnt[words] += 1.0
        pc[ix] += 1.0
        prev *= rho
        prev[ix] = 1.0
    col = M.sum(0, keepdims=True)
    share = (M + 1e-9) / (col + 0.05)
    out = M * share ** b
    return out + fl + 1e-6
```


### THEORY 2: kachergis — held-out NLL 17135.1
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


## WHERE THEY DIFFER ON THE EXISTING CORPUS (mean accuracy per condition)
| label | condition | words x objects | trials | per trial | test | n_subj | human acc | xsl_14 acc | kachergis acc | held-out NLL xsl_14 minus kachergis |
|---|---|---|---|---|---|---|---|---|---|---|
| 201 | 3x4 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.191 | 0.347 | 0.360 | -1.6 |
| 202 | 3x4 1/.5 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.224 | 0.355 | 0.344 | -6.4 |
| 203 | 3x4 1/.66 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 25 | 0.211 | 0.349 | 0.356 | -0.9 |
| 204 | 3x4 +6o | 18x24 | 36 | 3.0x4.0 | 24-AFC | 20 | 0.272 | 0.337 | 0.298 | +0.0 |
| 205 | 2x4 | 18x18 | 54 | 2.0x4.0 | 18-AFC | 33 | 0.303 | 0.348 | 0.441 | -1.6 |
| 206 | 3x3 +1w/o | 18x18 | 54 | 3.0x3.0 | 18-AFC | 39 | 0.167 | 0.295 | 0.416 | -11.6 |
| 207 | 4x4 +2w/o | 18x18 | 54 | 4.0x4.0 | 18-AFC | 39 | 0.103 | 0.140 | 0.248 | -8.0 |
| 211 | 1x3 6/18 4AFC | 18x18 | 108 | 1.0x3.0 | 4-AFC | 43 | 0.674 | 0.780 | 0.814 | -31.4 |
| 212 | 2x3 6/9 4AFC | 18x18 | 54 | 2.0x3.0 | 4-AFC | 38 | 0.686 | 0.820 | 0.759 | -13.4 |
| 213 | 2x4 6/12 4AFC | 18x18 | 54 | 2.0x4.0 | 4-AFC | 31 | 0.624 | 0.673 | 0.726 | -13.5 |
| 214 | 3x4 6/8 4AFC | 18x18 | 36 | 3.0x4.0 | 4-AFC | 36 | 0.688 | 0.677 | 0.647 | -3.1 |
| 215 | 1x3 | 18x18 | 108 | 1.0x3.0 | 18-AFC | 23 | 0.744 | 0.504 | 0.565 | -2.6 |
| 216 | 2x3 | 18x18 | 54 | 2.0x3.0 | 18-AFC | 23 | 0.580 | 0.560 | 0.487 | -0.6 |
| 217 | 2x4 6/12 | 18x18 | 54 | 2.0x4.0 | 18-AFC | 14 | 0.325 | 0.375 | 0.414 | -4.5 |
| 218 | 3x4 6/8 | 18x18 | 36 | 3.0x4.0 | 18-AFC | 13 | 0.419 | 0.376 | 0.362 | -0.9 |
| 219 | 2x3 6/9 | 18x18 | 54 | 2.0x3.0 | 18-AFC | 32 | 0.547 | 0.545 | 0.492 | -0.7 |
| 220 | 3x4 9/12 | 18x18 | 54 | 3.0x4.0 | 18-AFC | 33 | 0.685 | 0.493 | 0.442 | +0.3 |
| 221 | 1x4 | 18x18 | 108 | 1.0x4.0 | 18-AFC | 40 | 0.190 | 0.292 | 0.512 | -28.8 |
| 222 | 1x3 | 18x18 | 108 | 1.0x3.0 | 18-AFC | 40 | 0.392 | 0.479 | 0.559 | -14.0 |
| 223 | 4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 77 | 0.312 | 0.338 | 0.305 | -0.3 |
| 224 | 3x3 | 18x18 | 36 | 3.0x3.0 | 18-AFC | 36 | 0.426 | 0.491 | 0.401 | +1.9 |
| 225 | 2x2 | 18x18 | 54 | 2.0x2.0 | 18-AFC | 19 | 0.787 | 0.624 | 0.545 | -0.2 |
| 301 | 2x2 5x 24AFC | 24x24 | 60 | 2.0x2.0 | 24-AFC | 46 | 0.514 | 0.581 | 0.476 | -3.0 |
| filt0E_3L | filt0E_3L | 12x12 | 18 | 2.0x2.0 | 11-AFC | 31 | 0.372 | 0.504 | 0.475 | +0.1 |
| filt3E_3L | filt3E_3L | 12x12 | 27 | 2.0x2.0 | 11-AFC | 30 | 0.449 | 0.461 | 0.412 | -19.5 |
| filt6E_3L | filt6E_3L | 12x12 | 36 | 2.0x2.0 | 11-AFC | 30 | 0.424 | 0.454 | 0.401 | -2.8 |
| filt9E_3L | filt9E_3L | 12x12 | 45 | 2.0x2.0 | 11-AFC | 31 | 0.414 | 0.454 | 0.409 | -2.1 |
| filt0E_6L | filt0E_6L | 12x12 | 36 | 2.0x2.0 | 11-AFC | 23 | 0.487 | 0.554 | 0.590 | -1.5 |
| filt3E_6L | filt3E_6L | 12x12 | 45 | 2.0x2.0 | 11-AFC | 23 | 0.482 | 0.482 | 0.490 | +6.1 |
| filt6E_6L | filt6E_6L | 12x12 | 54 | 2.0x2.0 | 11-AFC | 23 | 0.553 | 0.473 | 0.462 | -1.8 |
| filt9E_6L | filt9E_6L | 12x12 | 63 | 2.0x2.0 | 11-AFC | 23 | 0.513 | 0.472 | 0.457 | -5.9 |
| filt0E_9L | filt0E_9L | 12x12 | 54 | 2.0x2.0 | 11-AFC | 31 | 0.586 | 0.577 | 0.659 | -2.1 |
| filt3E_9L | filt3E_9L | 12x12 | 63 | 2.0x2.0 | 11-AFC | 31 | 0.575 | 0.505 | 0.547 | -1.5 |
| filt6E_9L | filt6E_9L | 12x12 | 72 | 2.0x2.0 | 11-AFC | 31 | 0.590 | 0.495 | 0.507 | +2.3 |
| filt9E_9L | filt9E_9L | 12x12 | 81 | 2.0x2.0 | 11-AFC | 31 | 0.601 | 0.495 | 0.495 | +9.1 |
| 2_x8_39_4x4 | 2_x8_39_4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.411 | 0.354 | 0.302 | +0.5 |
| 3_x8_369_4x4 | 3_x8_369_4x4 | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.391 | 0.344 | 0.303 | +3.0 |
| freq369-3x3loCD | freq369-3x3loCD | 18x18 | 36 | 3.0x3.0 | 18-AFC | 34 | 0.485 | 0.430 | 0.356 | -19.3 |
| freq369-3x3hiCD | freq369-3x3hiCD | 18x18 | 36 | 3.0x3.0 | 18-AFC | 104 | 0.555 | 0.516 | 0.414 | -10.5 |
| freq369_36mx | freq369_36mx | 18x18 | 36 | 3.0x3.0 | 18-AFC | 66 | 0.454 | 0.467 | 0.381 | -33.8 |
| freq369_39mx | freq369_39mx | 18x18 | 36 | 3.0x3.0 | 18-AFC | 40 | 0.622 | 0.457 | 0.390 | -13.2 |
| cont_div6-12 | cont_div6-12 | 18x18 | 36 | 3.0x3.0 | 18-AFC | 62 | 0.435 | 0.479 | 0.399 | -6.7 |
| 1_max_temp_spat_cont_orig | temporal+spatial contiguity | 18x18 | 27 | 4.0x4.0 | 18-AFC | 30 | 0.489 | 0.417 | 0.309 | -2.6 |
| 4_no_spat_orig_max_tc | temporal contiguity only | 18x18 | 27 | 4.0x4.0 | 18-AFC | 32 | 0.405 | 0.417 | 0.309 | -0.3 |
| 1olap3tr_contr | 1 overlap per 3 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.317 | 0.366 | 0.305 | -3.2 |
| temp_cont_1olap2tr_contr | 1 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.280 | 0.357 | 0.307 | -0.9 |
| temp_cont_2olap2tr_contr | 2 overlap per 2 trials, contr | 18x18 | 27 | 4.0x4.0 | 18-AFC | 31 | 0.414 | 0.354 | 0.307 | -3.3 |

## WHAT A CORPUS DEBATE BETWEEN THEM FOUND
(no debate run)

## DESIGNS ALREADY QUEUED (propose something different)
- ambiguity_at_fixed_word_frequency: Both arms give every word the same number of exposures (reps per word). They differ only in how many words share a trial. Arm A has 1 word with several objects per trial, so no words compete within a trial. Arm B has 2 or 3 words per trial with their referents plus fillers. Kachergis learns by uncertainty- and familiarity-weighted attention, with almost no decay (C=0.996). Its fit predicts high accuracy for single-word, many-object trials (corpus 1x4: 0.51, 1x3: 0.56), where each word's row narrows by repeated cross-trial intersection. xsl_14 adds load-limited forgetting scaled by objects per trial, a throttle on returns for words with ambiguous rows (kappa<0, so repetition buys little while a word is still uncommitted), and commitment-gated exclusion. Its fit predicts low accuracy for 1x4 (0.29), and its predictions for multi-word trials are comparable or higher (3x3 frequency designs about 0.45-0.5 versus Kachergis about 0.38-0.41). The contrast A minus B therefore has opposite signs under the two theories. It reads as the effect of within-trial word competition versus pure referential ambiguity, with word frequency held equal. The decision is made on a difference of arms, so the between-subject ability offset largely cancels. (power 1.00)

## Response

{"name":"massed_vs_spaced_same_frequency","rationale":"This targets the temporal-contiguity mechanism. xsl_14 has a fading trace (prev, rho=0.23) that multiplies the gain of any word-object pair repeated on the immediately preceding trial by 1 + r, which is about 3.1 with r=2.1. It also lets that gain feed commitment. Kachergis has no trial-to-trial trace. Its only temporal term is near-uniform decay (C=0.996), so for equal exposure counts the order of repetitions should matter little. In one arm, 9 words appear in unbroken runs of L consecutive trials (massed) and 9 words appear with gaps of at least 2 trials (spaced). Both groups get exactly 6 exposures, trials are 3x3, and every word has a distinct consistent referent. Frequency, trial size, set size and test format are therefore identical, and only the temporal arrangement of repetitions differs. Because the contrast is a within-arm item contrast, the person ability offset cancels. The knob L (run length 2 or 3) tunes how much of the massed group's exposure falls on trace-boosted repeats.","generator":"def make_design(knobs, rng):\n    L = int(knobs['L'])\n    T = 36\n    nw = 3\n    nm = 9\n    ns = 9\n    reps = 6\n    B = T // L\n    n_events = (reps * nm) // L\n    ev_per_word = reps // L\n    def build(strict_m, strict_s):\n        m_b = np.full(B, n_events // B)\n        extra = n_events % B\n        if extra > 0:\n            idx = rng.choice(B, extra, replace=False)\n            m_b[idx] += 1\n        slots = [b for b in range(B) for _ in range(int(m_b[b]))]\n        evw = [w for w in range(nm) for _ in range(ev_per_word)]\n        okm = False\n        for _ in range(400):\n            rng.shuffle(evw)\n            pos = {}\n            ok = True\n            # assign sorted slots (blocks ascending) to shuffled words\n            ss = sorted(slots)\n            for b, w in zip(ss, evw):\n                for b2 in pos.get(w, []):\n                    if b2 == b or (strict_m and abs(b2 - b) < 2):\n                        ok = False\n                        break\n                if not ok:\n                    break\n                pos.setdefault(w, []).append(b)\n            if ok:\n                okm = True\n                break\n        if not okm:\n            return None\n        massed = [[] for _ in range(T)]\n        for w, bl in pos.items():\n            for b in bl:\n                for t in range(b * L, (b + 1) * L):\n                    massed[t].append(w)\n        for t in range(T):\n            if len(massed[t]) > nw:\n                return None\n        cap = [nw - len(massed[t]) for t in range(T)]\n        for _ in range(300):\n            rem = {nm + i: reps for i in range(ns)}\n            placed = [[] for _ in range(T)]\n            fail = False\n            for t in rng.permutation(T):\n                for _k in range(cap[t]):\n                    cands = []\n                    for w, rr in rem.items():\n                        if rr <= 0 or w in placed[t]:\n                            continue\n                        if strict_s:\n                            if (t > 0 and w in placed[t - 1]) or (t < T - 1 and w in placed[t + 1]):\n                                continue\n                        cands.append(w)\n                    if not cands:\n                        fail = True\n                        break\n                    mx = max(rem[w] for w in cands)\n                    top = [w for w in cands if rem[w] == mx]\n                    w = top[int(rng.integers(len(top)))]\n                    placed[t].append(w)\n                    rem[w] -= 1\n                if fail:\n                    break\n            if not fail and all(v == 0 for v in rem.values()):\n                return massed, placed\n        return None\n    res = None\n    for strict_m, strict_s in [(True, True), (False, True), (False, False)]:\n        for _ in range(20):\n            res = build(strict_m, strict_s)\n            if res is not None:\n                break\n        if res is not None:\n            break\n    if res is None:\n        raise RuntimeError('no design')\n    massed, placed = res\n    nobj = nm + ns\n    omap = rng.permutation(nobj)\n    trials = []\n    for t in range(T):\n        ws = [int(w) for w in massed[t]] + [int(w) for w in placed[t]]\n        ws = [ws[i] for i in rng.permutation(len(ws))]\n        os_ = [int(omap[w]) for w in ws]\n        os_ = [os_[i] for i in rng.permutation(len(os_))]\n        trials.append([ws, os_])\n    test = []\n    for w in rng.permutation(nobj):\n        w = int(w)\n        test.append({'word': w, 'candidates': [int(o) for o in range(nobj)], 'correct': int(omap[w]), 'group': 'massed' if w < nm else 'spaced'})\n    return {'arms': {'a': {'trials': trials, 'test': test}}}\n","knobs":{"L":[2,3]},"contrast":"def contrast(acc, groups):\n    a = acc['a']\n    g = groups['a']\n    return a[g == 'massed'].mean() - a[g == 'spaced'].mean()\n","prediction":"xsl_14 predicts a positive contrast (massed minus spaced), roughly +0.05 to +0.15. Each repeat on the next trial gets the (1 + r*prev) gain boost, which pushes massed pairs toward commitment faster, and commitment then amplifies later gains. Kachergis predicts a contrast near 0, or slightly negative, because it has no trace of the previous trial and almost no decay, so equal exposure counts give nearly equal associations. The decision midpoint is therefore a small positive difference, and a clearly positive observed contrast supports xsl_14."}

## Usage

```json
{
  "input_tokens": 8789,
  "output_tokens": 7394,
  "reasoning_tokens": 5300,
  "provider": "Claude Platform on AWS"
}
```
