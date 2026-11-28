# Debate: xsl_12 vs inc_fazly

## Round 1: xsl_12 vs inc_fazly

- proposed by xsl_12 on 221, 222: human 0.2014 (SE 0.0497); xsl_12 0.1936 (0.2 SE off), inc_fazly 0.0525 (3.0 SE off); winner: xsl_12
- proposed by inc_fazly on filt3E_9L, filt6E_9L, filt9E_9L: human 0.5887 (SE 0.0222); xsl_12 0.4911 (4.4 SE off), inc_fazly 0.5941 (0.2 SE off); winner: inc_fazly

**Verdict:** revise theory 1 (xsl_12), new_model. Keep xsl_12's competitive, interference-limited architecture, the load-dependent encoding, and the contiguity boost. Fix its handling of repetition in low-ambiguity, high-repetition regimes. (1) Remove or sign-flip the negative kappa exposure term. Replace it with a term that depends on how consistent a word's evidence is, so that a word whose top object has co-occurred on nearly every exposure gets an increasing, not decreasing, gain: expo = (1 + cnt*consistency)^kappa_pos, with kappa_pos >= 0. Keep diminishing returns only for words whose evidence is conflicting. (2) Make retrieval-induced interference on absent partners proportional to competing mass. This stops it from eroding a well-supported committed pair when the other word and object on the trial are fillers or unambiguous. Gate this with commit, so deff is scaled by (1 - commit*ratio) for the committed object's link. (3) Let commitment exclusion only redistribute credit away from claimed objects and never reduce the committed pair's own gain. Check the changes against 221/222, where the object-count effect must be kept, and against filt*_9L, 215, 220 and 225, where accuracy should rise by about 0.1 with repetition.

**Revision** (nll gate): xsl_13 accepted (held-out NLL 16886.6)

## Round 2: xsl_13 vs inc_fazly

- proposed by xsl_13 on 216, 219: human 0.5633 (SE 0.0431); xsl_13 0.5527 (0.2 SE off), inc_fazly 0.4549 (2.5 SE off); winner: xsl_13
- proposed by inc_fazly on filt6E_6L, filt9E_6L, filt9E_9L, 2_x8_39_4x4, 3_x8_369_4x4: human 0.4936 (SE 0.0205); xsl_13 0.4237 (3.4 SE off), inc_fazly 0.4914 (0.1 SE off); winner: inc_fazly

**Verdict:** revise theory 1 (xsl_13), new_model. Keep xsl_13's central claim and replace the commitment-binary exposure and interference machinery with a cumulative-evidence gate. (1) Remove the negative-kappa diminishing returns, which are a main cause of the flat or declining accuracy with exposure in filt*_6L/9L and the 4x4 mixed-frequency conditions. Set the exposure exponent to a smooth non-negative function of the word's evidence concentration, for example e = kpos*conc, where conc = max_j rs_j (or 1 - normalized entropy of the word's row), instead of the sigmoid commit at th=0.76. Even partially-concentrated words should then get increasing returns. (2) Make interference erosion shielding graded by the same concentration: M[w,o] erodes by deff*(1 - rs_wo/mx)^k, not only for committed words. Also make deff for absent-partner erosion scale down for pairs with a long trace, so fillers cannot wash out consistent pairs in 2x2 designs. (3) Keep the exclusion term (xi) and the share^b test discount, since these produce the wins on 216/219/215/225, but apply the discount only after a word's own max row share exceeds a small threshold, so it does not suppress weak words in the high-ambiguity conditions 201-207, where xsl_13 is still too high. Refit d, kpos, and th (or drop th) after these changes, and check that filt6E_6L, filt9E_9L, 2_x8_39_4x4 and 3_x8_369_4x4 rise to about 0.5 without lowering 216/219.

**Revision** (nll gate): no candidate beat xsl_13

## Round 3: xsl_13 vs inc_fazly

- proposed by xsl_13 on 207: human 0.1026 (SE 0.0096); xsl_13 0.1498 (4.9 SE off), inc_fazly 0.2562 (16.1 SE off); winner: xsl_13
- proposed by inc_fazly on filt0E_3L, filt3E_3L, filt6E_3L, filt9E_3L, filt0E_9L, filt3E_9L, filt6E_9L, filt9E_9L: human 0.1734 (SE 0.0258); xsl_13 0.0425 (5.1 SE off), inc_fazly 0.1450 (1.1 SE off); winner: inc_fazly

**Verdict:** revise theory 1 (xsl_13), new_model. Keep the competitive, load-limited, commitment-gated exclusion core, and change how repetition is credited. (1) Remove the blanket diminishing-returns exponent on uncommitted words. Replace (1+cnt)^kappa, which depends on the word's total exposure count, with a penalty that depends on competition: scale the exponent by the entropy of the word's association row, or by the number of co-present candidate objects that still carry mass. Then a word in a low-ambiguity 2x2 filtering design with repeated consistent pairings keeps full or increasing returns. A word in a 4x4 high-ambiguity design with scattered evidence still stays near floor, so 207 is preserved. (2) Make the commitment gain grow with the count of consistent co-occurrences of the top pair, for example expo = (1+n_consistent)^kpos_eff with kpos_eff allowed up to about 0.5 or more. Also use a soft threshold on the margin between the top and second object. This lets late repetitions (3 to 9) lift the target pair beyond the saturation ceiling. (3) Raise or make Mmax dependent on the committed state, so that sat=1-M/Mmax does not clip the late-repetition gain. Refit the parameters so that the filt*_3L to filt*_9L gain is about 0.15-0.17 while 207 stays near 0.10.

**Revision** (nll gate): xsl_14 accepted (held-out NLL 16880.2)

## Round 4: xsl_14 vs inc_fazly

- proposed by xsl_14 on 225, 207: human 0.6840 (SE 0.0549); xsl_14 0.4836 (3.7 SE off), inc_fazly 0.3038 (6.9 SE off); winner: xsl_14
- proposed by inc_fazly on filt6E_9L, filt9E_9L, filt6E_3L, filt9E_3L: human 0.1766 (SE 0.0377); xsl_14 0.0414 (3.6 SE off), inc_fazly 0.1231 (1.4 SE off); winner: inc_fazly

**Verdict:** revise theory 1 (xsl_14), new_model. Keep the competitive, load-limited, commitment-gated core, but change how committed pairs accumulate and how they are protected. (1) Make late gain for committed pairs non-saturating and stronger. Raise the kpos cap above 1, or replace (1+n_pair)^kpos with a gain that scales with the word's evidence for that object, such as lam*enc*(1+n_pair)^kpos*inc. The saturation term sat = 1 - blk/Mmax should then be made soft or Mmax raised for committed pairs, so a word with a clear referent keeps growing across 9 versus 3 late repetitions. (2) Protect committed associations from decay. Apply the deff multiplicative interference to committed pair cells only weakly: scale it by (1 - commit*protect), with protect fitted. Early filler trials and other late trials then stop eroding the committed row, which is what holds the 9L filtering conditions near 0.50. (3) Keep the ambiguity-scaled throttling for uncommitted words. This preserves the near-floor result in dense designs (207, 206, 3x4) while sparse designs (225, 215, 220) rise toward 0.7–0.8.

**Revision** (nll_and_query gate): no revision passed the gate for xsl_14

- Commitment-protected pair-count accumulation: held-out NLL 16882.7; filt6E_9L, filt9E_9L, filt6E_3L, filt9E_3L 3.6 -> 3.6 SE: failed
- Commit-gated repetition-count amplification at test: held-out NLL 16904.9; filt6E_9L, filt9E_9L, filt6E_3L, filt9E_3L 3.6 -> 3.6 SE: failed
- Contingency-weighted retrieval on competitive commitment learning: held-out NLL 16894.1; filt6E_9L, filt9E_9L, filt6E_3L, filt9E_3L 3.6 -> 3.6 SE: failed

## Round 5: xsl_14 vs inc_fazly

- proposed by xsl_14 on 223: human 0.3124 (SE 0.0293); xsl_14 0.3379 (0.9 SE off), inc_fazly 0.4354 (4.2 SE off); winner: xsl_14
- proposed by inc_fazly on 2_x8_39_4x4, 3_x8_369_4x4, temp_cont_2olap2tr_contr, 218: human 0.4087 (SE 0.0268); xsl_14 0.3570 (1.9 SE off), inc_fazly 0.4199 (0.4 SE off); winner: inc_fazly

**Verdict:** revise theory 1 (xsl_14), new_model. Keep the competitive, load-limited core, the contiguity trace and the commitment gating. Replace the hard commit-gated, ambiguity-throttled exponent (kappa*amb*(1-commit)) with a soft, evidence-proportional accumulation channel that is always on, as in Fazly. Specifically, add to each trial's increment a term proportional to the current normalized association, rsb = M/(rowsum+c). This gives a rich-get-richer alignment, roughly lam2*enc*(rsb+eps)/(sum over present objects of (rsb+eps)). It should not be multiplied by (1-xi*cl) exclusion or by the (1+cnt)^(kappa*amb) discount, so partly learned words keep gaining gradually from repeated pairs. Mix it with the existing sharp commit-gated path using one new weight parameter. Make the throttle depend on the within-trial competition margin, meaning how much the pair's alignment exceeds the alternatives, rather than on the word's global row entropy. Then dense 4x4 designs with no consistent pair stay near floor (223 and 207 preserved), while designs with repeated or overlapping pairs (2_x8_39_4x4, 3_x8_369_4x4, temp_cont_2olap2tr, filt*E_9L, 220, freq369_39mx) climb toward 0.4-0.6 instead of plateauing near 0.35-0.50. Refit lam, kappa and kpos after the change.

**Revision** (nll_and_query gate): no revision passed the gate for xsl_14

- Soft rich-get-richer alignment channel on commitment-gated competitive learning: held-out NLL 16911.5; 2_x8_39_4x4, 3_x8_369_4x4, temp_cont_2olap2tr_contr, 218 1.9 -> 1.4 SE: failed
- Two-timescale contiguity trace on competition-scaled commitment learning: held-out NLL 16886.0; 2_x8_39_4x4, 3_x8_369_4x4, temp_cont_2olap2tr_contr, 218 1.9 -> 1.7 SE: failed
- Margin-gated uncommitted returns: held-out NLL 16905.2; 2_x8_39_4x4, 3_x8_369_4x4, temp_cont_2olap2tr_contr, 218 1.9 -> 1.7 SE: failed

## Round 6: xsl_14 vs inc_fazly

- proposed by xsl_14 on 216, 223: human 0.2673 (SE 0.0834); xsl_14 0.2218 (0.5 SE off), inc_fazly 0.0185 (3.0 SE off); winner: xsl_14
- proposed by inc_fazly on filt6E_9L, filt9E_9L, 218, 2_x8_39_4x4, 3_x8_369_4x4: human 0.4823 (SE 0.0226); xsl_14 0.4126 (3.1 SE off), inc_fazly 0.4803 (0.1 SE off); winner: inc_fazly

**Verdict:** revise theory 1 (xsl_14), new_model. Keep the competitive, load-limited core and the ambiguity-scaled returns, but remove the ceiling that flattens late learning on consistent 2x2 designs. (1) Replace the hard saturation sat = 1 - blk/Mmax with a softer, evidence-dependent cap. For a committed pair, let the effective Mmax grow with the pair count, e.g. Mmax_eff = Mmax*(1+pc)^kpos, or drop saturation for committed pairs. (2) Make the per-trial multiplicative decay deff (applied to the whole word row and object column) weaker for a pair that has already been confirmed, i.e. scale deff by (1-commit*ratio) for the winning pair. Interference should then erode competitors, not the committed association. (3) Add a recovery term so that, after a word commits, the competitor mass in its row is pruned and its credit is redistributed to the winner, giving fazly-like sharpening with repetition. The aim is for filt*E_9L, 218 and the 4x4 moderate-repetition conditions to rise from about 0.41-0.50 to about 0.48-0.59, while keeping the sparse-versus-dense drop (216 vs 223), 207 and 225.

**Revision** (nll_and_query gate): no revision passed the gate for xsl_14

- Protected-commitment competitive learner with evidence-growing ceiling and winner sharpening: held-out NLL 16914.7; filt6E_9L, filt9E_9L, 218, 2_x8_39_4x4, 3_x8_369_4x4 3.1 -> 2.5 SE: failed
- Competitive load-limited learner with parallel count familiarity trace: held-out NLL 16893.8; filt6E_9L, filt9E_9L, 218, 2_x8_39_4x4, 3_x8_369_4x4 3.1 -> 3.2 SE: failed
- Consistency-gated growing returns with protected committed links: held-out NLL 16889.7; filt6E_9L, filt9E_9L, 218, 2_x8_39_4x4, 3_x8_369_4x4 3.1 -> 3.2 SE: failed

