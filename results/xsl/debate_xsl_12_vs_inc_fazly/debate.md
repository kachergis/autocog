# Debate: xsl_12 vs inc_fazly

## Round 1: xsl_12 vs inc_fazly

- proposed by xsl_12 on 221, 222: human 0.2014 (SE 0.0497); xsl_12 0.1936 (0.2 SE off), inc_fazly 0.0525 (3.0 SE off); winner: xsl_12
- proposed by inc_fazly on filt3E_9L, filt6E_9L, filt9E_9L: human 0.5887 (SE 0.0222); xsl_12 0.4911 (4.4 SE off), inc_fazly 0.5941 (0.2 SE off); winner: inc_fazly

**Verdict:** revise theory 1 (xsl_12), new_model. Keep xsl_12's competitive, interference-limited architecture, the load-dependent encoding, and the contiguity boost. Fix its handling of repetition in low-ambiguity, high-repetition regimes. (1) Remove or sign-flip the negative kappa exposure term. Replace it with a term that depends on how consistent a word's evidence is, so that a word whose top object has co-occurred on nearly every exposure gets an increasing, not decreasing, gain: expo = (1 + cnt*consistency)^kappa_pos, with kappa_pos >= 0. Keep diminishing returns only for words whose evidence is conflicting. (2) Make retrieval-induced interference on absent partners proportional to competing mass. This stops it from eroding a well-supported committed pair when the other word and object on the trial are fillers or unambiguous. Gate this with commit, so deff is scaled by (1 - commit*ratio) for the committed object's link. (3) Let commitment exclusion only redistribute credit away from claimed objects and never reduce the committed pair's own gain. Check the changes against 221/222, where the object-count effect must be kept, and against filt*_9L, 215, 220 and 225, where accuracy should rise by about 0.1 with repetition.

**Revision:** xsl_13 accepted (held-out NLL 16886.6)

## Round 2: xsl_13 vs inc_fazly

- proposed by xsl_13 on 216, 219: human 0.5633 (SE 0.0431); xsl_13 0.5527 (0.2 SE off), inc_fazly 0.4549 (2.5 SE off); winner: xsl_13
- proposed by inc_fazly on filt6E_6L, filt9E_6L, filt9E_9L, 2_x8_39_4x4, 3_x8_369_4x4: human 0.4936 (SE 0.0205); xsl_13 0.4237 (3.4 SE off), inc_fazly 0.4914 (0.1 SE off); winner: inc_fazly

**Verdict:** revise theory 1 (xsl_13), new_model. Keep xsl_13's central claim and replace the commitment-binary exposure and interference machinery with a cumulative-evidence gate. (1) Remove the negative-kappa diminishing returns, which are a main cause of the flat or declining accuracy with exposure in filt*_6L/9L and the 4x4 mixed-frequency conditions. Set the exposure exponent to a smooth non-negative function of the word's evidence concentration, for example e = kpos*conc, where conc = max_j rs_j (or 1 - normalized entropy of the word's row), instead of the sigmoid commit at th=0.76. Even partially-concentrated words should then get increasing returns. (2) Make interference erosion shielding graded by the same concentration: M[w,o] erodes by deff*(1 - rs_wo/mx)^k, not only for committed words. Also make deff for absent-partner erosion scale down for pairs with a long trace, so fillers cannot wash out consistent pairs in 2x2 designs. (3) Keep the exclusion term (xi) and the share^b test discount, since these produce the wins on 216/219/215/225, but apply the discount only after a word's own max row share exceeds a small threshold, so it does not suppress weak words in the high-ambiguity conditions 201-207, where xsl_13 is still too high. Refit d, kpos, and th (or drop th) after these changes, and check that filt6E_6L, filt9E_9L, 2_x8_39_4x4 and 3_x8_369_4x4 rise to about 0.5 without lowering 216/219.

**Revision:** no candidate beat xsl_13

## Round 3: xsl_13 vs inc_fazly

- proposed by xsl_13 on 207: human 0.1026 (SE 0.0096); xsl_13 0.1498 (4.9 SE off), inc_fazly 0.2562 (16.1 SE off); winner: xsl_13
- proposed by inc_fazly on filt0E_3L, filt3E_3L, filt6E_3L, filt9E_3L, filt0E_9L, filt3E_9L, filt6E_9L, filt9E_9L: human 0.1734 (SE 0.0258); xsl_13 0.0425 (5.1 SE off), inc_fazly 0.1450 (1.1 SE off); winner: inc_fazly

**Verdict:** revise theory 1 (xsl_13), new_model. Keep the competitive, load-limited, commitment-gated exclusion core, and change how repetition is credited. (1) Remove the blanket diminishing-returns exponent on uncommitted words. Replace (1+cnt)^kappa, which depends on the word's total exposure count, with a penalty that depends on competition: scale the exponent by the entropy of the word's association row, or by the number of co-present candidate objects that still carry mass. Then a word in a low-ambiguity 2x2 filtering design with repeated consistent pairings keeps full or increasing returns. A word in a 4x4 high-ambiguity design with scattered evidence still stays near floor, so 207 is preserved. (2) Make the commitment gain grow with the count of consistent co-occurrences of the top pair, for example expo = (1+n_consistent)^kpos_eff with kpos_eff allowed up to about 0.5 or more. Also use a soft threshold on the margin between the top and second object. This lets late repetitions (3 to 9) lift the target pair beyond the saturation ceiling. (3) Raise or make Mmax dependent on the committed state, so that sat=1-M/Mmax does not clip the late-repetition gain. Refit the parameters so that the filt*_3L to filt*_9L gain is about 0.15-0.17 while 207 stays near 0.10.

**Revision:** xsl_14 accepted (held-out NLL 16880.2)

