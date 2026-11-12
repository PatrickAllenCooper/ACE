p = 'assemble_notes.py'
s = open(p).read()
reps = [
("- The per-system winner on those eight systems is local_info three times, ranker twice, sft, random and coverage-or-others once each. No single alternative policy realizes the envelope: the best single policy mean is TR's own 11.625.",
 "- Winners on those eight systems: local_info alone on w19 and w23 and tied on w17 and w21, ranker on w05 and w12, sft on w16 and tied on w17, random on w11, coverage tied on w17, frozen tied on w21. No single alternative policy realizes the envelope: the best single policy mean is TR's own 11.625."),
("The best LLM arm is strictly below the best numerical arm on one system (w16, sft 10 versus coverage and frozen 11).",
 "The best LLM arm is strictly below the best numerical arm on one system (w16, sft 10 versus coverage 11), tied on 14 and above on 9."),
("**Against random, only TR is nominally better on the predeclared endpoint (sign-flip p 0.011).** The ranker, frozen, sft and dpo are indistinguishable from random on every endpoint examined (all p above 0.2). The screen can barely tell its best numerical policy from random.",
 "**Against random, only TR is nominally better on the predeclared endpoint (sign-flip p 0.011).** The ranker, frozen, sft and dpo are not separated from random on any endpoint examined (over ten endpoints the smallest sign-flip p is 0.09, sft on the natural-scale area, and the smallest Wilcoxon p is 0.19). The screen can barely tell its best numerical policy from random."),
("Their sign also flips with the endpoint: dpo minus frozen is +1.54 cost units at threshold 0.04, -1.04 at threshold 0.02 and -0.06 on log-area.",
 "Their sign also flips with the endpoint: dpo minus frozen is +1.54 cost units at threshold 0.04, -1.04 at threshold 0.02 (in `key_contrasts.csv`) and -0.06 on log-area."),
("Required informative systems for that same policy improvement: (2.8016 * 7.53 / 1.71)^2 = about 152, against 305 mixed systems of which about 178 are informative. Removing uninformative systems saves about 40 percent of the total and nothing more.",
 "Required informative systems for that same policy improvement: (2.8016 * 7.53 / 1.71)^2 = about 152, against 305 mixed systems of which about 178 are informative. Removing uninformative systems about halves the number of systems to run, but the number of informative systems needed stays about the same."),
("Effects the size of ranker minus TR or frozen minus TR (2.0 units, larger SD) reach about 0.2 at 24 systems and 0.9 at 96.",
 "Effects the size of ranker minus TR or frozen minus TR (2.0 units, larger SD) reach about 0.2 at 24 systems and 0.9 at 96 with the t-test (0.5 to 0.7 with the Wilcoxon test)."),
("3. **Then interleave observation.** obs is 38 percent of TR's actions and 56 percent of local_info's (1 of 48 in steps 1-2, rising to about half from step 5).",
 "3. **Then interleave observation.** obs is 38 percent of TR's actions and 56 percent of local_info's. For TR it is 1 of 48 in steps 1-2, 35 percent in steps 5-8 and 45 to 47 percent afterwards. For local_info it is 2 of 48 in steps 1-2 and 56 to 69 percent from step 5."),
("5. **Graph matters for TR.** In uniform-menu forks TR uses mostly Y and joint actions. In uniform-menu colliders it mixes x+, y+, y- and obs. local_info is much less graph specific.",
 "5. **Graph matters.** In uniform-menu forks TR uses mostly Y and joint actions (65 of 96). In uniform-menu colliders it mixes x+, y+, y- and obs. local_info uses Y-involving actions almost only in colliders (34 of its 36)."),
("- ranker: only x- (141) and x+ (435). 12 of 24 runs are exactly \"x- seven times, then x+ seventeen times\".",
 "- ranker: only x- (141) and x+ (435). 10 of 24 runs are exactly \"x- seven times, then x+ seventeen times\" and 3 runs are x+ throughout."),
("In chains and colliders 16 percent remains, because the downstream slope on Y is still learned from natural variation in Y.",
 "In chains and colliders 16 percent remains, because the parts of those queries that run through Y (the Z-on-Y coefficient, and in colliders the natural Y mean) are still learned from natural variation in Y."),
("dpo never attains on w11, w12 and w19. The rank correlation of this residual with final error is 0.84 for dpo, 0.36 for the ranker and 0.21 for TR.",
 "dpo never attains on w11, w12 and w19. The rank correlation of this residual with final error is 0.84 for dpo, 0.36 for the ranker and 0.21 for TR. This is an asymptotic residual, not a strict lower bound for a finite run, because errors in different mechanisms can cancel (the ranker ends w12 at 0.032)."),
]
for a, b in reps:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('ok', len(reps))
