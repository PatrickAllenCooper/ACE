p = 'assemble_notes.py'
s = open(p).read()
reps = [
("Direct check of the first row: the X-mean estimate recovered from saved predictions is **bit-for-bit constant across all steps in 24 of 24 dpo runs and 24 of 24 ranker runs**, in 13 of 24 sft runs, and in 0 of 24 runs of each of the other five policies (script `a17_xroot.py`).",
 "Direct check of the first row: the X-mean estimate recovered from saved predictions is **constant across all steps, to floating-point roundoff, in 24 of 24 dpo runs and 24 of 24 ranker runs** (median range 1e-15), in 13 of 24 sft runs, and in 0 of 24 runs of each of the other five policies, where its median range is 0.21 to 0.31 (script `a17_xroot.py`)."),
("- Minimum detectable mean difference for a paired t-test with 24 systems (two-sided 0.05, 80 percent power) at the observed paired SD of about 6.2 cost units is **3.7 cost units**. That exceeds the absolute ceiling of 3.04.",
 "- Minimum detectable mean difference for a paired t-test with 24 systems (two-sided 0.05, 80 percent power) at the median observed paired SD of 6.2 cost units is **3.7 cost units**. That exceeds the absolute ceiling of 3.04. Across the seven observed SDs the minimum detectable difference ranges from 1.8 (coverage, SD 3.07) to 4.0 (frozen, SD 6.76)."),
("  - \"Ten systems already met the target ... limits how much the average endpoint can distinguish\" is correct and understates the issue: the minimum detectable difference at 24 systems exceeds the maximum possible improvement over TR.",
 "  - \"Ten systems already met the target ... limits how much the average endpoint can distinguish\" is correct and understates the issue: at the typical paired SD the minimum detectable difference at 24 systems (3.7 units) exceeds the maximum possible improvement over TR (3.04 units)."),
("## 0. Inputs, scripts and conventions",
 """## Top-line summary

1. **The saved data are internally consistent and every number in the handoff and the results report reproduces.** No numerical disagreement between documents and data.
2. **Ten of 24 systems are at target after initialization**: 5 of 8 chains, 5 of 8 forks, 0 of 8 colliders; 7 of 12 matched-regime and 3 of 12 shifted-regime systems; 6 of 12 uniform-menu and 4 of 12 heterogeneous-menu systems. They are weak-effect systems. On three of them an all-zero prediction already meets the target.
3. **There is almost no measurable room above target risk on the predeclared endpoint.** The absolute ceiling is 3.04 cost units per system (73 in total). The hindsight best-of-8 envelope is 2.04 (49 in total, 42 of them in five systems) and is plausibly mostly selection among noisy runs. The minimum detectable difference at 24 systems is 3.7 units at the typical paired SD.
4. **Single runs are dominated by measurement-noise replicate variance.** Arms do not share noise. dpo and sft ran the identical action prefix for at least 10 actions in every system, and their first-crossing costs differ on 7 of the 14 informative systems. Replicate SD of log10 NMSE at a fixed cost is about 0.3.
5. **On the predeclared endpoint target risk is separated only from dpo (sign-flip p 0.0068, Holm 0.048), random (0.011) and sft (0.022).** The four learned arms are not separated from random or from each other on any endpoint. Target risk and local information cannot be ranked.
6. **The curve-area and fixed-cost endpoints are not more sensitive than capped cost.** A stricter threshold (0.02) is.
7. **Power:** about 305 systems (80 percent) or 408 (90 percent) for a 1-unit improvement over the planner with single runs, 78 or 104 for 2 units, the same 305 or 408 for noninferiority with a 1-unit margin. For the log-area endpoint about 193 or 258 systems for a 20 percent improvement and 859 or 1,150 for 10 percent. Replicates could cut the 1-unit figure to about 93 to 172.
8. **Good behavior here is a mix, an opening rule and one sign bit, not a fixed sequence.** Both strong policies open with X at the extreme on the far side of the natural X mean (23 of 24 and 24 of 24 runs), then interleave observation (38 and 56 percent of actions) and use Y only when it is cheap. A fully fixed coverage cycle is within noise of the planner at the predeclared threshold.
9. **The constant x- policy loses most on colliders (+7.0 cost units) and where x- is the low-leverage side** (log-area +0.54 against target risk on 7 systems, -0.03 on the other 17). It can never update the X root mechanism and cannot separate intercept from slope in X's children.
10. **Error curves are strongly non-monotone.** Error rose after 43.9 percent of acquisitions. 61 of 172 runs that crossed the target later left it.
11. **The supervised arm's stop is step-linked, not accuracy-linked.** 23 stops after 21 to 23 actions at total cost 29 to 31, five of them above target.
12. Recommendations are in section 10. Open questions are in section 13.

## 0. Inputs, scripts and conventions"""),
]
for a, b in reps:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, 'w').write(s)
