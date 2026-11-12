p = 'assemble_notes.py'
s = open(p).read()
a = s.index('### 7.1 Mechanism-level explanation')
b = s.index('### 7.2 The sign bit')
new = """### 7.1 Mechanism-level explanation

What set_x_neg can and cannot identify, from the learner's update rule (`llm_intervention_scm.py` lines 82-98):

| mechanism | effect of a response under do(X = -1) |
|---|---|
| X root (natural mean of X) | never updated, because the clamped node is skipped (lines 83-85) |
| children of X (Y in chains and forks, Z in forks and colliders) | updated, but every new regressor has X = -1, so only the child's value at X = -1 is learned. Intercept and slope are not separated beyond what the prior and the eight initial observations give |
| Z given Y in chains | updated with naturally varying Y around its X = -1 level, so the Z-on-Y coefficient is learned, with leverage limited by the noise in Y |
| Y root in colliders | updated (Y is not clamped), and the Z-on-Y coefficient is learned from natural Y |

Which queries need what:

| query | chain X -> Y -> Z | fork Y <- X -> Z | collider X -> Z <- Y |
|---|---|---|---|
| q0 observe | natural X mean and both slopes | natural X mean, intercept and slope of Z on X | natural X and Y means, all of Z's coefficients |
| q1 do(X=-1), q7 do(X=-1, Y=1) | q1: exactly what x- samples. q7 equals q6 | both equal what x- samples | q1 and q7 are learned (value at X = -1 plus Z-on-Y) |
| q2, q3 do(X=0), do(X=1) | need the slope of Y on X | need the slope of Z on X | need the slope of Z on X |
| q4-q6 do(Y=y) | only Z given Y (learned) | identical to q0 (natural X mean) | natural X mean times the slope of Z on X |

Direct check of the first row: the X-mean estimate recovered from saved predictions is **bit-for-bit constant across all steps in 24 of 24 dpo runs and 24 of 24 ranker runs**, in 13 of 24 sft runs, and in 0 of 24 runs of each of the other five policies (script `a17_xroot.py`).

Empirical decomposition. Per-query error contributions were computed from reconstructed true target means (section 11.3). Mean contribution to NMSE at the final state, times 1000, by graph (8 systems each). "X-mean-dependent share" is the share of the row total carried by the queries whose true value depends on the natural X mean (q0 in chains; q0 and q4-q6 in forks and colliders).

{T['perq_abs']}

- **Chains:** dpo ends at 61.1 against 25.7 for TR. 39.9 of dpo's 61.1 (65 percent) is the observational query, against 6.1 for TR. The X = -1 query is learned better by dpo than by TR (2.4 against 9.7).
- **Forks:** dpo ends at 16.9 against 11.3. The excess is in the X = 0 and X = +1 queries (7.9 against 1.9), which need the slope that data at X = -1 cannot give. The X-mean-dependent queries are no worse than TR's (8.4 against 8.2).
- **Colliders:** dpo ends at 31.3 against 14.6. The excess is in the do(Y) queries (18.1 against 3.7) and the observational query (6.9 against 2.9). Both need the natural X mean times the slope of Z on X. The two X = -1 queries are learned best of any policy (2.6 in total against 3.8 for TR).
- The ranker (X-only, both signs) shows the same signature in chains (43.0 of 57.7 in q0) but not the fork slope problem (0.6 in X = 0 and +1), because it samples both X = -1 and X = +1. frozen (observe only) is the mirror image: smallest q0 error, largest error at X = -1 in chains.
- **These means are driven by a few systems.** w19 and w12 hold 98 percent of dpo's X-mean-dependent final error in chains. w11 and w23 hold 75 percent in colliders. w21 and w20 hold 68 percent in forks. These are the systems where X has a strong total effect on Z and the eight-observation estimate of the X mean is off.

Fraction of the initial error that remains at the final state, as a ratio of sums over the eight systems of each graph (below 1 = reduced):

{T['retained']}

The same as the median over systems of the per-system ratio (robust to the dominant systems):

{T['retained_median']}

- The X = -1 queries are the best learned part of dpo under both views (3.6 to 7.1 percent of the initial error remains in sums, 2 to 19 percent in medians).
- In forks dpo retains 62 percent of the X = 0 and X = +1 error in sums (TR 15 percent) and 54 percent in medians (TR 41 percent).
- For q0 the two views differ. In sums dpo retains 164, 76 and 97 percent (chain, fork, collider) against 25, 74 and 41 percent for TR. In medians q0 gets worse in chains for every policy (4.4 for dpo, 11.8 for TR, 2.0 for frozen) because q0 starts out accurate after eight observational samples, and in colliders dpo's median (0.36) is close to TR's (0.31). The q0 deficit of dpo is real in the systems where it matters and absent in the typical system.
- Size of the frozen-X-mean problem: if every other mechanism were known exactly, the residual error from the initial X-mean estimate alone would exceed 0.04 on w11 (0.041), w12 (0.043) and w19 (0.145), and exceed 0.01 on w10, w20 and w23 as well. dpo never attains on w11, w12 and w19. The rank correlation of this residual with final error is 0.84 for dpo, 0.36 for the ranker and 0.21 for TR. This is an asymptotic residual, not a strict lower bound for a finite run, because errors in different mechanisms can cancel (the ranker ends w12 at 0.032).

"""
s = s[:a] + new + s[b:]
open(p, 'w').write(s)
print('ok')
