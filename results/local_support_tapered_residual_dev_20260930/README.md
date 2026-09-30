# Acquired-context support taper for nonlinear residual repair

Post hoc development on the same 12 connected systems and saved acquisitions as the preceding node-agnostic gate. Source revision `7a9317a7`. The selected RBF correction is multiplied at each prediction input by `exp(-d²/(2w²))`, where `d` is distance to the nearest **acquired** two-parent context and `w` is the RBF width chosen by acquired-label cross-validation. The linear source-warm prediction is used as the fallback. This uses neither simulator truth nor sealed evaluation labels to set a prediction, but the idea was chosen after seeing the prior failure; these systems cannot confirm it.

Across source size/policy arms, mean all-motif MSE (linear → untapered → tapered):

- Source16/risk: .04068 → .01813 → **.01262**; changed-motif .11446 → .03900 → .02063.
- Source16/fixed: .04443 → .03139 → **.01918**; changed-motif .12440 → .07638 → .03812.
- Source64/risk: .03708 → .06251 → **.01292**; changed-motif .10965 → .19426 → .02895.
- Source64/fixed: .03692 → .02690 → **.01351**; changed-motif .10716 → .06974 → .02731.

The source64/risk catastrophic motif-0 cell (seed 2007) is strongly suppressed by the taper, but the worst remaining motif-0 degradation is seed 2006/source64/risk: .00461 linear versus .10443 tapered. Mean unchanged-node error also rises on the fixed arms (.01016→.01107 at source16; .00681→.00759 at source64), exceeding the proposed 5% protection margin as a point ratio. The rule remains unsafe for promotion, despite better averages. Its next design must distinguish beneficial out-of-bank residual structure from spurious in-bank repairs using acquired evidence alone; further tuning on these 12 systems is exploratory and calls for a fresh frozen evaluation.

All 480 linear and untapered scores match the prior saved-data run within 1e-10. The 480-row CSV replays byte for byte (SHA-256 `0c1d05597ed7362e1985dbcfaa5f7355ea2f2da83327fcd793cf7da6c8f9107d`). Input/output hashes and source revision are in `complete.json`. Zero new acquisition queries, CURC jobs, closed-model calls, or non-ACE job changes; CURC SSH remained disconnected.
