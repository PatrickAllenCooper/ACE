# Next transfer test: sparse repair when localization is hard

Status: design only; no new target responses generated. This follows the fresh floor-six result, whose allocation and thresholds were selected after earlier development. It must not be counted as another confirmation of that result.

## What the existing evidence permits

The current 30-node, one-changed-node held-out-form experiment mostly gives away localization. On the 20 fresh systems at source_n=16, the strong change ranks first in 19 systems and second in one. The weak change ranks first in 18, third in one, and outside the logged top eight in one. At source_n=64, the strong change ranks first in all 20; the weak change ranks first in 17 and third/fourth in three. The floor-six policy therefore gives 11 responses to the changed node in 79 of 80 cells. These ranks and counts are read from the archived `actions.json` and `system.json` records, not a new simulation. The changed-node score improvement is unsurprising under that nomination rate. The soft mixture often matches scratch on the changed node, leaving no demonstrated reuse benefit.

## Scientific question

Can a modular source library lower target adaptation cost when **multiple** local mechanisms change and the initial assay may confuse a true mechanism change with noise or upstream distribution shift?

## Staged experiment

1. **Development only:** create 12 new 30-node systems with three changed nodes. Include one strong in-bank coefficient change, one weak held-out-form change, and one unchanged mechanism whose *parent distribution* shifts due to an upstream change. Generate those changes from distinct hidden mechanisms; serialize the graph and all conditional means. Keep source_n=16 and 64 as separate strata. Use the same four-response-per-node assay and a fixed 200-response target budget. The untouched nodes must include both descendants and nondescendants of changes. Report the complete rank distribution, top-four recall/precision, and false positives before comparing final prediction error. Do not alter the existing floor-six rule during this screen.
2. **Controls:** uniform target allocation; frozen floor-six/top-four allocation; a six-response floor with remaining responses allocated by a soft ranking rule declared before the development run; source-warm, scratch, and mixture estimators on each *identical acquired dataset*. A learned source representation receives a separate training-cost ledger. Compare acquisition effects within estimator and transfer effects within acquired data.
3. **Promotion decision:** freeze a new protocol on independent systems only if one development policy protects unchanged-node error and improves the changed-node adaptation curve against both uniform allocation and scratch on the same data. Require the portfolio's 20% primary gain and a paired system-level interval excluding zero on at least 20 fresh systems; require a prespecified 5% unchanged-node noninferiority interval. A point ratio alone is insufficient. Include both source-size strata, count all acquired and generated simulator responses separately, and report rank failures.
4. **If the numerical library provides no same-data benefit:** retain the sparse-change acquisition result as a distinct numerical study. Do not scale to a neural module library or claim a foundation-model transfer mechanism merely because active allocation helps.

The 12-system development screen is short numerical work suited to local CPU. CURC becomes useful for a 1–5M parameter modular neural baseline or a larger independent corpus after the data and evaluation contracts are implemented and checked. No Azure or other closed-model call is part of this stage.
