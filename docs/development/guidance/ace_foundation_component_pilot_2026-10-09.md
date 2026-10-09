# Component pilot: conditional prediction before acquisition

Status: prospective development protocol, October 9, 2026. Author: Codex for Patrick. This is a new synthetic development screen, not confirmation or an alteration to accepted ACE studies.

## Question and fixed design

Can an off-the-shelf numerical foundation model predict individual mechanisms and composed outputs better on the SAME small intervention history? Does the specified language-selection pipeline improve prediction over the specified numerical-selection pipeline using the SAME grammar? No adaptive acquisition or sample-efficiency claim can follow from this first screen.

Use six worlds, seeds 91000–91005. Known chain X→M→Y. Mechanism M uses, in order, linear, quadratic, tanh, linear, quadratic, tanh. Y uses the next family cyclically. Independent coefficients and offsets are drawn once per world from the fixed implementation. Disturbances are independent Gaussian SD0.05. Fully observed, no hidden confounding. These small familiar forms are development tasks, not pretraining-contamination controls.

Each world has exactly32 shared training responses:8 observational root draws,12 root interventions cycling −1,−0.5,0,0.5,1, then12 internal interventions over the same menu. An intervention on M excludes that row from M fitting but permits its measured value as input to Y. Random root draws in internal interventions remain observed. No environmental calls during fitting. All methods receive identical eligible rows.

Private evaluation contains256 common root interventions drawn uniformly on [−1,1], noise disabled, from a distinct fixed random stream. Report mechanism M and Y prediction on true measured parents separately from composition X→predicted M→predicted Y. Normalize each MSE by the corresponding eligible training-label variance only (floor1e−12, report activation). No held-out selection, tuning or initialization search. No causal identification or broad-domain generalization claim.

## Frozen comparisons

- Polynomial ridge degree3, alpha0.01, one input per mechanism.
- ExtraTrees128trees, min_samples_leaf2, seed0, one CPU.
- TabPFN2.2.1, explicit v2 regression checkpoint at HF revision4972a65a1b30806315c6f92499959ffbfc69a673, one estimator, CPU, seed0. Record actual dependencies and checkpoint hash before execution. No package-default model download.
- Numerical grammar selector: candidate bases [1,x], [1,x,x²], [1,tanh(x)] with ridge1e−6. First75% eligible rows fit candidate coefficients; final25% rank held-out training prediction MSE; then selected family refits all eligible rows. Ties follow listed order. This uses only training data, not private evaluation.
- Language proposal: Qwen3-0.6B cached revisionc1899de289a04d12100db370d81485cdf75e47ca, greedy CPU generation, thinking disabled, max64newtokens, seed0, exactlyone request per world with first8 eligible pairs per mechanism, anonymized names and the SAME three allowed family strings. One strict JSON object with keys M,Y; invalid requests are retained and fall back to the numerical selector. No retries or model-generated code. Fit the proposed family on all eligible rows, same ridge1e−6. Invalid-output fallback is reported separately from a valid proposal.

The language arm deliberately tests a modest family-choice interface; it is not a full agent. Numerical selection sees all32 paid rows whereas the prompt uses an8-pair excerpt per mechanism. This is a pipeline comparison: selection information and procedure differ, so neither positive nor negative differences isolate language preference alone. A negative result does not rule out every language model/interface. Informative/misleading independently authored metadata and richer grammars remain future protocols.

## Resources, gates and reporting

One local CPU thread per inference/fit process; CPU only, no GPU allocation. Technical smoke uses a fixed artificial straight line unrelated to these worlds, at most2minutes. A separate greedy language smoke with a generic grammar-format prompt checks executable compatibility only. Freeze source, environment, weights and protocol hashes before scored runs. If unavailable/failed, retain failure for every affected planned cell; no replacement model or favorable rerun. Maximum complete pilot30minutes elapsed and1800childCPU seconds; existing working cap8CPUcorehours/1GPUhour remains outer limit. Stop on limit, preserve partial outputs and identify unattempted cells.

Record wall/processCPU/peakRSS, raw proposals, training/evaluation arrays, model configuration, seeds and all30planned world-method cells. Summaries are descriptive arithmetic and geometric ratios with per-world values; no significance tests on this development set. Language invalidity, eligible counts, normalization floors and failed cells are mandatory. Pilot gain does not authorize confirmation or demonstrate intervention efficiency. A crossed acquisition study requires its own freeze and usefulness evidence from this screen.
