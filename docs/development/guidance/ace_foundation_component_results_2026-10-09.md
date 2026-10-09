# Foundation-model mechanisms: first development screen

Date: 2026-10-09. This is a separately frozen six-world development experiment, not an amendment to accepted ACE studies or evidence of intervention efficiency. All 30 planned cells completed. No GPU, paid API, adaptive acquisition, held-out tuning, or replacement run was used.

## Evidence and question

The source revision is `68248cdd` (full revision in the freeze). Independently pinned pilot freeze: `e7c5ac2f837fe8251f2eb2637137e152c02003b0f922a10076fd76c516f7ebad`. Full artifacts are in `results/ace_foundation_component_20261009/ace-foundation-component-pilot-20261009-01`, with exclusive originals under `/Users/pat/ACE_Study_Results/2026-10-peter-baseline`. The repository custody index binds every copied artifact.

Each of six known X→M→Y worlds supplies the same 32 paid training responses to five methods. M has20eligible natural labels; Y has32. Private evaluation uses256common noise-disabled root actions per world. The endpoints are M prediction, Y prediction with measured parents, and composed Y prediction. NMSE uses eligible training-label variance only. The three generating function families are all present in the numerical selector's grammar; this is a deliberately favorable setting for a small correctly specified prior, not a test excluding foundation-model pretraining overlap.

## Results

Against numerical grammar selection, geometric means of the six paired composed-Y NMSE ratios were:

- Polynomial ridge: **1.8290**; worse in five of six worlds.
- ExtraTrees: **12.9572**; worse in six of six worlds.
- TabPFN-v2: **2.6432**; worse in five of six worlds.
- Language pipeline: **1.0000**, entirely from **six of six invalid proposals falling back to numerical grammar**. This is not a successful language-model result.

Lower ratios are better. Arithmetic means of paired ratios are distinct summaries, not ratios of pooled errors: polynomial3.5088, ExtraTrees50.4935, TabPFN3.8806, language fallback1.0. The source report preserves all endpoint/world values and absolute errors. No significance test is reported for six development worlds.

For TabPFN, geometric mean ratios are1.4505for M-local and6.3890for Y-local. A gain in a particular mechanism/world therefore cannot stand in for a composed-target gain. There were no failed scientific cells or normalization floor activations in the saved results (the reporter retains explicit failure/zero rules for other outcomes).

The language failures are substantive as well as syntactic: outputs include Markdown fences, illegal family strings such as `Y` and numeric strings, and a repeated data array. Merely removing fences would not make the declared interface reliable. Raw outputs, prompts, token counts and fallback identities are retained. No repair, retry or favorable valid-only summary was performed.

## What this changes in the method proposal

**Observed:** a small numerical family selector outperformed the tested general numerical foundation model on five of six composed tasks when its grammar contained the true mechanisms. The current tiny language interface provided zero valid proposals. Neither result establishes that all foundation models fail, or that a stronger semantic prior cannot help on a different task.

**Explanation to test:** model-class fit matters before experiment selection. When a small grammar already contains the truth and coefficients can be estimated from20–32eligible rows, a broader prior can add finite-sample bias/variance and computational overhead without adding useful representational support. The pilot is consistent with that explanation; it does not identify the internals of TabPFN's prior or causally isolate why its errors differ.

**Method recommendation:** keep an inexpensive numerical mechanism baseline as the default. Treat a foundation model as a candidate residual/alternative expert that must earn weight using training-only predictive checks. Preserve natural-label eligibility, legal interventions and observed-parent/composed-target reporting. Do not replace the proven acquisition learner with this TabPFN adapter on the strength of this screen.

**Language recommendation:** test a constrained typed proposal channel on artificial grammar fixtures before another scientific language arm. The model must select from the legal family/operator set, with token and failure costs charged. A future constrained decoder is a new method and requires a new freeze; these six failed proposals stay failed. Prompt examples, richer metadata and a larger language model would change the treatment and cannot be retroactively folded into this run.

## Next experiment worth preparing

Prepare a fresh, separately frozen *prior mismatch and recovery* screen before adaptive acquisition. Cross the existing correctly specified grammar with (1) an omitted smooth family and (2) a sparse change in one mechanism. Retain an unchanged-mechanism stratum and no-change control. Compare numerical grammar, TabPFN, and a training-only mixture with fixed cost limits. Prespecify the new function family, fresh seeds, mixture weighting/calibration rule, held-out supported action distribution, charged responses and abstention treatment before execution. No favorable world replacement.

The decisive question is whether broader pretrained structure helps when the compact prior is actually inadequate, without damaging mechanisms it already models well. Only an improvement that survives that screen would justify a crossed estimator×acquisition study against random, coverage and simple variance. No intervention-efficiency gain follows from this fixed-history pilot.

## Compute and attempt disposition

Pilot execution:13.9936elapsed seconds,14.526519childCPU seconds,5,062,934,528peak RSS bytes. Whole-process wait4 accounting includes imports, loading, fitting, generation and evaluation; worker phase metrics have a narrower scope. The slight CPU/elapsed difference means the thread setting alone is not proof of a strict instantaneous one-core ceiling during native-library startup. CPU limits and all CPU consumption remain charged; no GPU was used.

Four compatibility attempts (preserved predecessor01 and corrected03 for each model) plus the pilot consumed **28.436977childCPU seconds** and **0.029621supervisorCPU seconds** in the recorded terminal receipts. Preparation, tests, reviews, file hashing and Git/reporting commands are outside those measured totals; do not label this a complete sprint CPU cost. A conservative3600CPU-second preparation allowance plus480seconds reserved for four smokes and1800seconds for the pilot totals5880seconds reserved, within the28800-second exploration ceiling. Preparation02 never executed and is not an allocation.

All scientific worlds were generated only after the corrected supervisor/source/weight/runtime/smoke/commit gate passed. There are192shared training responses and1536private evaluation actions, reused across methods rather than multiplied by five. Existing accepted studies and failed delivery replay06 remain unchanged. No additional replay is authorized by this work.
