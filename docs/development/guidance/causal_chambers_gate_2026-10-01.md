# Causal Chambers fallback: bounded offline prediction gate

Source audit pinned the public dataset repository to `0fa8222dc761829270c8959e0ba53b261b075e1c`. Local custody: `results/causal_chambers_source_audit_20261001/source_receipt.json`, README, generator (MIT notice retained), variable dictionary. No measurement rows have been downloaded or evaluated.

## Selection and limitations

Select `lt_malus_v1` for the next metadata/data integrity audit. Independent experiment authors jointly manipulate two polarizer angles while holding other settings fixed within each experiment. This supplies a documented physical joint-input mechanism, unlike the rejected CausalMan probe menu. It is a pre-existing physical dataset, not a callable simulator. Its twelve color/brightness conditions are settings of the same apparatus, not twelve independent worlds. No arbitrary-action response, repeated-measurement noise estimate, or foundation-model benefit has been established.

The README lists the angle grid with an inclusive +90 endpoint. The pinned generator uses `np.arange(-90,90,0.1)` and formats commands to one decimal place, implying an upper commanded endpoint of +89.9. Freeze the actual recorded command domain after verifying the archive rather than assuming the prose range. The generator requests 1000 measurements per condition; observed counts remain unverified. Recorded sensor angles must not silently substitute for commanded actions.

## Next bounded CPU step

Download only the named archive after checking its size, verify published MD5 `cc49b95d85410e0b5ea3bcd1479428e3`, and retain a SHA-256 receipt. Inspect filenames, action columns, timestamps, missingness, duplicate settings, actual counts and protocol/measurement alignment without fitting outcome models. Keep outcome values sealed from model selection until the split and controls below are committed. Do not execute the hardware generator or contact the remote lab.

Construct splits by joint angle-region blocks within acquisition sessions, retaining time information for drift checks. Reserve entire color/brightness conditions for a separately labeled condition-transfer analysis. Those conditions are not independent-system replication. Use only commanded inputs and declared source settings as predictor features; exclude downstream sensor values and hidden physical models. Normalization must be fitted on training rows only.

Required first controls: constant predictor; additive angle predictor; joint flexible predictor; physics-informed Malus-law model with fitted calibration. Separate known-law assistance from generic representation learning. Compare on identical labeled rows. The first gate asks whether there is useful prediction headroom over these controls, not whether adaptive acquisition wins. If a simple calibrated law solves the task, retain it as a sanity control and stop scaling this dataset.

A later saved-pool selection experiment can reveal only labels of selected recorded actions, charge each revealed label, and compare uniform/space-filling/uncertainty selection under the same predictor. Such a result is pool-based label efficiency, not online physical experiment design. Never interpolate unobserved actions and count the interpolation as an experimental response.

## Language-source intake contribution

This is a second independent publisher/source group alongside Opentrons. The independently authored protocol describes coupled angle settings, fixed nuisance controls, and measurement ordering. It is a candidate source for offline command-contract tasks, not an adjudicated gold fixture. Zero newly adjudicated language tasks and zero model calls. The endpoint discrepancy is a useful ambiguity case: the correct contract must name its source of authority rather than silently resolving a conflict in the model's favor.

Source: https://github.com/juangamella/causal-chamber/tree/0fa8222dc761829270c8959e0ba53b261b075e1c/datasets/lt_malus_v1

## Archive integrity completed, 02 October 03:30 UTC

The 602,229-byte archive matches the published MD5 and has SHA-256 `584490fc05191c21debd75c70c94ee80358f66482694f98c21f405d695f1b3e9`. Durable local raw custody: `/Users/pat/.cache/ace/causal_chambers/lt_malus_v1.zip`; compact audit in `results/causal_chambers_archive_audit_20261001`. All twelve files contain 1000 rows. Checked action metadata has no empty fields; timestamps strictly increase in each file. `blue_128` has 999 distinct angle pairs and `green_128` has 998; all other files have 1000. Splits must group identical commanded pairs within condition to avoid duplicate-action leakage. No outcome statistics were computed; outcome validity, sensor flags and protocol alignment require their own frozen checks. An archive checksum is not a scientific validity result.

Next freeze explicit angle-region splits, condition roles, outcome column, and baseline hyperparameter selection before reading outcomes. Retain both commanded angles (`pol_1`, `pol_2`) as inputs; measured angles are downstream sensors and cannot be substituted without declaring a different prediction task. A blocked split must report coverage and extrapolation separately; ordinary random-row accuracy cannot establish compositional generalization.
