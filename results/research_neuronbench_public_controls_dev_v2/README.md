# Five-world NeuronBench public-control development run

Frozen [v2 protocol](../../docs/development/guidance/protocol_neuronbench_public_controls_dev_v2_2026-09-28.json), public questions and ten plans committed before outcome jobs at ACE revision `a70c815cc190ee50999db338af9d9b1ad8b9f225`. Pinned upstream NeuronBench `c354622458c460b419cab821d482c879f0578377`. CURC checkout `/scratch/alpine/paco0228/ACE/code_neuronbench_controls_a70c815`; isolated Python `/scratch/alpine/paco0228/ACE/envs/neuronbench_py311`; output `/scratch/alpine/paco0228/ACE/results/research_neuronbench_public_controls_dev_v2`. Jobs `33085652`–`33085661` on account `ucb736_asc1`, all COMPLETED 0:0 in 13–21 seconds, with empty stderr. No closed-model API calls.

Each method used four distinct public actions per world at exact cost 4, with no acquisition/forecast overlap, six complete heldout labels, and the common fixed-penalty public-only ridge forecaster. Independent local validation matched plan/problem/source hashes, receipt hashes, private-file hashes, exact action cost, score parity, and CURC/local checksums.

Official floored spike forecast MSE (lower is better):

| Designed world | Random | Public waveform coverage |
|---|---:|---:|
| h_sag | 45.4633 | 2.2961 |
| na_fatigue | 493.8772 | 88.8386 |
| ca_rebound | 22.1660 | 28.2513 |
| d_type | 4.0092 | 61.2919 |
| textbook_M | 1.0463 | 0.6943 |

Coverage wins three of five new designed worlds and loses two; the separately labeled `z_rebound` v1 canary favored coverage 3.4100 versus 51.8401. These six worlds are a designed development collection, not independent population draws or fresh confirmation. The weak control has no mechanistic hypothesis library or foundation model. The large `d_type` reversal and large `na_fatigue` errors show that generic waveform coverage plus a four-sample ridge predictor is unreliable; no pooled superiority claim or architecture promotion follows. The v2 eligibility correction was made after discovering benchmark menu overlap, before any five-world outcome job.
