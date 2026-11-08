# Realization02 recovery authoring handoff

Stable authoring snapshot; all four scripts complete and ready for independent recovery-delta review. No unfinished collector block. Predecessor01 scripts are preserved byte-for-byte under `predecessor01/`; `recovery02-vs01.diff` is the complete source delta. Parent-owned original files were not edited. This is authoring, not runtime qualification or submission approval.

## Independent inputs and API

`build_packet.py` requires `--source-commit` and path plus `--NAME-sha256` for each of: `freeze`, `freeze-review`, `adoption`, `namespace-admission`, `predecessor-freeze`, `failure-acceptance`, `no-draw-proof`, `failure-inventory`, `failure-readback`, `recovery-authority`. Also required: `--failure-custody`, pointing at either accepted complete terminal01 copy. The two inventory inputs are the original `inventory_transport.json` and `inventory_after_transport.json`, not reconstructed inventories. No fake defaults for successor freeze/source/adoption/authority. Accepted baseline is the independently pinned failure-acceptance01/accounting.json (63cd36df...), containing 39 jobs/80 journals,11048 CPU/1601 GPU,3600 preparation,zero outstanding. New reservation is 600 CPU only:15248 CPU/1601 GPU including preparation.

`authority-interface.md` specifies the exact authority/namespace/adoption objects. The authority object has 18 keys (the concrete adoption02 object matches the authored shape). Local read-only inspection of adoption02 found the expected fields/pins; no build or authority-consuming helper was run. The adopted candidate03/source are supplied through CLI; neither candidate02 nor candidate03 freeze hash is hardcoded in source.

`launch_once.py`: required `--packet-commit`, `--packet-sha256`, `--packet-review`, `--packet-review-sha256`; run only the committed concrete launcher. `monitor_once.py`: required `--packet-sha256`, `--submission-result-sha256`, `--evidence-directory` (exclusive). Monitor bytes are unchanged from the accepted timeout-corrected predecessor; it imports the new colocated builder constants. `pull_terminal.py`: required `--job-id`, `--packet-sha256`, `--submission-result-sha256`; standalone, current unambiguous terminal job only.

## Recovery delta

The full67-source metadata map must retain exactly the four authorized infrastructure path corrections; the actual realize packet still captures22 source files plus original proposal. Same32/24 schedule/science/design,2CPU7680MiB,300sec,120/260/280/290,ucb736_asc2/acpu/cpu-normal. Existing parent static constant resolver is retained; no project module executes during source capture.

Builder authenticates the accepted failure/no-draw proof, all41 original raw object names/bytes and directories from both accepted inventories, original local claim, all committed sources, successor freeze and independent adoption/authority. Future build invocation acquires both coordinator locks, records a read-only original remote registry check in both exclusive packet directories, and never claims a namespace. Build failures remain exclusive and nonretryable.

Launcher keeps both coordinator locks, validates all committed packet members and authority, reads the unchanged physical old local registry, and requires the only prior realization root to be accepted01. Both remote readiness passes compare all39 rows/all80 journal pins, no active ACE jobs, complete41 failed raw objects, and unchanged original remote claim before mutation/submission. Only new recovery02 registry locations are created exclusively. New local/remote claims and intents bind authority,failed job,failure acceptance,no-draw proof,original claim and successor freeze. Exactly one sbatch call; any uncertain submission retains intent/registries/partial timeout output and forbids retry. Original registries/root are never written.

Collector captures all new ROOT bytes plus full failed01 ROOT and both remote claims, both local claims, and complete new local/external coordinator histories. Accepted original41 roster is checked without sampling/truncation. Second remote readback compares all raw file hashes/directories/hardlink metadata, exact scheduler rows and journals. Two independent local byte copies; original hardlinks, if any, are represented as metadata with independent copies. Parent's authenticated remote-intent/local-intent semantic join is retained. Failed/missing stage outputs on a terminal job are preserved without scientific JSON decoding or acceptance. Current job lineage and requested/actual resources remain checked. Expected total after a new terminal job is40 jobs/82 journals; charges are11048+actualCPU and1601+actualGPU, never success-only accounting. Binding discrepancies retain custody and fail closed. No output status implies scientific/realization acceptance.

## Paths and collector pin meanings

Remote root: `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02`.
Repo: `results/ace_llm_intervention_scientific_realization_20261108/launch02` and `terminal02`.
External: `ACE-LLM-intervention-scientific-realization-20261108-02` and `ACE-LLM-intervention-scientific-realization-terminal-20261108-02` under `/Users/pat/ACE_Study_Results/2026-10-peter-baseline`.
Stage outputs are `ROOT/realization`; task outputs `ROOT/realization/session/worker/task`.
Collector `--packet-sha256` means **outer `ROOT/packet.json`**, not inner `ROOT/inputs/packet.json`. The outer pin covers the inner packet and admission pins. Collector raw layout: `remote/launch/` (new root), `remote/failed01/` (complete old root), `remote/namespace.json` (new recovery claim), `remote/original-namespace.json` (original claim), with the two local claims in `namespaces/`.

## Validation and limits

Only AST parse/compile, including embedded remote Python snippets; no target/project imports or script invocation. `static01/` retains exact checker, stdout/stderr, receipt and measured local process/wall costs. No tests, old checks, packet build, output/namespace claim, remote operation, allocation, world draw, model/tokenizer, or scientific output decode occurred. No repository source/Git changes. Static success does not establish end-to-end runtime behavior; parent independent authoring and concrete packet reviews remain required. Collection finalization stays paused.

## Exact source pins

- `build_packet.py`: `b67e2ac29bd4d06a540d8358179a2dcf63b12bd297141f990750b88b674371cb`
- `launch_once.py`: `135b21ed12676a10b8d28ef01c7a8da754af2a0a33a4db15b64e072e25a10543`
- `monitor_once.py`: `1ac08e52ecbb8dd85c6a28b3ad53fa0f44d952d89f78e6dfac22924a2af8f3c2`
- `pull_terminal.py`: `deebd55fb60d7d3b426fdd5919c9f7d15484c817cbad7fbf5f38804c8e4d86ee`

## Read-only adoption02 metadata pins

- `adoption.json`: `ea0457db2d927101366bdd3763aadf7fe4884bb91ef16d5efeb8e91d0b6e5a07`
- `namespace-admission.json`: `5027bca99d1c1cdfc9cf89dd5639593308e8eb37f7f7ad2fb8aacb6ded5c0e36`
- `recovery-authority.json`: `ba6ef3937d4aa3cad68a64ae7f02e098041c5bddff5e68d5e5b0833f30cd2a50`
- `freeze-review.md`: `088b1bc2cdf5855b1b2f285b3ac751813cd5655844ced202fd956cccb29ff416`
