# Foundation component prototype: bounded code review

Date: 2026-10-09. Static review only; no experiments, imports of the prototype, installs, smoke runs, or hypothesis tests. Only this review file was written.

Reviewed working-tree source: `scripts/research/foundation_component_pilot.py` (266 lines).

SHA-256: `9b5e836960868b794dccf8008f09fd26810f69de89d5d5286e5834c5cea44079`

Reviewed protocol: `docs/development/guidance/ace_foundation_component_pilot_2026-10-09.md` (29 lines).

SHA-256: `32fc7b0c0eebc1883fb6ea939406982f22a12dff93daa89fab6037ec05d7fd02`

Both files were untracked at review time. Repository HEAD was `4d7a078a2f868a28b22ff96daffc069f5a74e0d3`; that commit does not identify these reviewed file contents. The hashes above were checked again after reading. This is a prospective prototype review, not approval of a frozen or scored run.

## Required corrections

### 1. Enforce elapsed-time limits during blocking work, not just between cells (P1)

Source lines 197–221, 232, 243–247; protocol line 27.

The only elapsed-time check runs before each cell. Language model loading precedes that check, and generation, fitting, and prediction can each block past the remaining 30-minute allowance. A final cell can overrun and still write `complete.json` without another deadline check. `RLIMIT_CPU` limits process CPU time; it does not enforce an elapsed-time deadline, and it is not an aggregate child-process CPU limit. Imports and preflight also precede the recorded phase timer.

Before executing either technical smoke, supply an independent deadline supervisor or equivalent interruptible execution boundary. Bound each smoke as declared and enforce one shared pilot deadline through loading, fitting, and evaluation. If subprocesses are possible, account for and stop the owned process tree; otherwise explicitly establish the single-process assumption. Preserve terminal evidence on timeout. Do not treat the existing between-cell check as enforcement of the protocol cap.

### 2. Persist the complete planned-cell ledger before failure-prone startup (P1)

Source lines 189–204, 220–225, 250–262; protocol lines 27–29.

Missing or mismatched weights/dependencies fail before the output directory and `started.json` exist. A wall timeout is raised outside the cell exception handler. The CPU limit can terminate the process during work, and a cell has no durable in-progress record until it finishes. Consequently these paths leave neither a disposition for the interrupted cell nor explicit records for the remaining planned cells. The normal completion path records all 30, but that does not satisfy the required failure/stop behavior.

Create a durable 30-cell plan before preflight/model loading, record cell starts, and let an outer supervisor finalize failed, blocked, and unattempted dispositions after interruption without inventing scores. Distinguish a failed preflight from an attempted model fit. Retain completed cells and prohibit a favorable automatic retry.

Also make record writes robust: `write()` creates the final filename before JSON serialization succeeds, and the per-cell write sits outside the exception handler. A serialization failure can therefore leave a truncated, exclusively created result and abort subsequent recording. Validate finite metrics as well as finite predictions, serialize before publishing the final record, and preserve serialization/I/O failures in the terminal receipt. Finite predictions alone do not guarantee finite squared errors.

### 3. Make the freeze gate establish the declared models and protocol (P1 before scoring)

Source lines 189–195; protocol lines 19, 21, 27–29.

The TabPFN checkpoint has an explicit content pin, but the language and dependency checks accept whatever entries the supplied dictionaries contain, including empty dictionaries. The loader then consumes the language directory's configuration, tokenizer/chat template, generation configuration, and weights. No check establishes that all resolved load inputs are covered, that required dependency versions are present, or that the protocol is pinned. All existing checks use `assert`, which disappears under optimized Python execution.

Before freeze, require a validated manifest/schema covering the actual language load inputs and required runtime packages, and preserve provenance tying those pins to the two declared model revisions. Require and verify the protocol pin as well as the source/checkpoint pins. Use explicit failures for mandatory checks. Persist the resolved configuration and actual dependency inventory with the run, rather than relying only on whichever entries happen to be supplied. A separate preparation/launcher gate can provide this, but it must itself be frozen and required; none is established by these two reviewed files.

This finding is about an incomplete verification boundary, not evidence that the currently intended cached weights are wrong.

### 4. Make resource evidence available for failed runs and interpretable (P2)

Source lines 197–204, 252–261; protocol line 29.

Peak RSS is recorded only on normal completion, as `maxrss_native`, without platform or units. Failed or terminated runs lose that measurement. Recorded process CPU starts after imports/preflight, whereas the process CPU resource limit applies to the process's accumulated CPU usage; these quantities do not describe the same interval. The report therefore cannot yet substantiate complete-run resource accounting.

Have the execution boundary retain wall time, CPU accounting scope, and peak RSS with explicit units for every terminal outcome, including startup failure and timeout. Separate startup/loading and cell work if phase timings are retained, and state which interval counts toward the cap. No additional GPU allocation or memory limit is requested by this review.

### 5. Define the promised ratio summaries before freezing (P2)

Protocol line 29; source lines 257–262.

The protocol promises arithmetic and geometric ratios but does not specify the reference arm, numerator/denominator direction, whether an arithmetic summary means a mean of paired ratios or a ratio of means, or how zero errors and failed cells enter a summary. The implementation emits per-cell metrics without those summary definitions. With three endpoints and five methods, choosing these after scoring would leave avoidable discretion.

Freeze the intended comparisons and formulas for each endpoint, including zero/nonfinite and failure handling and the number of paired worlds contributing. Retain all planned cells and identify language fallback cells explicitly; do not silently summarize a favorable successful subset or count fallback outcomes as valid language proposals. Aggregation may remain a separate reporting step; no significance tests or information-matched language contrast are required.

## Scope checks with no required scientific correction

The intervention construction and eligibility masks match the stated design: M has 20 eligible rows and Y has 32. Internal interventions replace M while preserving its observed value as an input to Y. The implementation computes local M, local Y on true parents, and composed Y separately, with normalization from eligible training labels and explicit floor indicators.

No direct private-evaluation leakage into fitting, numerical family selection, or the language prompt was found in the inspected data flow. The language prompt uses the first eight eligible pairs, which here are the observational rows; numerical selection uses the larger eligible histories. That difference is already disclosed as a pipeline comparison and is not a requirement to isolate an LLM information effect. Invalid parsed output falls back to numerical selection and is marked separately; language model loading/generation failures remain failed cells on the handled paths.

This review does not establish package/API compatibility, checkpoint identity, runtime feasibility, or performance. Those remain for the bounded technical smokes and subsequent freeze after the corrections above. No expansion to acquisition experiments or prior accepted-study reviews is requested.
