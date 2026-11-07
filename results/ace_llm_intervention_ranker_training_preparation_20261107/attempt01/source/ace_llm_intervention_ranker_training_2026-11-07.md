# Numerical training with matched examples

The new CPU trainer consumes the exact reviewed exposure proposal used to prepare
the language-model schedule. It does not use the older numerical trainer's default
eight supervised epochs and four preference epochs. The old implementation and its
evidence remain unchanged. This is an isolated prospective training component, not
an adopted scientific recipe or an intervention-efficiency result.

`prepare_plan` accepts complete raw matched records, independent record and receipt
pins, an ordered roster, the exposure proposal and its independent digest, an explicit
numeric recipe, checkpoint indices and an outer provenance receipt pin. It rebuilds
the exposure proposal and requires exact equality, then derives the frozen complete
numerical features. The plan retains raw records and hashes the derived FP32 tensors.
It does not persist a second large JSON copy of those padded tensors. `fit` rebuilds
and checks all of this before initializing parameters. A pin cannot establish that
the original training worlds, responses or source/runtime were properly admitted.

The proposed scientific control is the existing CandidateRanker: a 128-unit tanh
MLP, initialized once at seed109, with the fixed complete signed-log numerical state
encoding. It uses CPU FP32, one Torch thread and strict deterministic algorithms.
The explicit Adam recipe is learning rate .001, betas(.9,.999), epsilon1e-8,
weight decay0, no clipping, no fused/foreach implementation and beta.1 for DPO.
One optimizer carries moments across both stages. These settings are a prospective
control recipe, not a tuned winner. The caller must explicitly request the recipe.
An artificial-only variant uses four hidden units for small control tests. It is
rejected for policy-training scope. The caller must set thread and determinism
settings before execution. No global Torch or system settings are changed by the API.

Each record receives its proposed supervised step. Each selected strict pair receives
its proposed DPO step, after all supervised steps. The supervised ranker is copied
and frozen at that boundary. Its full per-record log probabilities are cached and
checked again at completion. DPO uses the same chosen/rejected pair as the language
model. There is no inferred epoch count, batching, extra label exposure, retry,
checkpoint selection or private-evaluation access.

The two objectives differ. Numerical supervision is categorical cross-entropy over
legal actions. LLM supervision is mean action-token loss. Numerical DPO uses
categorical action log probabilities, while LLM DPO uses complete command-sequence
log probabilities. The same examples and pair directions do not imply equal losses,
parameters, compute or gradient magnitude. The numerical encoding omits variable
names, units and domain prose, as already disclosed by the shared-task design.

A terminal state with only the stop action gives the numerical model exactly zero
supervised loss and gradient. That row is retained and receives its declared Adam
step. Existing optimizer moments can still change parameters on such a step.
Zero-gradient events are counted explicitly. They are not dropped as failed data
or presented as informative supervision. Missing or nonfinite gradients do fail.

The exclusive output preserves the input plan, initialization timestamps, initial
parameters, supervised parameters and optimizer, fixed reference, every step's
loss/gradient/update intent and return, and final parameters/optimizer. Intermediate
checkpoints occur only at declared indices. A raising optimizer call has an unknown
or partial outcome and retains current state. Any failure preserves the completed
prefix and current model/optimizer/gradients/reference where storage works, with no
automatic rollback or resume. The outer process must handle hard termination and
storage failure, and must preserve preflight rejection inputs itself.

Training completion is saved as `training_complete_export_pending`. The trainer
then exports both supervised and final policies with the existing ranker serializer,
bound to that training receipt. Each export is loaded into a fresh CPU ranker and
must reproduce all training-state scores exactly. Only after both checks does the
final `terminal.json` say complete. This is a numeric export/reload check, not private
evaluation. Original tensors have role-specific checkpoint hashes in the receipt.
An outer admission process must still authenticate that lineage before deployment.

The input bounds are inherited from the exposure planner: at most256 records and
512 updates. Feature width is at most20,000 and the initialized network has at most
3,000,000 parameters. These are code bounds, not a measured scientific resource
reservation. CPU preprocessing, feature reconstruction, training, copies and reloads
all count toward the later execution projection. Full plan validation occurs before
the API's training elapsed timer, so the outer command receipt must include it.

New local tests use handcrafted public states and newly initialized tiny numerical
rankers. They compare the saved result with an independently written literal Adam
and DPO loop, exercise the zero-gradient stop row, and retain injected update,
reference, nonfinite-loss and export failures. They do not load a pretrained model,
call an environment, generate a scientific world or rerun earlier suites. An
artificial test pass does not qualify the target CURC runtime or first-screen data.
Source, inputs, failures and command costs belong to the separately preserved
preparation capsule. Independent review and integrated runner qualification remain
required before scientific use under the existing project authority.
