# Paired intervention execution and private reporting

The new paired-world runner connects a supplied SCM, public intervention policies,
charged responses, sealed histories and the reviewed cost reports. Its purpose is
to compare policies on the same system without losing failed runs. It does not
generate the scientific training or development worlds, train a policy or qualify
the target model runtime.

The interface has three steps. `prepare_plan` fixes the supplied world digest,
public task, learner, legal menu, initial schedule, acquisition cap, response-stream
bindings and ordered policy roster. Each policy has an explicit command, timeout
and provenance pin. Persistent workers also have an expected model/service identity.
`run_world_arm` executes one of those arms once. `finalize_paired_world` authenticates
the retained arm outputs, checks shared initialization and creates private reports
for the full roster. The initial plan is independently pinned at each step.

Arm execution and finalization are separate so a scheduler can run numerical arms
on CPU first, then reuse a single language-model service across supplied worlds.
The runner does not allocate a GPU or start a persistent service. The caller owns
its startup, source/model/runtime authentication, resource limits, monitoring and
cleanup. It must close persistent workers before final reporting. A clean closure
receipt must match the recorded process identity and an independent byte pin.
Failure or missing closure leaves that arm unresolved at the common cost cap.
This does not by itself qualify an LLM or prove its GPU utilization.

Only the existing `TaskProcessPolicy` or `PersistentPolicy` interface can serve an
arm. The live command, timeout and identity must match the declared specification.
Only the validated public task envelope reaches the policy. Truth coefficients,
private errors and other arms' trajectories are not API inputs to that process.
This is trusted process separation, not a sandbox against malicious programs with
the same filesystem privileges.

Each arm gets a fresh learner and its own charged executor. The initial response
keys are shared, while acquisition measurement bindings are distinct across arms.
Every initial and acquired response is reserved in that arm's journal before the
environment is called. There is no uncharged response cache or extra preliminary
initial draw. Repeated initial keys produce paired values but each arm still calls
the supplied environment adapter and pays its declared logical cost.

The wrapper records the exact returned JSON-compatible values before passing them
to the executor. A prepared simulator outcome, wrapper-observed return and accepted
executor receipt are different events. The separate sealing adapter validates
their hashes, order, intervention masks, charges and terminal state. An exception
after a simulator prepares a value but before the wrapper sees it remains an
unknown return. A malformed value remains known and rejected where exact returned
JSON can be retained. Unsealable evidence remains an unresolved capped roster cell.

Normal episode failures are retained and returned as raw evidence, allowing other
predeclared arms to run. There is no retry, alternate policy or replacement world.
Interruptions are preserved and propagated. The outer supervisor must handle hard
termination, disk errors and process deadlines. An arm receipt means only that
the retained raw artifacts are ready for validation. It does not mean the arm
succeeded or that the experiment is scientifically accepted.

Shared initialization is checked before computing any private error. Complete
initial histories first become candidates. The sealer must validate their entire
execution chain before they can serve as witnesses. All valid complete histories
must agree exactly. Conflicting witnesses leave every arm in that world unresolved,
rather than choosing one as authoritative. Partial histories are then checked
against the common witness. If no full initial history exists, all cells remain
unresolved and their original charge journals remain available. No missing initial
response is generated or inferred to make reporting possible.

After those gates, the private reporter reconstructs errors from accepted receipts
and the fixed query law. It reports first-crossing cost, final error, actual charges,
attainment and failures. Every declared policy enters the roster summary, including
missing outputs. Failed or unresolved cells receive the common acquisition cap.
Actual partial charges remain separate. A retrospective private threshold crossing
is not a deployable stopping rule and is never sent back to a policy.

The component accepts artificial and development scopes only. Source and runtime
admission, independently frozen world namespaces, original receipt custody and
an authenticated scientific policy specification remain outer obligations. Pins
bind bytes, not the truth of provenance assertions. No scientific acceptance or
protocol-freeze flag is set by this interface.

The proposed 24-world screen still needs a complete supervising driver, exact
policy construction and a resource freeze. In particular, the existing numerical
CLI derives its decision seed with a SeedSequence of seed and request count. The
prospective screen describes a keyed decision-index law. Those are not identical
and must be explicitly reconciled before adoption. The caller must also bind SFT
and DPO checkpoint lineage, tokenized training data and persistent-service cleanup
across all worlds. This integration does not silently change either seed rule.

New integration tests use a hand-specified one-node artificial SCM, real small
public-policy subprocesses and one model-free persistent worker. The private curve
has a hand-computed conjugate-regression oracle. Other controls exercise an invalid
policy, missing arm, unknown return after a prepared outcome, invalid returned
measurements, partial initialization, conflicting paired histories, changed custody
pins and failed service cleanup. These tests create new artificial responses and
SCM updates. They do not draw scientific parameters, load pretrained weights,
perform neural policy training, rerun accepted studies or repeat older test suites.
