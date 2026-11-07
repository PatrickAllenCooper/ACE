**Independent bounded screen-policy review, November 7, 2026.**

**Required findings: 0.** The new adapter agrees with the prospective screen's
declared numerical behavior keys and the existing public task transport within
the scope below. This is component review, not scientific admission, target
runtime qualification or evidence that any policy reduces experimental cost.

Read [screen policy](/Users/pat/code/ACE/scripts/research/llm_intervention_screen_policy.py),
its new test file and [guide](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_screen_policy_2026-11-07.md).
Compared the necessary interfaces in the existing screen protocol, world seed
function, public task validator, episode driver and four numerical algorithms.
Inspected the retained preparation receipt, process records, rejections and
source copies. Used only static source inspection and stdlib reading, hashing
and comparison of saved JSON. No test, policy, model, working-model simulation,
scientific world generation, target-runtime import or remote operation was
executed. No commit was made. Only this review document was authored.

**Exact key and seed agreement.** The adapter reconstructs the same trajectory
namespace as the protocol's development episode:

```text
SHA256(canonical({protocol, scope, world_id, purpose: "trajectory", arm}))
```

It requires the prefix `[scope, world_id, "policy", namespace, "behavior"]`
and appends the derived decision index once. The existing `world.keyed_seed`
therefore hashes exactly:

```text
["ace-linear-gaussian-world-v1", scope, world_id, "policy",
 [namespace, "behavior", decision_index]]
```

The tuple in `keyed_seed` and the list in the receipt have identical canonical
JSON. The full SHA-256 digest becomes the big-endian seed integer. The adapter
does not truncate it, append the arm a second time or apply the historical CLI's
request-count `SeedSequence` conversion. The existing algorithms retain their
own internal RNG behavior, including target-risk's action-specific streams.
The exact-seed claim concerns their supplied seed, not removal of those internal
algorithm steps. Decimal seed strings and fixed-width hexadecimal receipts avoid
loss through a downstream floating-point JSON number.

Independently reconstructed the namespace, key JSON and seed digest for all
eight saved successful subprocess calls without invoking policy code or an RNG.
All agree. The 96 saved development bindings have the expected 24 declared
world IDs, four controls per ID, 96 distinct arm-specific namespaces and the
exact protocol prefix. This comparison used declared metadata only. It did not
realize any development world or qualify the fixture's placeholder source pins.

**Initial history, decisions and charges stay distinct.** `unpack_envelope`
requires complete public histories before selection. Requests, returned responses
and history length agree. Receipt IDs are contiguous, actions/clamps/measurements
and costs match the menu, and eligible-row counts agree with the receipts.
The adapter then checks the fixed initial action prefix, both versions of its
cost sum, the total budget and maximum response bound. Only after these checks
does it compute `requests - len(initial_actions)`.

For the development schedule, eight initial responses therefore precede decision
zero. One subsequent two-unit intervention leads to decision one, nine requests
and ten spent units. The saved smaller fixture demonstrates the same separation:
indices zero/one correspond to requests two/three and spent units two/four for
all four controls. It does not subtract initial cost from the request count or
use acquisition cost as a seed index.

This agrees with the existing episode loop, which starts its decision counter
after initialization and makes one charged executor call per successful choice.
Failures terminate that loop. A terminal input or an input without an affordable
action rejects before selection. Normal budget stopping remains with the
executor. The optional in-process counter check rejects disagreement. The CLI
cannot establish that an identical input was never secretly retried. The guide
correctly assigns call-index, process-file and journal reconciliation to the
outer supervisor.

**Actual transport and algorithm scope.** `TaskProcessPolicy` wraps the state
with its immutable prediction task and forwards exactly `{version, state, task}`.
The new CLI unpacks that envelope and emits one decision JSON object on stdout.
It writes the key, binding/input/decision hashes, counters and planning metadata
to stderr, which the unchanged `ProcessPolicy` preserves with its command,
input hash and return code. All eight retained successful calls have matching
input, binding and decision hashes and return zero.

The three original controls receive the validated state and exact seed.
Target-risk also receives the public task's target, queries, weights and scale.
Both planning controls receive 32 draws per feasible action. Random and coverage
retain zero fantasy draws. The one-node fixture offers two feasible actions, so
both planning receipts count 64 fantasies. Its target-risk values also admit a
direct arithmetic check: posterior variance `1/3` at scale two gives risk `1/12`.
An observation reduces it to `1/16`, while clamping that sole head does not update
it. The saved choice of observation and reduction `1/48` agree. These are public
working-model calculations, not measured utility or private outcomes.

The saved inputs are two fabricated completed histories per control, not eight
complete trajectories. The tests' calls to existing pure algorithms provide
algorithm-agreement evidence. They were not repeated during this review.

**Public information and rejection behavior.** The binding contains only
declared identity, RNG, initial-schedule, budget and draw metadata. The adapter
uses no private world descriptor, mechanism coefficient, evaluator, training
feedback or environment callback. Public paid measurements and fitted public
SCM summaries remain legitimate inputs. Working-model fantasies and temporary
conjugate updates are not environment responses.

The exact binding schema, independent supplied digest, reconstruction and
canonical comparison reject changed versions, extra fields, mismatched
prefix/namespace/arm, invalid numeric types and altered draws. Only artificial
or declared-development numerical behavior bindings are accepted. Training,
confirmation, teacher/snapshot roles and learned arms reject explicitly.
Development additionally fixes the eight initial observations and 32-unit
budget. The separate training behavior namespace and teacher snapshot index
are not inferred or silently substituted here.

Saved evidence contains 18 binding rejections, 13 public-history/counter
rejections, three development-schedule rejections and one budget-terminal
rejection. Five additional CLI failures cover duplicate keys, nonfinite JSON,
oversized input, an extra private field and a wrong binding pin. Each retained
CLI failure has exit two, empty stdout and a rejected receipt. The wrong-pin
case crosses the actual `TaskProcessPolicy` boundary and propagates the existing
process failure. There is no fallback or arm substitution.

The caller must still supply honest public prose and authenticate the graph,
menu, task, prior, scope and identity against the adopted proposal. Schema checks
are not a same-UID sandbox or an independent replay of every posterior entry.
These limits are stated in the guide and are outside this adapter's scope.

**Retained evidence and pins.** Verified all 95 manifest members and exact
membership of the 96-file tree including its manifest, totaling 2,602,535 bytes.
All files match the corresponding exclusive external custody copy at
`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-screen-policy-preparation-20261107-01`.
All 19 source copies match their recorded sizes/digests and current source files.
The recorded single test command selected the new suite, and its stderr records
nine methods passing. No rerun is recorded or was performed here.

Author-run measurements are 1.296824 child CPU seconds, 1.441983875 elapsed
seconds, 47,759,360 bytes peak child RSS and a separately measured 0.002283
supervisor CPU seconds. Child CPU includes waited descendants. Peak child RSS
is not summed simultaneous process-tree memory. These local Python 3.9.6/NumPy
2.0.2 measurements exclude preparation, review and custody work and do not
qualify the target runtime or reset the cumulative preparation allowance.

- Screen policy:
  `3d34247e476117e6521528a889780948598117462b7aec53b7179e8f515bf67f`.
- New test source:
  `1898fd7d424c6ccf4f090ed67555a482cbbcc6bfd25394e2f445043fdaa11a83`.
- Current guide and saved final guide:
  `85dff4142c68d8173a7f946a3de841ce497b57c31d71b0b8903cdd2185240c24`.
- Preparation manifest:
  `953d51d6a50f36fab1c0e526a391191388450f187ebd1fcb5b292d758a11d5ba`.
- Invocation receipt:
  `61c5bfb75dae472c2331b06123a26bf1730b87a8c9fe98217ef3011fb006787f`.
- Completion record:
  `aa03f6e6772fc5a22ca0d2350e727daa8259077caa018f7f0061dccf95f4c645`.
- Source-pin map:
  `2fd430284d9dd3b4d27ae57158d6b82347f2db15a35a4c6b53ac2732e8575ba8`.
- Saved test stderr:
  `fb8dcbd809f007b8fbd2bb86045f86d2503ebde4de58e3876b6e20cc6cf8c588`.

The preparation's earlier review-pending wording remains historical. This
separate document supplies the bounded independent component disposition.
Complete source/runtime/controller integration, binding each policy command to
the paired plan, preserving the distinct shared initial namespace, multiworld
supervision, full scientific freeze and training/teacher integration remain
external requirements. No LLM benefit or intervention-efficiency conclusion
follows from these artificial interface checks.
