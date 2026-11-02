# A shared task and learned numerical control

Status: isolated implementation and local artificial checks. The scientific training, development and confirmation namespaces remain ungenerated. This changes neither the archived delivery manuscript nor the accepted GPU04 packet.

## The comparison this enables

The experimentalist supplies a target, the interventions at which predictions are useful, their probability weights, an error scale and an accuracy goal. Every policy receives that same task together with the same paid history, legal intervention menu and current SCM. The policy chooses a charged intervention or stops. The environment then returns only the requested measurements.

The numerical control learns to rank actions from this information. It provides a comparison to learned acquisition without a language model. It complements the target-risk planner, which estimates the value of each intervention explicitly. These are different controls. The planner tests whether language beats a strong numerical decision rule. The learned control tests whether learning a cheap action rule already explains the benefit.

Neither control is declared optimal or empirically superior. Their performance must be measured on the same fresh systems as the LLM. If the LLM only beats random, that will not meet the agenda's stronger claim.

## One public envelope

`llm_intervention_task.py` wraps the existing charged state in `ace-public-prediction-task-v1`. The task includes only a target node, legal prediction queries, positive weights summing to one, a fixed positive scale and a positive normalized-MSE tolerance. No answer, evaluation error, hidden coefficient or private seed is a task field.

The policy prompt states the objective explicitly: predict the noise-disabled structural response, with squared error divided by the scale squared. The tolerance describes the desired answer. It does not certify that the learner has reached it. A policy cannot consult a private evaluation score when deciding to stop.

The validator checks the full structured allowlist, including graph, menu, mechanism summaries, histories and queries. Receipt identities, intervention masks, measured values, charges and eligible row counts must agree. Failed or ambiguous response histories cannot become new policy inputs. Extra nested fields are rejected. This rejects accidental structured disclosure. It cannot detect a dishonest claim hidden in free-form text or sandbox a malicious same-user program. The frozen world runner must still establish the provenance of every public field.

`TaskProcessPolicy` uses the existing process boundary. `run_task_episode` saves the task and its digest before initial measurements, including when initialization consumes the budget and no policy decision occurs. The original executor still handles reservation, action validation, observations and failures. No old worker or frozen packet is amended.

The same envelope feeds the random, coverage, local-information, target-risk and learned numerical process entrypoints. The new LLM prompt renderer accepts it too. Actual Qwen tokenization, adapter reload, inference and multi-episode handling of this new prompt are not yet qualified by GPU04's older artificial prompt.

## What the numerical learner sees

The fixed representation allows up to eight nodes, 32 menu actions, 64 public prediction queries and 80 returned responses. Exceeding a bound raises an error. There is no silent truncation. These are implementation limits, not the scientific world's frozen dimensions.

Each candidate row contains the complete bounded numerical state: graph adjacency, intervention bounds, known noise, eligible row counts, posterior means, full within-head covariance, menu values and masks, each paid measurement and hard clamp, the public query law, scale, tolerance and budget. It also contains the candidate's intervention values, requested measurements and cost. Stop is a separate candidate. Only affordable interventions are ranked. Padding has explicit presence masks.

The representation has 8,891 features per candidate. It applies a fixed signed-log transformation without fitting normalization statistics on other worlds. Numerical model computation uses float32. Names, units written as text and domain prose are omitted. Renaming a node consistently does not change the features. Reordering the graph's valid topological order can change them. The flattened network is not a permutation-invariant graph architecture, and this limitation must be reported or addressed if transfer across graph order is a central claim.

No target-risk acquisition scores, true environment parameters or training reward labels are included as features. Numerical facts in future domain notes must also be available in the structured state. Early synthetic task prose should be derived only from the public graph and task. A later semantic-prior study needs explicit controls for the additional information conveyed by language.

## How it learns

`llm_intervention_ranker.py` uses a shared two-layer tanh network to score each candidate row, with 128 hidden units per layer by default. At the full feature width it has 1,154,817 trainable parameters. This is similar in count to the tested Qwen adapter, but it does not make the models equal in architecture, total parameters, pretraining or compute.

The supervised phase minimizes cross-entropy on the labeled action among the feasible candidates. The preference phase uses the chosen and rejected actions at the same state, compared with a frozen copy of the supervised numerical policy. The reference copy has independent parameter storage and no gradients. It stays fixed as the current policy changes.

The scientific runner must supply the same training states, candidate menus, supervised choices and strict preference pairs to the numerical and LLM arms. Numerical classification and token-level language training are different objectives even with the same labels. Record their actual exposures, steps, costs and failures. Do not describe this as equal compute or a pure causal estimate of pretraining benefit. A specific pretraining claim would need an architecture-matched control or narrower wording.

Training uses fixed final iterates. It does not choose a seed or checkpoint by private performance. Fitting rejects development and confirmation scopes. An artificial scope exists only for interface checks. Scope strings alone do not prove world independence, which remains a requirement of the frozen runner and source review.

Ties must remain in the full supervision dataset. The fitting primitive receives only strict pairs and reports states with no strict preference. It rejects a request for preference updates when no strict pair exists. Duplicate, contradictory or malformed pairs are rejected instead of repaired.

The checkpoint stores the architecture, representation version, trained weights and the training-receipt digest. Loading first checks an independently supplied checkpoint digest, then checks the receipt pin, schema, feature width, exact parameter keys and finite weights. The scientific runner must independently authenticate the actual receipt and training sources. Passing arbitrary matching strings is not training provenance.

## Stopping and useful answers

The learned policy and LLM may emit stop. That must not manufacture an efficiency gain. If a policy stops without reaching the private useful-accuracy threshold, report nonattainment and the common capped cost under the frozen endpoint rule. Actual units spent are also reported, but cannot replace the censored cost in the primary comparison. A retrospective private threshold crossing is not a deployable stopping certificate.

Every selected intervention remains in the same charged executor. The initial data, returned measurements, errors and failures remain paired by system. Unselected candidate actions receive no free environment outcomes at deployment. Model-only planning calls and their computation are reported separately.

## Evidence and remaining work

The first preparation passed ten new artificial methods. These cover the shared task and numeric representation, nested-field rejection, counter and eligibility mismatches, covariance and history sensitivity, consistent renaming, affordability, exact loss and preference-gradient formulas, independent reference storage, fixed training, and checkpoint pins. A charged target-risk episode and a zero-decision episode exercise the new process path.

A separate new method checks the full-width learned-policy checkpoint through the same process path. Its weights are deliberately zero so the expected action follows the declared tie rule. That checks identity and execution, not learning or utility. The earlier numerical training fixture does update a small neural model on artificial labels. No pretrained model or scientific system was used.

The first command used 1.129219 child CPU seconds, 1.471296167 elapsed seconds and 315179008 peak child RSS bytes. The separate process fixture used 1.010986 child CPU seconds, 1.022328875 elapsed seconds and 223526912 peak child RSS bytes. These are local preparation costs, not CURC allocation or whole-cycle CPU. Exact source and logs are in `results/ace_llm_intervention_task_ranker_preparation_20261102` and exclusive external originals.

The next scientific path is to qualify the complete new prompt and checkpoint interfaces, generate charged supervision on separately frozen training worlds, fit both policy families, then evaluate on untouched development systems. Final confirmation and the engineering task remain separate. Production resource requests require actual demand measurements and an eligible production QOS. No new allocation has been made for these modules.

## Review and correction

Independent review identified two source defects. Episode size limits are now checked before any response can be charged, including a zero-decision episode. Numerical initialization now uses only CPU RNG state and explicitly CPU/float32 layers. Both findings are closed. Two new regression methods and two affected methods pass in preparation03, using1.078357childCPU seconds,1.389958elapsed seconds and313147392peakRSS bytes. There are thirteen distinct passing task/ranker methods across the preserved attempts. See [review](reviews/ace_llm_intervention_shared_task_review_2026-11-02.md).
