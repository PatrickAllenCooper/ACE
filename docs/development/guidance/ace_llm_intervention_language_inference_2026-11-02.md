# Reloading an LLM intervention policy on the shared prediction task

## Purpose

The successful GPU04 smoke established one real Qwen adapter training and action path. It did not establish that a saved adapter could be restored for a new task prompt or used across experimental episodes. `llm_intervention_language_policy.py` supplies the next reusable primitives. Its current tests are local and artificial. No new scientific policy has been trained or evaluated.

The intended comparison gives frozen, supervised and preference-trained LLMs the same public task, legal interventions, paid observations and fitted SCM summaries that numerical controls receive. Each arm uses a separately frozen model state. This implementation does not choose a winning model or checkpoint from scientific scores.

## Adapter restoration

The loader requires independently pinned raw checkpoint bytes, a training receipt digest, the exact projection names, rank and scaling. It checks the raw bytes before `weights_only` deserialization. It checks every key, tensor shape, FP32 dtype and finite value before changing the model. The base must be on CPU and must not already contain adapters.

After validation, the loader installs the existing q/v adapters, copies the saved tensors, snapshots reference buffers, and freezes the model for inference. Adapter constructors run within an isolated CPU random state. Global accelerator seed functions are not called. The same base weights remain in the wrapped linear projections.

These checks cannot authenticate a base model merely because its projection names match. The worker must separately bind and verify the model files, source, architecture, runtime and actual training receipt. The returned metadata explicitly leaves that provenance unqualified. The library does not load a pretrained model, allocate a GPU or execute an intervention.

## Shared prompts and legal commands

The language prompt serializes the same validated public envelope used by the numerical policy. It identifies the target, query law, prediction scale and desired accuracy. It explains that the accuracy goal is not evidence that the goal has been met. Private evaluation cannot be used as an observed stopping signal.

Tokenization applies the chat template with thinking disabled and adds the assistant generation prefix. Prompt tokens are never truncated. The complete prompt plus the declared generation allowance must fit the explicit context limit and the model's configured maximum. Every affordable intervention and stop enters the finite grammar. A terminal state admits stop only. No preferred action or reward enters the grammar.

`training_action_batch` uses precisely that prompt and the same command tokens followed by EOS. Prompt and padding positions are masked from the supervised label loss. Labels that do not correspond to feasible deployment commands are rejected. Thus the training converter and decoder have an explicit common representation. Actual model/tokenizer and training-batch qualification is still required.

`generate_task_decision` requires a frozen model on one device. It uses a fresh greedy generation configuration with one beam and one unpadded prompt. The callback is bound to that exact prompt. Nonfinite legal-command scores, malformed prefixes, missing EOS and unavailable commands fail. A successful return contains the original prompt tokens and complete generated sequence. A worker must persist that evidence before the charged executor receives the command, and must retain failures and interrupted attempts. This library is not that worker or a sandbox.

## Local checks and limits

Six new artificial methods pass. They cover an independent linear adapter oracle, rejection before mutation, CPU RNG preservation, common training and deployment tokens, invalid generation and finite-score rejection, and terminal/budget command sets. Small fabricated tensors and a fake generator were used. The generator test imports the installed Transformers generation configuration, but it does not instantiate Qwen or test the real generation loop. No neural training, pretrained weights or scientific worlds were used.

The six-method command used 0.959031 child CPU seconds, 1.24656125 elapsed seconds and 289652736 peak child RSS bytes. These are one command's costs. They exclude implementation and review overhead.

## Measured prompt sizes before another GPU allocation

A separate CPU-only profile used five authenticated files from the cached Qwen3-0.6B tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca`. Their hashes match the accepted GPU04 input freeze. No model weights were copied or loaded. The local versions are Transformers 4.51.3, tokenizers 0.21.4 and torch 2.14.1. These differ from the target environment and do not qualify it.

Five declared artificial configurations produced the following prompt lengths:

- Two nodes, two interventions, one paid row and two target queries: 613 tokens.
- Two nodes, two interventions, eight paid rows and two target queries: 877 tokens.
- Five nodes, eight interventions, eight paid rows and eight target queries: 2526 tokens.
- Five nodes, eight interventions, 32 paid rows and eight target queries: 4341 tokens.
- Eight nodes, 32 interventions, 80 paid rows and 64 target queries: 16010 tokens.

The longest command in each fixture was nine tokens including EOS. Each profile allowed 32 generation tokens. The configurations use synthetic zero measurements and dense topological graphs. Their 129 responses are charged artificial fixture responses, not scientific training or evaluation data. These measurements are examples, not a universal upper bound for arbitrary textual names, descriptions or numerical values. No conclusion about intervention quality follows.

The old 512-token smoke cannot support the new task prompt. Even a 4096-token cap would reject the five-node, 32-row fixture. We should preserve the full shared information and measure the intended training/inference context before freezing production resources. This profile does not establish GPU memory, throughput or safe batch size. The current loss primitive produces logits at all sequence positions, so longer histories may make its vocabulary projection expensive. Any action-position scoring optimization needs its own equivalence and gradient checks before use.

The profiling command used 1.767928 child CPU seconds, 2.405888 elapsed seconds and 492371968 peak child RSS bytes. Source, tokenizer pins, all five public states, tokens, artificial charge journals, script and logs are preserved under `results/ace_llm_intervention_language_policy_preparation_20261102/token_profile01`. The five tokenizer files are retained in exclusive external custody rather than duplicated in Git.

## Next qualification

The next bounded stage must authenticate the target tokenizer and runtime, reload a pinned artificial adapter, check state changes across multiple prompts, retain exact command tokens and exercise the charged process boundary. New long-context memory and timing measurements must precede a production GPU request. CPU preparation comes first. There is no reason to spend another GPU allocation on the already accepted 512-token fixture.

Fresh scientific policy-training, development and confirmation worlds remain ungenerated. Charged branch supervision, matched labels, final-iterate model selection, all policy failures, experimental cost and private evaluation need a separately frozen campaign. The question of lower experimental cost remains open.

## Review correction before target execution

Independent review found one required source defect. A fresh `GenerationConfig` alone can still inherit stored model defaults in the target Transformers implementation. The call now explicitly sets `use_model_defaults=False`. A new artificial regression exercises the actual local configuration-preparation helper with conflicting sampling, beam, forced-token, suppression and repetition settings. It demonstrates the inherited behavior as a control, then verifies that the corrected call retains the declared settings. The target implementation was also inspected at its [versioned upstream source](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/src/transformers/generation/utils.py). Actual target-runtime execution remains open.

The optional label-hardening suggestion was also implemented. Training completions must match canonical legal command text, in addition to matching encoded tokens. A normalizing tokenizer cannot silently repair a noncanonical training label. This is checked by one new negative fixture.

Those two new methods and two affected generation methods passed. The original adapter tests were not repeated. This correction command used 1.312065 child CPU seconds, 1.340641458 elapsed seconds and 346849280 peak child RSS bytes. There are eight distinct language-policy fixture methods across the two preserved commands. The token profile retains its original source and pins. The correction does not alter the prompt builder or its measured token lengths, and the profile has not been relabeled as execution of later source bytes.
