# Tokenization startup recovery

November 8, 2026. This is the proposed operational correction for failed tokenization job35415334. Independent failure acceptance and review of the exact new packet must precede another submission.

## Observed failure

TOKEN01 failed1:0 after129 allocated seconds on3CPUs, charging387 CPU seconds and no GPU time. Complete terminal custody retains31,039 raw files and218,721,958 bytes in two independent local copies. Both remote inventories and all87 original packet inputs match. The disposition is `2f55409dce3d44e7ee2e5eed6a359ffa1b46cdec03a7672e23eaca73f52fd679` and the custody inventory is `b02ecbbd83a3d9f48a3e7248fbb2512be66ffadac723a05bb2fb038465035cc9`.

The recorded exception is `ValueError: original absolute deadlines required`. The traceback reaches the owned-child supervisor's deadline check before its worker-spawn call. The parent had completed runtime staging and reached the launch boundary after the original120-second startup deadline. This is a recorded deadline-precondition failure, not a recorded signal interruption. Diagnosis `d4cb918b048b9fdc0dc843dc303ca4bbf436094b697fc8833e9b83114afbfb63` binds the exact source and traceback. It does not identify which filesystem or startup activity caused the delay.

No completed token preparation is claimed. The failure receipt reports zero environment responses, training updates and inference calls. Independent acceptance must establish the stronger no-worker and no-tokenizer boundary from source control flow, traceback and complete custody, rather than relying only on missing output. Removal of the staged runtime tree is not established.

## Narrow correction

Retain scientific freeze `15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d`, source commit `a9346022f145cc29439325b6f291597b2e6ec037` and token preparation entry `47bbf574269b9187f1237e4a9731115228b06bbb5ab65937df3e0f14bd37b8f3`. The existing source accepts explicitly supplied ordered deadlines. No scientific source change is proposed.

Change tokenization startup from120 to300 seconds and use the new exclusive output `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_02`. Its tokenizer input directory moves with that output. Work550, cleanup570 and hard590 remain measured from the original monotonic start. The request remains3CPUs,11520MiB,600seconds,zeroGPUs,accountucb736_asc2,partitionacpu,qoscpu-normal and no requeue. A full startup allowance leaves250 seconds before the unchanged work deadline. Completion within that remaining time is not guaranteed.

Keep all160 collected records,3584 preference pairs,32 training systems, fixed tokenizer files, numerical and LLM exposure rules, model revision, seeds and both existing namespace claims unchanged. Do not regenerate collection, inspect development outcomes, change labels or replace worlds. A new operational amendment binds the failed attempt and accepted collection without relabeling the original scientific freeze.

## Admission and completion

The terminal accounting reconstruction gives13,373 allocated CPU seconds and1,601 GPU-slice seconds across43 jobs and88 journals. Independent failure accounting must confirm these values. Including the unchanged3,600CPU preparation allowance and one1,800CPU-second retry reservation would admit18,773CPU seconds and1,601GPU-slice seconds. Full remaining caps project55,493CPU seconds and7,181GPU-slice seconds against the57,600/7,200 ceilings. These are reservation projections, not process CPU or whole-cycle costs. No other-stage cap may increase without a new reconciliation.

Before submission, require independent failure acceptance, an explicit no-learned-work proof, the exact43-job/88-journal baseline, separate operational amendment review and concrete packet review. Preserve complete failed TOKEN01 and collection01 custody alongside accepted collection02. Reconcile both shared controller locks, all jobs and submission journals, namespace claims, remote input bytes and current site readiness. Submit once into the new output. An ambiguous submission must be reconciled, never blindly retried.

Successful tokenization still requires the original source/runtime checks, exact exposure and token-length validation, complete terminal custody, scheduler resources and cleanup evidence, and independent result acceptance. It would extend the accounting chain to44 jobs and90 journals. GPU training remains gated on that acceptance. This operational recovery provides no evidence yet about LLM policy quality or intervention savings.
