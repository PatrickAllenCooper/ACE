# Integrated scientific contract: science, training and endpoints

2026-11-08. **Required corrections: 2.** Static consistency review of `ace_llm_intervention_scientific_contract_2026-11-08.md`, SHA256 `5aa70064b5adb35efc9ba34bc02f7cdfd98b8bf09a8468720a7b2103f0d30094`. This checks the new integrated prose against the unchanged original proposal and implemented scientific choices. It does not reopen component correctness, tests, actual qualification, or Godel's resource/runtime review. The document remains a draft; no combined freeze or execution approval is issued.

## Required corrections

**1. P1 — describe strict-set pair selection, not selection followed by a strictness test (line 23).** The draft says an unordered pair is selected at each nonterminal record and used only if its label is strict. `llm_intervention_training_exposure.py:99–100` first filters to strict comparisons and then selects the smallest `(selection_sha256, unordered_commands)` tuple. A record with a tied minimum-hash pair and another strict pair therefore receives a DPO update under the implementation, but would receive none under the draft's stated rule. This changes D and the admitted schedule. The existing exposure proposal explicitly describes the implemented rule and acknowledges that strict/tied eligibility depends on supervision.

Replace that passage with: “For each record with at least one strict comparison, select the strict unordered pair with the smallest predefined hash, breaking hash ties lexically on its sorted commands. A record with no strict comparison receives no DPO update. Execute these selected pairs in original roster order after all 160 SFT updates. D is the number of records containing at least one strict comparison, at most 128; D=0 terminates admission without recollection or a replacement rule.” Preserve every tied and unselected comparison. Do not change the already reviewed implementation to match the erroneous prose.

**2. P2 — name LoRA alpha and its actual multiplier correctly (line 25).** “Rank8 … with scale16” conflates alpha with the forward-pass scale. The fixed recipe and scientific worker install `rank=8, alpha=16`; `llm_intervention_learning.py:116` sets `self.scale = alpha / rank`, and the forward pass multiplies by that value. The engine also checks the effective scale is 2. State “rank 8, alpha 16 (effective adapter multiplier 2), on query/value projections.” This preserves the actual recipe and removes a factor-eight ambiguity before adoption.

## Other scoped consistency

The original proposal artifact at `results/ace_llm_intervention_screen_protocol_preparation_20261107/test-attempt01/fixtures/prospective-proposal.json` still has byte SHA256 `8e3214958f671dfee5ef0a2d7adc70e8f5cdc26edf0fbeff02e7925921ff13da` and canonical semantic SHA256 `585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8`. These identify the JSON proposal, rather than its explanatory Markdown. The integrated document correctly keeps the proposal's twelve source pins separate from the later executable closure and leaves confirmation unexecuted.

The 32 training/24 development identities, four regimes, 12/10/10 versus 8/8/8 graph counts, independent draws across cost strata, within-world pairing, public/private separation, eight charged initial responses and 24 subsequent acquisition units agree with the proposal. Eight acquisition actions and public stop are distinct; there are eight equally weighted target queries, public scale 2 and threshold .04. The 160-record/3,584-pair roster, four branch replicates, retained negative/tied labels, 9,216-attempt and 10,240-cost ceilings and prohibition on outcome-driven replacements remain consistent.

Apart from the two corrections above, the recipe statements agree with source: singleton unpadded scoring, seed 109, bf16 base/fp32 adapters, AdamW .0001/beta .1 and stated optimizer parameters, continuous optimizer, fixed post-SFT reference, complete tokens/EOS, and zero-based checkpoint indices 159 and 159+D. The numerical ranker uses the same records and selected pair schedule with its own fixed 128-hidden-unit Adam recipe. Matched exposure is appropriately distinguished from matching pretraining, architecture or compute. The 4,744-token bound is enforced by the scientific GPU supervisor's admission check; token preparation retains full IDs under the underlying 8,192/32 limit. This statement is a gate, not evidence of future scientific fit.

The three evaluation roles are pristine base, fixed final SFT and fixed final DPO checkpoints; their parameters do not change across episodes. The 120 CPU plus 72 GPU cells and 28 world-paired contrasts match the implemented campaign/report envelope. No development-error checkpoint selection or autonomous private-attainment stopping is authorized.

Endpoint prose agrees with `llm_intervention_cost_curves.py` and `llm_intervention_screen_reporting.py`: normalized squared error is computed from posterior predictions against private structural means on all fixed queries; the curve begins after the paid initialization at acquisition cost zero. A successful cell's capped acquisition endpoint is its first crossing, or 24 if it never crosses. Failed/evaluation-failed/unresolved cells receive cap 24 even if a retained prefix crossed; actual paid cost and retained prefix attainment remain separately visible. Final attainment and first crossing are distinct. The draft's requirement to report final error, attainment, failures and all curves alongside retrospective cost comparisons avoids treating early stopping without accuracy as a saving. All 28 contrasts remain descriptive, with no complete-case filtering or confirmatory claims. Private finalization replays saved response receipts to update the SCM learner for evaluation; it does not optimize a policy or draw new responses.

## Source bindings and limits

Relevant source SHA256 values read for this interface/prose check:

| Source | SHA256 |
| --- | --- |
| `llm_intervention_training_exposure.py` | `98a03ff0438bddf1267c0689b92f8692b00b7901b966cc2e16201320ede62f23` |
| `llm_intervention_learning.py` | `065f9f68c55518190a82c6a69dc7be45a83e4ab03664e6f5df078ce6112f7aca` |
| `llm_intervention_training_engine.py` | `1e25ceb9b2de9b84666d99cdb6c02087000b9c2bfe6ea493d13dcff1a8fe5a49` |
| `llm_intervention_scientific_training_gpu_worker.py` | `ab2c999a35b247e03df87dce9f19d771c50f84747381bfbc0ed980e9c832a878` |
| `llm_intervention_ranker_training.py` | `0909428c47288ebc201abd988046d9be2283f70344d3764dcc9f6bfb845c0587` |
| `prepare_llm_intervention_scientific_training_cpu.py` | `47bbf574269b9187f1237e4a9731115228b06bbb5ab65937df3e0f14bd37b8f3` |
| `supervise_llm_intervention_scientific_training_gpu.py` | `55be5ba74fa16f3e05d5632a1d0ff4ac1b0a3187507849e82d6594810f3545fd` |
| `prepare_llm_intervention_scientific_campaign_cpu.py` | `8938d0d3a450f75e8f333989b70c89206de88ebcec928f7fe8d6a0d68c6166df` |
| `llm_intervention_cost_curves.py` | `f9420b77823f9becc7a3885cd258283ceda406655b4ba077ecbefb19fd0896e8` |
| `llm_intervention_screen_reporting.py` | `4c324e9d11ce8fc77f796e22a1c4b81234219bf4d108d9922c2497e484b63c8c` |
| `llm_intervention_campaign.py` | `9a2717dd68c37ddde6d7fb63c6cc519c067f7b1d457d4f49f919c18342424bbe` |

No tests, project imports, runtime/model/tokenizer execution, scientific draws, saved-response replay or remote operations were performed. No author source/contract was edited; nothing was staged or committed. Resource sizing, scheduler/runtime evidence and GPU-outer corrections remain with their separate reviewers. Optional suggestions: none. Required disposition remains two prose corrections pending parent changes and a focused byte-pinned recheck.

## Concurrent parent correction

Completion recheck found draft SHA256 `7c55845a71517c3fdbf6b9e6bad3d414abf10000b104b08ce371d26395f54cdf`. Line 23 now explicitly selects the minimum-hash pair among strict comparisons, includes the lexical tiebreak, retains ties/unselected comparisons, and gives no update when no strict comparison exists. **Finding 1 is resolved.** The same draft still says “scale16” at line 25: **one required correction remains, finding 2**. Concurrent resource/cleanup prose is outside this review's scope and remains with Godel. The original findings and original document pin above are preserved.

## Focused alpha correction — final disposition

Reviewed contract SHA256 `f10373f9dda3cd6bd0f68afb9964119e29877289619be457d4a4e3cde1bb1ad3`. The adapter paragraph now states “alpha16, giving an effective multiplier of 2,” preserving rank 8 and query/value targets. This agrees with the fixed recipe and `alpha/rank` implementation. The corrected strict-set pair-selection paragraph remains present. **Both required findings are resolved; 0 remaining required findings in this science/training/endpoint prose scope.** This was a focused text/pin recheck only, with no component review or tests repeated. Resource/runtime review, remaining outer corrections and concrete combined freeze/source/packet bindings remain separate prerequisites.
