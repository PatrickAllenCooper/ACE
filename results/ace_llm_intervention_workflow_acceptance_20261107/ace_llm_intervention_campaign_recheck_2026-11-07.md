# Campaign R1 delta recheck — 2026-11-07

**R1 is closed; no required campaign findings remain at the reviewed pins.** This narrow recheck covers publication reconciliation and the optional accounting wording. It supplements, without modifying, the [initial review](/Users/pat/code/ACE/docs/development/guidance/reviews/ace_llm_intervention_campaign_review_2026-11-07.md), whose SHA-256 remains `ef42f3c7a0c76bf6ec98410c1079e99cf7f2e8dc71ed7ce15fa5afbee2d6aad9`. This is component integration acceptance only, not scientific or live-runtime admission.

Exact current SHA-256 pins, matching preparation02:

| File | SHA-256 |
| --- | --- |
| `scripts/research/llm_intervention_campaign.py` | `9a2717dd68c37ddde6d7fb63c6cc519c067f7b1d457d4f49f919c18342424bbe` |
| `scripts/research/test_llm_intervention_campaign.py` | `74de075e677315d89c02530047aa021b184b7576e85da83f3744aeaf84775f07` |
| `docs/development/guidance/ace_llm_intervention_campaign_2026-11-07.md` | `d5f81e402466e9d9382f4671159fd84b571216ec9890a5d9bc3a0328ef305b92` |

The saved [delta.patch](/Users/pat/code/ACE/results/ace_llm_intervention_campaign_preparation02_20261107/delta.patch) exactly matches the preserved old/new bytes. Only these three files changed in the 16-file source/guide closure; dependency pins and the original eight test methods are unchanged. Any subsequent source change needs its own delta review.

## Required finding closed

[Campaign lines 372–396](/Users/pat/code/ACE/scripts/research/llm_intervention_campaign.py:372) capture the intended receipt bytes before publication and inspect the published path on exception. Exact matching bytes retain `finalization.json` authority and produce a separate `post-publication-diagnostic.json`; confirmed absence produces `failure.json`; unreadable or different bytes produce `publication-uncertain.json` requiring outer reconciliation. The original exception propagates when diagnostic persistence succeeds. No branch restarts finalization or overwrites its exclusive intent. [Guide lines 107–114](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_campaign_2026-11-07.md:107) now explicitly define these outcomes. This resolves the contradictory success/failure disposition demonstrated by R1.

The five added [publication tests, lines 330–434](/Users/pat/code/ACE/scripts/research/test_llm_intervention_campaign.py:330) isolate this control flow with fabricated admission/reporting and actual exclusive filesystem writes/links. Saved artifacts agree with their assertions:

| Recorded case | Observed disposition |
| --- | --- |
| Normal publication | Matching candidate/receipt; no failure |
| Pre-link error | Candidate retained, no published receipt, failure recorded |
| Post-link interrupt | Matching receipt remains authoritative; separate diagnostic with matching expected/observed hashes; no failure |
| Unexpected receipt bytes | Uncertainty diagnostic; no failure or success-authority diagnostic |
| Unreadable post-link receipt | Uncertainty diagnostic records `PermissionError`; no failure |

All five retain the finalization intent; each test checks that another finalization is rejected. Saved successful receipt hashes also bind their summary bytes correctly. Diagnostic-write failure, repeated interruption or hard termination still requires the disclosed outer storage/runtime reconciliation; the correction does not claim otherwise.

## Optional findings

**O1 remains optional and unaddressed.** [Guide lines 98–105](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_campaign_2026-11-07.md:98) still do not explicitly state that unresolved cells have `charges: null` and that their partial charges may be available only in raw evidence. Cap plus full declared initialization remains a comparison endpoint, not actual spend. No accounting implementation changed. O2's broader optional failure coverage remains as recorded in the initial review; it is outside this narrow correction.

## Evidence and scope

Read-only byte/inventory checks confirmed both local and external copies of preparation02: **58 files / 404,415 bytes per copy**, manifest SHA-256 `de56b50de619f1758f33d00d0962aea64b6af8b7a47dbcca1f64497858dacecf`. The original **1,405 files / 2,563,875 bytes per copy** also remain intact against the initial independently retained manifest `9856342ff9927cdcae7fbbe1cc7f7b3916d49ae6bc5536a4ec08df58f42acbf7`.

The [receipt](/Users/pat/code/ACE/results/ace_llm_intervention_campaign_preparation02_20261107/receipt.json), SHA-256 `9c838323aadb6ffd84245336a6648a5bc0fa2b73a9e040c757fde14d1b9a9ca2`, selects only `CampaignPublicationTests`; saved stderr records **five passes in one invocation**, with no original suite replay. Recorded cost: **0.104156 child CPU seconds, 0.151518584 wall seconds, 34,635,776 bytes peak child RSS**. The recorded run has zero worlds, environment responses, policy processes, models, training updates, GPUs or remote operations. These are verified saved claims/artifacts, not re-executed tests.

This recheck performed source and saved-artifact reads plus hashing only; no tests, behavior probes, source edits, remote work, ledger changes or commits. Initial limitations remain: actual provenance/admission, CPU/GPU allocation placement and utilization, total deadlines, hard-kill/storage recovery, duplicate-submission prevention and `TaskProcessPolicy` descendant/output containment belong to the outer supervisor.
