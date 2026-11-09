# Predicate mapping to the pinned original source

Original: `lineage/original-llm_intervention_campaign.py`, SHA256 `9a2717dd68c37ddde6d7fb63c6cc519c067f7b1d457d4f49f919c18342424bbe`. This is static source reasoning; the original source was not imported or executed.

| Original location | New treatment |
|---|---|
| `_load_attempt` 295–298: exact independently pinned attempt bytes, decode, exact field set | `_historical_attempt`: same hash/field/decode checks on the accepted captured/read-once bytes. Dispatch separately requires the caller's CPU pin to equal acceptance, so the live local-barrier self-hash cannot become authority. |
| 299–302: version/campaign/policy/phase/source-plan identity; false retry and private-score flags | Same conditions, with campaign itself additionally pinned to the captured bundle and exact eight-policy/24-world binding. |
| 303: candidate bytes equal header; full `_inventory(path)` equals declared evidence | Candidate byte equality retained. Declared evidence must equal the independently accepted complete map. Full payload inventory reads are deferred to explicit outer CPU pre/post gates. No module-global inventory patch. |
| 304–309: complete arm list, exact arm fields, integer world identity/status and reason | Same conditions; reason-name check uses the original plain-string length range 1–256. |
| 310–318: absent-arm constraints or exact world path/receipt pin, captured arm hash, plan/policy identity | Same branches and predicates. Wholly-successful admission subsequently excludes failed/unstarted arms, exactly as the original outer controller does. |
| 319–330 and `_closure` 164–174: service closure and service-free CPU branch | The fixed five CPU policies must not be in services; their service record must be `not_started`/null/null, and the live service path must not exist. No persistent CPU branch is admitted. GPU policies delegate to the untouched original function, retaining every branch. |
| Original entry 538–543: closed is None, no errors, all arms completed | Retained at capture and each CPU callback. |
| `_root` 132–137: campaign/source-plan bytes equal authenticated values | Original remains unchanged; adapter additionally checks the exact pinned campaign/source-plan/start bytes through safe live reads. |
| Original local barrier 204–209: live attempt self-hash | Original remains unchanged, but the CPU callback rejects any value different from the independent accepted pin. |
| Original `_load_attempt` used by GPU phase/service validation | Only the exact five CPU IDs receive the metadata path. The three known GPU IDs use the saved original callable and arguments; all unknown IDs, paths or campaign values fail. |

No source-plan, model, service command, seed, menu, charging, policy ordering, numeric algorithm, new GPU evidence validator, resource or deadline change is made. The callback is a scoped operational admission change with explicit external historical post-custody obligations. It must not be installed around the finalizer or treated as execution acceptance.
