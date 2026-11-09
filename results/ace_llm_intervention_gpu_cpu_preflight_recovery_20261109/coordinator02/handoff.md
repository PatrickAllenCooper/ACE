Coordinator02: narrow R1 correction ready for independent recheck.

Only `authoring/pull_terminal.py` current-job resource predicate changed: `normalized[1] == ace_llm_scientific_campaign_gpu02`. The GPU02 builder name now matches. Every other collector byte and all five other coordinator files are unchanged. Original GPU01 historical validation, allocation/resource/no-requeue gates and all API/CLI/root/custody/accounting behavior remain unchanged.

Collector SHA256: `3697547ddace45fda812c069c08d2d299e75b61fe9db717807a5379645771c27`.
New test SHA256: `821d09e2af2480749ad26922887c9d63ff91958766f1ecac34f93969f17834ea`.
Original review SHA256: `548243db747623de5f7ff644bc994c78e02ee54c03000938913b10bb935825b9`.

One NEW method, two artificial subcases, passed once. Exact requested4CPU15360M, allocated4CPU15G/oneA1003g20GB,48min, account/partition/QoS, pinned saved allocation with Requeue0/Restarts0. GPU02 accepted; predecessor GPU01 rejected. Saved observation/allocation/results remain in fixtures/. No actual allocation or scientific inputs used. Cost:0.116803 child CPU seconds /0.129819500 elapsed; maxRSS22,347,776 bytes. Original eight methods not repeated.

Complete coordinator01 (including old source/drafts/transcripts/copy receipt) is preserved byte-for-byte in predecessor01/. Original required review/findings/receipts preserved in review-predecessor/. Existing originals were rechecked unchanged. R1.diff is exactly the one-line delta; source-pins.json records all current sources. Use coordinator02/authoring as the successor source directory. Original interface.md remains applicable at predecessor01/interface.md. No builder/launcher/core packet changes, actual build, remote actions, locks, models/science or commit.
