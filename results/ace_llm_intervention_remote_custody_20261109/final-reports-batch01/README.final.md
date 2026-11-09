# Stable handoff: separate opaque finalizer report batch

Use **correction01/source/batch_reports.py** and the captured invocation/API in **correction01/README.md**. Initial authoring and both initial test attempts remain under authoring/. The three initial review findings and exact focused closure are preserved under reviews/initial and reviews/final. No original files were replaced.

Final source SHA256: 0737c80ff5f896b2b4f5bf1931d9ab8099cf8cbe518c305ccb25145ff7670b2b
Final test SHA256: 605ab5429c62a22b103ceb0174aaf0829dd65ef69d586d62b177cdd7680ba76a
Independent final review SHA256: 8e991fc65d432d7bb8994fc028afdec6ca7bc1b52928f264ca251689e84ef27c (0 required)

API: captured `bind_sources(...)`, then `fetch_reports(...)` with independent canonical original receipt/member-map byte pins, finalizer acceptance/custody pins, and new absolute report/log destinations plus the existing transport-helper path. Original backend527d and authority02a7 are unchanged. Parent remains responsible for acceptance, custody, locks/once-only gating and all actual invocation authorization; this source preparation is not actual retrieval or scientific acceptance.

The exact opaque additional selection is 192 cost reports +24 world summaries +24 world pin files +campaign summary/pins =242 files. All must exist with independently matched pins under cpu/cpu/worker/artifacts/campaign/private/. Missing reports cause an explicit failure; unresolved cells are never fabricated or silently dropped. Two full archive passes total per successful batch, not per file. No archive download, outcome JSON parsing or source/science mutation. The old local-selection receipt remains unchanged. Additional local stdout records duplicate selected bytes only and are explicitly capacity-accounted.

Nine distinct new fabricated methods passed across 6+1+2 runs, with no repeats. The final two correction methods used 0.293575250s wall /0.249391 child CPU /27,688,960 bytes child maxRSS. Original snapshots, synthetic partial streams/archives, error records, exact transcripts/resources and both reviews are retained. No actual remote/main/scientific/GPU/model/tokenizer work, repository edits, locks or builds occurred.

FINAL.json records the source/review lineage. inventory.json independently pins every preparation file except itself. Its total_payload_bytes excludes inventory.json; external handoff size includes that manifest. Keep this preparation immutable after handoff.
