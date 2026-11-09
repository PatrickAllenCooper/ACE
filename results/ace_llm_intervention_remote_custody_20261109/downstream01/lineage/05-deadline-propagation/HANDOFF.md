# M1 collector deadline propagation — narrow correction

Consolidated latest source is implementation/. It includes independently closed D1/D2/D3 and the previously reviewed metadata1200/1230/progress delta. All prior sources, tests, failures and reviews remain unchanged at their original paths; predecessor/ captures the exact immediately prior source. The initial deadline review7f6e4502 is retained in initial-deadline-review/.

Sole production delta: both collectors' captured REMOTE_OBSERVATION add `except TimeoutError: raise` immediately before the ordinary per-UPSTREAM_PINS `(OSError, ValueError)` handler. Thus the absolute deadline reaches the unchanged outer observation_error publication instead of consuming the one-shot alarm and continuing to later inputs. All ordinary file-error collection remains unchanged. The parent observer already rejects observation_error. No additional timer reset, resource cap, scientific gate, retry path, source/input closure or job/deadline change.

One NEW fabricated nested-handler method passed once for both collectors: injected original deadline stops before the next input and retains outer TimeoutError; ordinary OSError/ValueError controls still retain per-input errors and examine the next input; success control preserves both pins. Only captured handler/loop AST and toy functions execute; no real signal, complete collector, project/model/remote/scientific run.0.041182childCPU/0.048029209elapsed/19,906,560RSSbytes. Source snapshots/stdout/stderr/argv/resource receipt in attempt01. No old method repeated or new failed test.

Exact production delta is additionally rendered as implementation/{gpu,finalizer}/remote-observation.diff, each one line. Every other production byte is unchanged from the preceding handoff.

## Exact new pins

- `finalizer/pull_terminal.py`: `1440226e5cb135232eef4eb65a037eb6bd711d3a04e0256a92a3a9a3a0beedc4`
- `gpu/pull_terminal.py`: `7936f3a033ea512be2bc429f962bec149fe14d95803f33af2c6d7012c106dbc3`
- `test_deadline_propagation.py`: `eceb83dde4b7b7df76242c0d89d279bc6f390d6c3bf50fc1fa75ab6b5a07901b`

Complete final source index and test receipt pin are in handoff.json. Narrow M1 recheck requested; no operational/scientific acceptance claimed.
