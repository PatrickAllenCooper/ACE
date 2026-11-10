Preserved independent-review requests and author disposition

Leibniz reviewed initial e2f50 source and identified exactly two required findings: R1 normal SIGTERM could bypass finally/owned local process-group cleanup; R2 failure.json asserted partial_preserved=true before preservation had occurred. Source2b851 added termination/cleanup guards, explicit bounded ESRCH confirmation plus stream closure, and detached phase-aware preservation status/pins. Two NEW methods passed once; original five methods were not rerun.

Focused R1 recheck required SIGINT to be deferred during cleanup as well. Sourcea699 changes only cleanup_signals from2b851. One NEW method passed once, injecting a second Ctrl-C during owned-group confirmation. All prior seven methods were not rerun. Independent final review is supplied separately by Leibniz; this author note is not that review.

Initial complete draft26eb and tested/reviewed e2f50 remain under draft01/predecessor02; intermediate2b851 remains under predecessor03. Every test attempt retains exact tested bytes, stdout/stderr, fixtures and costs. All three test invocations passed; expected injected failure fixtures remain intact. No actual continuation was run.

Read-only author provenance inspection found that b476 did not contain transport-recovery-design-review01/review.md (git show returned128; stdout empty, SHAe3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855). That early inspection did not retain git stderr separately; its result was reported in the tool transcript. The driver therefore binds that existing independently pinned design review to its NEW execution-source/review commit, while all five original failure/probe artifacts were checked against b476. No old commit is relabeled and no future commit was invented.
