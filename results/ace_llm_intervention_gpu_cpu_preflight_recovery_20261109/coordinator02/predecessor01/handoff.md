GPU02 recovery coordinator ready for independent source review. Actual entry/helper/authority/preflight inputs remain caller-pinned and no actual packet was built.

Files under authoring/: build_packet.py, launch_once.py, monitor_once.py, pull_terminal.py. Unchanged compact_inputs.py and cpu_history_contract.py retained alongside. Exact source pins:

- build_packet.py: `85691a9c7d52e40b334d9e42ea2da67762a955953c115de8a7d58be385935c5a`
- compact_inputs.py: `a0e601a7be0ec2144faf0956de003dbe19b1a0f538b22a36a3506d316b698245`
- cpu_history_contract.py: `3893a30ec2e652dac78529602564a4a295ddf9258461530ec7c74557df809025`
- launch_once.py: `7bcd8c3db86446d92e954011a2351ebd773018b92fd7879607f8c2e9a9c9fe93`
- monitor_once.py: `e9d5273f3eb000de9d73f2dc7bb3ee55e1fdc4585fce3e34123afd9bd9248903`
- pull_terminal.py: `e6a90ef9f1ed6df55e225a2c16dfe9a7279603ceb5a6ce3666b592fc6d6d542f`

See interface.md for required CLI pairs, published failure schema and root-label contract. Original fullmap128 six sources are in predecessor/; interim drafts01/02, all8 new test-method transcripts across3 attempts, fixtures, source snapshots and diffs are retained. No old tests repeated, no failed test attempts. Measured childCPU/elapsed seconds by attempt:

- attempt01: return0, 0.155962 CPU / 0.168828583 elapsed.
- attempt02: return0, 0.127931 CPU / 0.145035417 elapsed.
- attempt03: return0, 0.129121 CPU / 0.143355875 elapsed.

New tests exercise reconciliation/error retention, no-work/source/input/charge/journal joins, only old-root/gpu append, absent-output/dangling-link/sibling guards, unchanged95inputs, static unchanged original helpers, one-sbatch ordering and new48/98 accounting. Tests imported only this new coordinator's stdlib metadata code and extracted isolated guards on artificial temporary trees. No producer imports, original scientific input reads or original test suite runs.

Required external gates before use: independent coordinator review, final entry/helper reviews and exact operational Git source manifest, adopted authority plus independently reviewed static preflight/context/readset, independently reviewed concrete packet. Collector full custody is evidence only; independent postcustody review remains mandatory. Finalizer consumer changes remain separate.
