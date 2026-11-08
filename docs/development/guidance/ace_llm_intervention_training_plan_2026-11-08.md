# Complete collection to a training plan

The new `llm_intervention_training_plan.py` connects the existing collection,
exposure and training-engine interfaces. It prepares a whole batch on the CPU
before a GPU training allocation. It does not collect responses or load weights.

The caller supplies independently pinned terminal, inventory and collection-plan
bytes, the ordered complete record roster, and every raw record and supervision
receipt. The bridge checks the original planned membership, complete world and
response accounting, row identities, pair rosters and byte bindings before the
first tokenizer call. A failed or incomplete batch cannot become a smaller
training set. No strict pairs means rejection, not a replacement training rule.

The existing exposure rule gives every record one supervised update and selects
at most one strict preference pair per record by its fixed identity hash. Ties
and unused pairs remain in the output. The numerical policy can consume the same
exposure object. Each prepared record uses the existing complete prompt and legal
command tokenization. The engine receives the explicit GPU09 arithmetic recipe,
112 adapter matrices with 1,146,880 parameters, and the caller's fixed checkpoint
indices. No training occurs in this bridge.

The Qwen adapter dimensions are explicit: 28 layers, rank 8, hidden width 1,024,
query output 2,048 and value output 1,024. Actual model admission still has to verify
this architecture. Source/runtime authentication, tokenizer byte authentication,
collection execution acceptance and scientific train/development separation remain
outer-runner duties. Matching receipt bytes alone cannot establish those facts.
The output therefore keeps execution and scientific qualification false.

Five new checks pass using the preserved two-world artificial collection and a
small artificial tokenizer. All four supervised examples, two selected preference
pairs, two unused strict pairs and two ties are retained. No old test suite was
rerun, and no new environment response, model fit or CUDA work occurred. The initial
command used 0.640512 child CPU seconds, 0.905194167 elapsed seconds and
199,294,976 peak RSS bytes.

Independent review found two missing consistency checks. Both are now implemented.
Receipt task and envelope digests must agree with the record before tokenization.
Every world must declare complete, error-free accounting with matching reserved,
returned and accepted counts. Two new focused methods cover five inconsistent
inputs with refreshed byte pins and one unchanged positive example. They passed
in 0.551701 child CPU seconds, 0.558357125 elapsed seconds and 199,426,048 peak RSS
bytes. The original five methods were not repeated. The initial source, tests
and receipt are preserved separately from the correction.

These are preparation costs only. Actual Qwen tokenization and integrated GPU
training remain to be qualified in their respective CPU and GPU stages.
