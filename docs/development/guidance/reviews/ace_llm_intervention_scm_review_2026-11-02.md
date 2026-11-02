# Linear-Gaussian SCM reference review

A distinct read-only reviewer inspected the new `llm_intervention_scm.py` and artificial fixtures. This review concerns the isolated reference learner, not scientific performance or a trained LLM.

One required finding: a known hard-clamped parent was incorrectly treated as missing when it was absent from the measurement list. The correction validates the clamp and merges its known value with measured parent values before building eligible regression rows. The clamped mechanism itself remains excluded. A contradictory measured clamp is rejected before any update.

The reviewer also requested stronger optional independent checks. A nonorthogonal batch posterior oracle now checks mean and full covariance for both nonroot heads. A controlled-RNG oracle verifies sampled covariance, noise standard deviation and predicted-parent flow. Eight artificial methods pass after the change. The narrow recheck confirms the required finding is resolved and identifies zero new material findings.

Final source SHA256: `296aadb60347da5c948028b7de10440e361f54a4427f2a0655e39961ca7e2a7b`. Test SHA256: `d02e8b9cd4160d550ff43743a41c5ce65d8687edd725455567d5624d1b19b930`. Both preparation attempts and logs are retained under `results/ace_llm_intervention_scm_preparation_20261102`. Final test CPU/elapsed/RSS is 0.088943 seconds / 0.132739791 seconds / 40,583,168 bytes.

Exactness is scoped to retained local linear-Gaussian regression factors with known independent noise and error-free covariates. Missing-parent rows are dropped, not marginalized. This is not qualification of a nonlinear learner, private-risk certification, model training or acquisition efficiency.
