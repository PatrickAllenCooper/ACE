# Prospective relative replay and custody review

Independent read-only reviewer Cicero (`01a117fa-b27c-74c0-ad80-ebb8feb08a43`).
This distinct implementation review covers the new supplemental relative replay,
its interface to the private planner, and the complete custody reconciler. No
actual B outcomes, checkpoint inference, fits, Slurm mutations or installs.

## Findings and disposition

1. **P1: semantic verification decoded outcomes before the fit barrier.** The
   existing package verifier decodes scientific JSON for identifier/binding checks.
   Fixed a byte/path/hash-only preflight before all640 fits and40 journals; defer
   semantic verification until the full barrier passes. A no-decoder sentinel
   test verifies the new byte preflight does not decode scientific JSON.
2. **P1: API could omit the independent manifest pin.** Fixed lowercase64hex
   SHA256 validation before any manifest access, including programmatic callers.
3. **P2: original acceptance/score hashes were not linked to derived outcomes.**
   Fixed projection metadata/digest syntax and explicit original-to-derived
   score/acceptance links. Newly authored metadata has its own hash; it is not
   represented as unmodified original bytes. The private planner preserves
   original receipts and records the explicit projection derivation.
4. **P2: distribution metadata did not authenticate imported code.** Fixed
   actual module version/origin checks and an authenticated snapshot loader for
   the entire archived learner closure. Helper source, model and prediction
   bytes also execute/deserialize captured authenticated snapshots. These checks
   are not a cryptographic certificate for installed dependency binaries.
5. **P2: complete custody did not bind fit bytes to the original seal.** Fixed
   exact640-cell seal membership and all original sealed receipt/model digests
   in the reconciler's required raw binding map. A missing-seal-cell test rejects.

The follow-up static review found no additional adapter/planner defect; the
remaining custody finding was then fixed. Eleven focused adapter tests and four
custody tests pass locally. They use constructed metadata/analytical numbers,
not selected-world outcomes or target-runtime inference. Eleven planner rejection/snapshot checks also pass; the26new checks pass together
from an unrelated cwd. Full positive assembly and target-runtime qualification
remain separate gates; none of these checks establishes
public release readiness, historical freeze authentication or full anonymity.

An actual partial-custody probe rejected at missing `complete.json` before
inventory/source/outcome access or output. The original frozen CURC chain,
accepted A/C/confirmation artifacts and manuscript remain unchanged.
