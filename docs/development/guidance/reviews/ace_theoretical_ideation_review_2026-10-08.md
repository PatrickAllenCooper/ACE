# ACE theoretical ideation review — October 8, 2026

## Scope

Patrick requested deep theoretical explanations and potentially improved methods.
One distinct read-only mathematical/scientific review was assigned to Volta
(`01a11cbb-16ec-7072-a104-37b16455f398`). It covered the new
[technical note](/Users/pat/code/ACE/docs/development/guidance/ace_theoretical_ideation_2026-10-08.md)
and [exact-arithmetic illustrations](/Users/pat/code/ACE/scripts/research/check_ace_ideation_algebra.py).
The reviewer executed no code, inspected no checkpoints and made no edits.
This is not a repetition of previous component reviews or the final integrated
paper review.

## Findings and dispositions

1. **Observation model and prospective expectation.** The covariance update
   needed explicit Gaussian noise and known-design assumptions. An action can
   induce random measured parent features; a prospective score must average over
   the predictive design/response distribution. **Fixed:** the note states the
   scalar and joint Gaussian observation models, distinguishes conditional
   updates from prospective utility, and requires an additional likelihood when
   observed features themselves convey parameter information. Mean-feature
   substitution remains a surrogate, even for linear mechanisms.
2. **Baseline coverage in the validation bound.** A no-harm replacement claim
   requires simultaneous concentration for the baseline as well as candidates.
   **Fixed:** the M fixed, validation-independent predictors explicitly include
   the online baseline. Bounded loss, iid validation and paid measurement scope
   remain explicit; there is no retrospective guarantee for the exposed grid.
3. **Objective versus architecture attribution.** An objective cannot remain
   fixed while comparing mechanism-only and terminal/stability losses.
   **Fixed:** the proposed comparison has explicit objective ablations with
   matched observations, initialization and fitting resources. Architecture
   attribution is reserved for a separate comparison holding the objective fixed.

The focused correction recheck returned **zero remaining required issues**.
It also confirmed the archived 95/640 same-target steps and 18 exact target/value
matches, which prevent interpreting rank-one score geometry as evidence of
identical historical policies. No algebra-source corrections were required.

Main subsequently added a clearly untested constrained-fitting alternative and
made spacing/readability edits. The constrained proposal restricts observed
mechanism losses and makes no identification or unseen-risk guarantee; it is
not a separately qualified implementation or reviewed mathematical theorem.

## Verification and limits

Six new groups of fabricated rational-arithmetic illustrations passed once.
Their source was verified unchanged after the review. These illustrate Gaussian
covariance updates, target ranking reversal, rank-one IVR scores, signed DAG
propagation/covariance, quadratic descent and design geometry. They do not
establish empirical premises or prove general theorems by testing examples.

The note links empirical statements to accepted A/C/F or historical acquisition
evidence and retains failed gates, unfavorable controls and the worsening
history. It uses no unqualified prospective B performance. Methods are proposals
and explanations are hypotheses. Primary literature establishes related work
and assumption limits; no novel optimal-design or inherited DAgger guarantee is
claimed.

The interactive companion was checked against the installed Canvas API types;
an invalid text-size literal was corrected. No standalone TypeScript compiler
was available in the inspected existing runtimes. No installation was performed,
and no rendered-page or complete TypeScript qualification is asserted.

No accepted fits, frozen workers, active launch inputs, manuscript, scientific
receipts or experiments changed. New experiments require a separate frozen
protocol and resource scope. Final paper/replay/reporting/release gates remain.

Evidence: [ideation preparation receipt](/Users/pat/code/ACE/results/ace_theoretical_ideation_20261008/preparation.json).
