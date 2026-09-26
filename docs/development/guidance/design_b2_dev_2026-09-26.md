# Matched-menu joint-intervention development screen

Run on 26 September 2026. This is a two-parent numerical interaction mechanism, not a whole SCM or foundation-model result. Its purpose is to separate the availability of joint actions from the policy used to select them. Twelve development seeds (100–111) vary the interaction sign and size. Background standard deviations are 0, 0.15, and 0.5; actuator penalties λ are 0, 1, and 4. Five policies share a budget of 400 cost units and batches of eight samples: random or posterior-risk single-target selection, and random, cyclic quadrant coverage, or posterior-risk paired selection. Pair policies have the same four-action menu. The charged cost is samples × (1 + λ × number of actuators); raw samples and actuator uses are logged separately. All 108 cell receipts and metrics are in `results/local_design_b2_dev_20260926/`.

Mean final weighted predictive MSE on the 12 systems:

| Background SD | λ | Random single | Risk single | Random pair | Coverage pair | Risk pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 1.22249 | 1.22246 | .00004 | .00006 | .00006 |
| 0 | 4 | 1.22274 | 1.22270 | .04746 | .00071 | .00052 |
| .15 | 0 | .00069 | .00024 | .00004 | .00006 | .00006 |
| .15 | 4 | .00247 | .00438 | .04746 | .00071 | .00052 |
| .5 | 4 | .00052 | .00061 | .04746 | .00071 | .00052 |

At λ=4, pair policies make five eight-sample decisions (40 raw samples, 360 cost units); single policies make ten (80 raw samples, 400 units). Pair-arm results are independent of background SD because both parents are actively fixed. When background SD=0, single-parent actions cannot identify the interaction coefficient; the paired-action gain is an action-space result. At positive background variation, single actions can identify it, and high paired-actuator cost can make single policies competitive. Within the pair menu, simple coverage is close to posterior-risk design: at λ=4 the latter wins only 7/12 seeds. The large random-pair mean at λ=4 reflects repeated quadrants under just five decisions, not evidence for a uniquely advanced selector.

The design policy samples candidate contexts from the specified background distribution but never queries the environment or uses hidden truth while scoring. Each arm has its own seeded observation noise, so these paired-system comparisons also contain simulation noise; do not turn small coverage-versus-risk differences into a significance claim. No new CURC sweep is justified on this same tiny generator. Next B gate: embed interaction motifs in larger SCMs and vary motif count separately from node count, with fixed acquisition budgets and the same action-menu controls. Closed-model menu compilation remains disabled.
