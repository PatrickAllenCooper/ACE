Local preliminary matched-student ladder, 17 Sept 2026 (MacBook, 3 seeds x 40 episodes).
5-node GroundTruthSCM; every arm uses ACE's (64,64)/100-epoch student
(`--student_arch ace`, the runner default from commit 489acb0); PEV/random_ens
use a K=5 bootstrap ensemble. End-of-campaign non-root broad-range MSE per node:
ACE 0.165, round_robin 0.161, random 0.223, random_ens 0.172, PEV 0.045,
PEV-var 0.039. Superseded by results/pev_ladder/ (CURC, 10 seeds x 171 ep) when
that lands; kept as the evidence behind the 17 Sept decisions.
