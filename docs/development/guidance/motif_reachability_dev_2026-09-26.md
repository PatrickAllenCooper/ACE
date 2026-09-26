# B3 disjoint-motif SCM development screen

Run on 26 September 2026, using no model API or CURC allocation. `scripts/research/motif_reachability.py` creates a known DAG with k independent three-node interaction motifs and N−3k independent observed roots. Each batch observes every node; an intervention sets one or two roots of one motif. The learner maintains one exact Gaussian posterior per child. The objective is mean motif prediction error, with posterior risk scored numerically. This is a deliberately limited reachability prototype: padding nodes have no effect on the objective, motifs do not connect or propagate, and the graph family is favorable to analytic design.

Three development seeds (100–102), N∈{15,30,100}, feasible k∈{1,3,10}, background SD∈{0,.15}, λ=4, and 400 cost units yield 48 validated cells. Each cell records five arms: random/coverage singles and random/coverage/posterior-risk pairs. Cost is `samples × (1 + λ × actuator count)` and is checked independently from the hashed receipt. The coverage control cycles across motifs before repeating action values. An initial menu-ordering error that overfocused on the first motif was corrected before recording the results below; the 48 archived cells were regenerated and revalidated after that fix.

Mean motif MSE (three development systems; selected settings):

| N | k | Background SD | Random single | Coverage single | Random pair | Coverage pair | Risk pair |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 1 | 0 | 1.3846 | 1.3843 | .0005 | .0003 | .0004 |
| 15 | 3 | 0 | 1.6407 | 1.5643 | .8385 | .4086 | .5250 |
| 30 | 10 | 0 | 1.7228 | 1.6280 | 1.3611 | 1.3920 | 1.3929 |
| 15 | 3 | .15 | .0408 | .0282 | .0874 | .1400 | .0368 |
| 30 | 10 | .15 | .2696 | .0885 | .7781 | .5644 | .4687 |

At fixed k, N=15/30/100 gives exactly the same output by construction because extra nodes are independent roots and never affect the motif learner. This is a **negative control for padding**, not evidence of graph-scale invariance. At fixed budget, increasing k leaves fewer useful paired probes per motif; pair capability alone no longer solves all interactions. With weak but nonzero background excitation and λ=4, coverage singles can outperform paired policies at k=10 because single interventions cost less and every batch passively observes every motif. Risk pairs do not consistently beat simple coverage pairs. This prototype therefore does not satisfy B's promotion criterion.

The next justified implementation is a connected sparse graph with downstream effects, bounded indegree, a fixed number of hard motifs embedded at different graph sizes, and a distinct sweep increasing motif count. It must include a graph-matched single-action controller and a cost-aware joint-coverage control. Freeze that simulator and its evaluation distribution before a fresh CURC confirmation. Do not repeat this disjoint family at larger seed count simply to increase significance.
