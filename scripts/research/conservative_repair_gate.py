"""Fixed-sample bounded-loss gate for independently randomized diagnostic blocks.

Inputs are paired block-average losses in [0, 1], from frozen predictors.
The caller must ensure independence, the prespecified evaluation distribution,
and a fixed sample size. This API cannot verify those statistical assumptions.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Decision:
    blocks: int
    mean_difference: float
    upper_bound: float
    replace: bool

def decide(baseline, candidate, *, comparisons, alpha=0.05):
    baseline, candidate = list(baseline), list(candidate)
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError('paired nonempty blocks required')
    if isinstance(comparisons, bool) or not isinstance(comparisons, int) or comparisons < 1:
        raise ValueError('total prespecified comparison count required')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0,1)')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in baseline + candidate):
        raise ValueError('losses must be finite and bounded in [0,1]')
    n = len(baseline)
    mean = math.fsum(c-b for b,c in zip(baseline,candidate))/n
    radius = math.sqrt(2*math.log(comparisons/alpha)/n)
    upper = min(1.0, mean+radius)
    return Decision(n, mean, upper, upper < 0)

def sufficient_blocks(gain, *, comparisons=10, alpha=0.05, beta=0.2):
    """Worst-case sufficient n for power >=1-beta if true gain >=gain.

    Gain is in bounded absolute loss units. This is not an empirical power
    estimate; Hoeffding controls both the selection radius and sampling error.
    """
    if not 0 < gain <= 1 or not 0 < beta < 1 or not 0 < alpha < 1:
        raise ValueError('invalid gain or probability')
    if isinstance(comparisons,bool) or not isinstance(comparisons,int) or comparisons<1:
        raise ValueError('invalid comparison count')
    return math.floor(2*(math.sqrt(math.log(comparisons/alpha))+
                          math.sqrt(math.log(1/beta)))**2/gain**2)+1

def empirical_bernstein(baseline, candidate, *, comparisons, alpha=0.05):
    """Maurer-Pontil (2009), Theorem 4, applied to D in [-1,1].

    Fixed sample, frozen predictors, i.i.d. independent diagnostic blocks.
    Bonferroni across comparisons. Post hoc use is a development diagnostic.
    """
    baseline, candidate = list(baseline), list(candidate)
    validated = decide(baseline, candidate, comparisons=comparisons, alpha=alpha)
    n = validated.blocks
    if n < 2:
        raise ValueError('sample variance requires at least two blocks')
    diff = [c-b for b,c in zip(baseline,candidate)]
    variance = math.fsum((d-validated.mean_difference)**2 for d in diff)/(n-1)
    logterm = math.log(2*comparisons/alpha)
    radius = math.sqrt(2*variance*logterm/n) + 14*logterm/(3*(n-1))
    upper = min(1.0, validated.mean_difference+radius)
    return Decision(n, validated.mean_difference, upper, upper < 0)
