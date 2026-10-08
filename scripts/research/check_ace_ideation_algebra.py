"""Finite analytic witnesses for new ideation; no models, fitting or responses.

These checks do not verify an empirical assumption or a general theorem.
They are separate from frozen experiment and accepted manuscript test suites.
"""
from fractions import Fraction as F
import json


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mv(matrix, vector):
    return [dot(row, vector) for row in matrix]


def inverse2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    det = a * d - b * c
    return [[d / det, -b / det], [-c / det, a / det]]


def checks():
    # Direct precision inversion versus a rank-one covariance update.
    covariance = [[F(2), F(1)], [F(1), F(3)]]
    x, g, noise = [F(1), F(2)], [F(2), F(-1)], F(3)
    precision = inverse2(covariance)
    updated = inverse2([[precision[i][j] + x[i] * x[j] / noise
                         for j in range(2)] for i in range(2)])
    cx = mv(covariance, x)
    denom = noise + dot(x, cx)
    rank_update = [[covariance[i][j] - cx[i] * cx[j] / denom
                    for j in range(2)] for i in range(2)]
    assert updated == rank_update
    observed_gain = dot(g, mv(covariance, g)) - dot(g, mv(updated, g))
    assert observed_gain == dot(g, cx) ** 2 / denom

    # Target relevance reverses the ranking by observation variance.
    terminal_covariance = [[F(9), F(0)], [F(0), F(1)]]
    target = [F(0), F(1)]
    gains = []
    for feature in ([F(1), F(0)], [F(0), F(1)]):
        cov_feature = mv(terminal_covariance, feature)
        gains.append(dot(target, cov_feature) ** 2 /
                     (1 + dot(feature, cov_feature)))
    assert gains == [F(0), F(1, 2)]

    # Compute ensemble covariances directly, independently of reduced formula.
    members = [F(-1), F(0), F(1)]
    references = [F(-2), F(0), F(1)]
    ref_variance = sum(dot([z * u for z in members],
                           [z * u for z in members]) / 3
                       for u in references) / 3
    scores, variances = [], []
    for scale in [F(1), F(2), F(3)]:
        predictions = [z * scale for z in members]
        variance = dot(predictions, predictions) / 3
        covariances = [dot(predictions, [z * u for z in members]) / 3
                       for u in references]
        score = sum(c * c for c in covariances) / 3 / (variance + F(1, 2))
        assert score == variance * ref_variance / (variance + F(1, 2))
        assert sum(c * c for c in covariances) / 3 / variance == ref_variance
        variances.append(variance)
        scores.append(score)
    assert scores == sorted(scores) and len(set(scores)) == 3

    # Signed fork paths cancel; recursive error and explicit path sum agree.
    residual = [F(1, 10), F(1, 5), F(1, 20)]
    jacobian = [[F(0), F(0), F(0)],
                [F(2), F(0), F(0)],
                [F(3), F(-1), F(0)]]
    error = []
    for i in range(3):
        error.append(residual[i] + dot(jacobian[i][:i], error))
    # I + H + H^2, applied to r, without matrix inversion.
    hr = mv(jacobian, residual)
    hhr = mv(jacobian, hr)
    assert error == [residual[i] + hr[i] + hhr[i] for i in range(3)]
    assert error[-1] == F(-1, 20)
    assert mv(jacobian, hhr) == [0, 0, 0]
    # Perfectly correlated local errors cancel in a contrast; a diagonal
    # covariance approximation would instead report variance two.
    contrast = [F(1), F(-1)]
    assert dot(contrast, mv([[F(1), F(1)], [F(1), F(1)]], contrast)) == 0
    assert dot(contrast, contrast) == 2

    # One well-scaled quadratic step decreases training loss but can increase
    # a different deployment loss. No optimizer or training dataset is used.
    hessian = [[F(1), F(0)], [F(0), F(4)]]
    theta = [F(2), F(1)]
    gradient, eta = mv(hessian, theta), F(1, 5)
    after = [t - eta * g0 for t, g0 in zip(theta, gradient)]
    objective = lambda v: dot(v, mv(hessian, v)) / 2
    change = -eta * dot(gradient, gradient) + eta ** 2 * dot(
        gradient, mv(hessian, gradient)) / 2
    assert objective(after) - objective(theta) == change < 0
    assert sum((a - t) ** 2 for a, t in zip(after, theta)) == F(4, 5)

    # More samples on the same diagonal cannot constrain its normal direction.
    normal = [F(1), F(-1)]
    information = [[F(5), F(5)], [F(5), F(5)]]
    assert mv(information, normal) == [0, 0]
    assert dot(normal, mv([[F(6), F(4)], [F(4), F(6)]], normal)) == 4
    endpoint_information, uniform_information = F(25), F(25, 3)
    assert endpoint_information / uniform_information == 3

    return {
        "status": "passed",
        "scope": "fabricated exact rational arithmetic; no empirical qualification",
        "groups": ["covariance update", "target ranking reversal", "rank-one IVR",
                   "signed DAG propagation and covariance", "quadratic descent",
                   "design nullspace and scalar endpoint information"],
        "target_gains": [str(v) for v in gains],
        "rank_one_scores": [str(v) for v in scores],
        "signed_terminal_error": str(error[-1]),
        "quadratic_loss_before": str(objective(theta)),
        "quadratic_loss_after": str(objective(after)),
        "new_fits": 0,
        "new_responses": 0,
        "checkpoint_loads": 0,
    }


if __name__ == "__main__":
    print(json.dumps(checks(), indent=2))
