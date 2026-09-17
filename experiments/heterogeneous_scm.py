"""Heterogeneous-mechanism SCM family.

Same hierarchical graph generator as LargeScaleSCM, but each non-root
mechanism is drawn from a menu of functional forms instead of being
(near-)linear. The Sept 2026 metric audit showed that on LargeScaleSCM no
acquisition rule -- ACE, Bayesian OED, max-variance -- beats Random: every
mechanism is about equally easy and uniform interventions are near-optimal
coverage of the evaluation domain. This family has mechanisms whose
difficulty (for a small ReLU student on broad-range inputs) differs by an
order of magnitude, which is the regime where choosing *where* to intervene
can matter.

Forms, with z = sum_p c_p x_p (c_p ~ U(0.3, 0.7) as in LargeScaleSCM):

  linear     z
  quadratic  0.25 z^2
  sin        2 sin(1.2 z)
  tanh       3 tanh(z)
  abs        |z|
  product    z + 0.3 x_p1 x_p2         (needs >= 2 parents, else linear)

Form assignment and coefficients are fixed at construction from coeff_seed,
so an ACE-side adapter can reproduce the system exactly (as for LargeScaleSCM).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np
import torch

from experiments.large_scale_scm import LargeScaleSCM

FORMS = ("linear", "quadratic", "sin", "tanh", "abs", "product")
FORM_WEIGHTS = (0.25, 0.15, 0.15, 0.15, 0.15, 0.15)


class HeterogeneousSCM(LargeScaleSCM):
    def freeze_coefficients(self, seed=None):
        super().freeze_coefficients(seed)
        rng = np.random.RandomState((seed if seed is not None else 0) + 7919)
        self.forms: Dict[str, str] = {}
        for node in self.nodes:
            parents = self.get_parents(node)
            if not parents:
                continue
            f = rng.choice(FORMS, p=FORM_WEIGHTS)
            if f == "product" and len(parents) < 2:
                f = "linear"
            self.forms[node] = str(f)

    def form_counts(self) -> Dict[str, int]:
        return dict(Counter(self.forms.values()))

    def mechanisms(self, data, node, n_samples=1):
        n = next(iter(data.values())).shape[0] if data else n_samples
        noise = torch.randn(n) * self.noise_std
        parents = self.get_parents(node)
        if not parents:
            return torch.randn(n)
        z = torch.zeros(n)
        for p in parents:
            z = z + self.coeffs[node][p] * data[p]
        f = self.forms[node]
        if f == "linear":
            y = z
        elif f == "quadratic":
            y = 0.25 * z ** 2
        elif f == "sin":
            y = 2.0 * torch.sin(1.2 * z)
        elif f == "tanh":
            y = 3.0 * torch.tanh(z)
        elif f == "abs":
            y = z.abs()
        elif f == "product":
            y = z + 0.3 * data[parents[0]] * data[parents[1]]
        else:
            raise ValueError(f)
        return y + noise
