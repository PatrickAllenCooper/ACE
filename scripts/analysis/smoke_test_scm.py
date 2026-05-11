#!/usr/bin/env python3
"""Smoke-test LargeScaleSCM extensions: canonical 30, anonymised 30, 50."""
import sys
import numpy as np

sys.path.insert(0, ".")
from experiments.large_scale_scm import LargeScaleSCM

print("=== Test 1: 30-node, canonical (regression) ===")
np.random.seed(42)
s30 = LargeScaleSCM(n_nodes=30)
print(f"  n_nodes={s30.n_nodes}, layer_sizes={s30.layer_sizes}")
print(f"  first 5 nodes: {s30.nodes[:5]}")
print(f"  last 3 nodes:  {s30.nodes[-3:]}")
print(f"  X3 parents:    {s30.get_parents('X3')}")
print(f"  X18 parents:   {s30.get_parents('X18')}")

print()
print("=== Test 2: 30-node, anonymised ===")
np.random.seed(42)
s30a = LargeScaleSCM(n_nodes=30, anonymize=True, anonymize_seed=42)
print(f"  first 5 nodes: {s30a.nodes[:5]}")
print(f"  alias[X3]:     {s30a.alias['X3']}")
print(f"  parents of {s30a.alias['X18']}: "
      f"{s30a.get_parents(s30a.alias['X18'])}")

print()
print("=== Test 3: 50-node (new size) ===")
np.random.seed(42)
s50 = LargeScaleSCM(n_nodes=50)
print(f"  n_nodes={s50.n_nodes}, layer_sizes={s50.layer_sizes}")
print(f"  total = {sum(s50.layer_sizes)}")
print(f"  X1 parents (root, should be []): {s50.get_parents('X1')}")
print(f"  X50 parents (last leaf): {s50.get_parents('X50')}")

print()
print("=== Test 4: generate works on all three ===")
for label, scm in [("30 canonical", s30),
                   ("30 anon",      s30a),
                   ("50 canonical", s50)]:
    np.random.seed(0)
    data = scm.generate(n_samples=5)
    n_keys = len(data.keys())
    ok = "OK" if n_keys == scm.n_nodes else "FAIL"
    print(f"  {label}: data has {n_keys} variables "
          f"(expected {scm.n_nodes})  [{ok}]")
