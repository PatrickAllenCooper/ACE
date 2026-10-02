# Response-free action-menu separation check

Implemented `scripts/research/check_action_identifiability.py`; outputs and source hashes in `results/action_identifiability_diagnostic_20261002`. This evaluates declared functions, not environment observations. No simulation, model call or fitted experimental outcome is used.

For a menu A, form baseline design B=[1,x,y,xy]. For a candidate function vector h, calculate ||h-B B^+ h||²/||h||². A zero residual means the baseline reproduces that function on the menu. A positive residual establishes finite-menu functional separation only: it does not guarantee useful signal-to-noise ratio, distinguish all possible functions, or identify a causal graph. Intercept makes this comparator stronger than our prior three-feature model.

On the four corners, B has full row rank; it can represent **every** scalar response on those four points. The earlier odd-function argument was a special case for the intercept-free comparator. Repeating corner samples reduces noise but cannot resolve this representational ambiguity.

A three-level grid {-2,0,2}² separates quadratics, the selected saturating function, and a radial function, but x³=4x at those coordinates. It therefore still aliases cubic and linear mechanisms. A five-level grid {-2,-1,0,1,2}² separates all five declared examples: residual-energy fractions are .41176 for each quadratic, .11077 for cubic, .13079 for saturating, and .51543 for radial. These checks use no test responses. They do not justify a universal expressivity or novelty claim.

## Technical next step

Use the five-level grid as a fixed generic **diagnostic** distribution if pursuing the bounded repair implementation. Compare an intercept-enabled baseline, fixed residual candidate, and scratch control using identical data. Do not select action points by private change labels. Verify separately whether the fixed RBF architecture can represent each alternative adequately; a separating menu alone does not establish candidate capacity. Then freeze a single bounded implementation check before sampling, retaining the earlier negative pilot.

The scientific question becomes whether a reusable mechanism predictor can exploit informative interior interventions with fewer samples than strong generic and physical controls. A foundation-model contribution would require matched-history transfer across independently held-out systems. Neither this algebra nor our one-system pilot provides that result. Prior numerical-prior candidate remains stopped after its failed full gate; neural repair scaling remains blocked on unchanged-mechanism protection.
