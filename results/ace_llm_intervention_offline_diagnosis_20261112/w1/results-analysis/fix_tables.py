p = 'make_tables.py'
s = open(p).read()
old = """th = pd.read_csv('threshold_sweep.csv').set_index('threshold')
TAB['threshold'] = md(th[['at_target_at_init', 'target_risk_never', 'no_policy_attains', 'informative(init>th & TR attains)', 'TR_mean', 'best8_mean', 'TR_minus_floor', 'random-TR mean', 'random-TR sd', 'random-TR d/sd', 'local-TR sd']])
TAB['threshold_means'] = md(pd.read_csv('threshold_sweep_policy_means.csv').set_index('threshold'))"""
new = """th = pd.read_csv('threshold_sweep.csv')
th['threshold'] = th.threshold.map(lambda v: ('%.4f' % v))
for c in ['at_target_at_init', 'target_risk_never', 'no_policy_attains', 'informative(init>th & TR attains)']:
    th[c] = th[c].astype(int)
th = th.rename(columns={'at_target_at_init': 'at target after init', 'target_risk_never': 'TR never attains', 'no_policy_attains': 'no policy attains',
                        'informative(init>th & TR attains)': 'informative for TR', 'TR_mean': 'TR mean cost', 'best8_mean': 'best-of-8 mean', 'TR_minus_floor': 'TR minus floor',
                        'random-TR mean': 'random-TR mean', 'random-TR sd': 'random-TR SD', 'random-TR d/sd': 'random-TR d_z', 'local-TR sd': 'local_info-TR SD'})
TAB['threshold'] = md(th.set_index('threshold')[['at target after init', 'TR never attains', 'no policy attains', 'informative for TR', 'TR mean cost', 'best-of-8 mean', 'TR minus floor', 'random-TR mean', 'random-TR SD', 'random-TR d_z', 'local_info-TR SD']])
tm = pd.read_csv('threshold_sweep_policy_means.csv'); tm['threshold'] = tm.threshold.map(lambda v: ('%.4f' % v))
TAB['threshold_means'] = md(tm.set_index('threshold'))"""
assert s.count(old) == 1
s = s.replace(old, new)
old2 = """TAB['uninformative'] = md(pd.DataFrame({'at target after init': pd.concat([meta[at].groupby(k).size() for k in ['graph_stratum', 'regime', 'cost_stratum']]),
                                        'of': pd.concat([meta.groupby(k).size() for k in ['graph_stratum', 'regime', 'cost_stratum']])}).fillna(0).astype(int))"""
new2 = """_u = pd.DataFrame({'at target after init': pd.concat([meta[at].groupby(k).size() for k in ['graph_stratum', 'regime', 'cost_stratum']]),
                   'of': pd.concat([meta.groupby(k).size() for k in ['graph_stratum', 'regime', 'cost_stratum']])}).fillna(0).astype(int)
_u = _u.loc[['chain', 'fork', 'collider', 'matched_low', 'matched_high', 'shifted_low', 'shifted_high', 'uniform', 'heterogeneous']]
_u.index.name = 'stratum'
TAB['uninformative'] = md(_u)"""
assert s.count(old2) == 1
s = s.replace(old2, new2)
open(p, 'w').write(s)

p = 'assemble_notes.py'
s = open(p).read()
reps = [
("""log10 NMSE at cost 12, 16, 24 and final:

{T['paired_log_e12_random']}

{T['paired_log_e16_random']}

{T['paired_log_e24_random']}

{T['paired_log_e32_random']}""",
"""log10 NMSE at total cost 12:

{T['paired_log_e12_random']}

log10 NMSE at total cost 16:

{T['paired_log_e16_random']}

log10 NMSE at total cost 24:

{T['paired_log_e24_random']}

log10 final NMSE (cost 32):

{T['paired_log_e32_random']}"""),
("A Wilcoxon test drops zeros, so its effective sample is the nonzero count (about 45 to 55 percent of systems here).",
 "A Wilcoxon test drops zeros, so its effective sample is the nonzero count (10 to 13 of 24 systems here)."),
("5. **Graph matters.** In uniform-menu forks TR uses mostly Y and joint actions (65 of 96). In uniform-menu colliders it mixes x+, y+, y- and obs. local_info uses Y-involving actions almost only in colliders (34 of its 36).",
 "5. **Graph matters.** In uniform-menu forks TR uses mostly Y and joint actions (65 of 96). In uniform-menu colliders it mixes x+, y+, y- and obs. local_info uses Y-involving actions almost only in colliders (34 of its 36).\n6. **Observe more when the known noise is high.** The observation share is 39 and 47 percent for TR in matched_high and shifted_high against 25 and 42 percent in the two low-noise regimes. For local_info it is 65 and 70 percent against 45 and 43 percent."),
("  - \"Direct SCM planning performed better\" holds for the mean of the predeclared endpoint. After Holm correction only dpo is separated from TR (0.048). TR is not separated from local_info, coverage, ranker or frozen on any endpoint, and local_info is better on the informative systems for error-based endpoints.",
 "  - \"Direct SCM planning performed better\" holds for the mean of the predeclared endpoint. After Holm correction only dpo is separated from TR (0.048). On that endpoint TR is not separated from local_info, coverage, ranker or frozen (sign-flip p 0.18 to 0.84). On post hoc endpoints frozen is nominally worse than TR on log-area (0.023) and on the error at cost 12 (0.012), and the ranker (0.0015) and coverage (0.035) are nominally worse at threshold 0.02. local_info is never separated from TR and is better on the informative systems for error-based endpoints."),
("None of the learned-arm contrasts (training recipe, supervised stage, preference stage, LLM versus numerical ranker) is distinguishable from zero on any endpoint.",
 "None of the learned-arm contrasts (training recipe, supervised stage, preference stage, LLM versus numerical ranker) is distinguishable from zero on any of the four endpoints computed for them (capped cost at 0.04 and at 0.02, log-area, final error; smallest sign-flip p 0.24)."),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('patched')
