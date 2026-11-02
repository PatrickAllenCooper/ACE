from pathlib import Path
p=Path('paper/aistats_ace_2027/paper.tex');s=p.read_text()
def change(a,b):
 global s
 assert a in s,a[:90]
 s=s.replace(a,b,1)
change('The archived attribution\nuses an exposed grid from one emulator. Its scored median across three refits\nsummarizes sensitivity, not a deployable selection rule.','Both archived studies\nuse an exposed grid from one emulator. Confirmation summarizes three scored\nrefits by their median, which is not a deployable selection rule. Attribution\nuses initialization zero for its primary comparisons.')
change('single deterministic emulator. The delivery procedure fits separate mechanism','single deterministic emulator with eight roots and five measured mechanisms.\nThe delivery procedure fits separate mechanism')
a=s.index('We use 20 independently parameterized systems in each of two graph-size strata.',s.index('\\begin{document}'));b=s.index('\n\nEach system supplies',a)
s=s[:a]+'''We use 20 independently parameterized systems in each of two families. The
five-node family has two roots and three measured mechanisms. Independent
coefficient perturbations vary an affine and sinusoidal target chain. Its target
depends effectively on one root, while the other branch is irrelevant to that
target. The thirty-node family has five layers, five roots and 25 mechanisms.
Each mechanism is a weighted parent sum, with a sinusoidal term at selected
nodes. Graphs and coefficients vary across systems and stay fixed across
responses. Appendix~\\ref{app:systems} gives their distributions and equations.
These are restricted generator families, not a representative sample of all
SCMs. They differ in structure and functions as well as node count, so the
family comparison does not isolate a scaling effect.'''+s[b:]
change('Each system supplies two shared 400-response histories: balanced varied-value\nroot interventions and matched random root interventions. Each also has one\nshared set of 400 held-out root actions.','''Each system supplies two shared 400-response histories. Balanced collection
favors the least-used of eleven levels at each root. Matched-random collection
samples roots uniformly in $[-3,3]$ under the same response budget and training
block exclusions. Each system also has 400 shared held-out root actions.''')
change('It is secondary for genuinely quantized targets outside the archival confirmation.\nIt is not interchangeable with continuous prediction error.','Exact-level error is secondary outside the archival confirmation and applies\nto genuinely quantized targets. It is not interchangeable with continuous error.')
change('No result is accepted from an\nincomplete primary matrix.','We report the complete registered primary matrix.')
change('The earlier\nproduct-form attenuation control is not retained. ','')
change('These are descriptive counts, not independent-world tests. Ten of the eleven','These are descriptive counts, not independent-world tests. Four neural\ncondition/arm entries have NMSE above one. Ten of the eleven')
change('Fourier regression outperforming delivery in\nevery condition rules out an external advantage over that control here. The\nfixed domain-informed control provides better predictions.','Fourier regression has lower held-out error than delivery in every condition.')
change('do not establish the chronology of the original protocol freeze. Python access\nhooks restrict trusted workers but are not an operating-system sandbox.','do not establish the chronology of the original protocol freeze.')
change('claim indexes. These files support checks of the reported numbers. They do not','''claim indexes. The published historical index retains empirical mappings but
omits obsolete agenda annotations. Its projection record identifies the original
index, whose pending and removed claim history remains preserved separately.
The prospective index states the scope of the current comparisons. These files
support checks of the reported numbers. They do not''')
# Compact operational narration, with all original detailed records retained externally.
a=s.index('The original prospective chain used 118,680 allocated CPU-seconds.',s.index('\\begin{document}'));b=s.index('\n\\section{Verification and evidence provenance}',a)
s=s[:a]+'''The identified prospective chain, including pilots, failed and successful
checkpoint reproductions, and reporting, used 132,995 allocated CPU-seconds
(36.9431 core-hours) across 59 one-CPU allocations. No GPU was used. Reserved
limits total 94.3683 of the 150 CPU-core-hour ceiling and include unused time.
The 640 original fits separately record about 110,802 process CPU-seconds and
113,596 summed elapsed seconds. These scopes overlap scheduler allocation and
must not be added to it. Detailed timing, memory and failed-attempt records are
retained separately. Complete project CPU and preparation, transfer, monitoring
and review overhead remain unknown.
'''+s[b:]
p.write_text(s)
q=Path('paper/aistats_ace_2027/delivery_evaluation_design.tex');t=q.read_text();t=t.replace(' Prospective results use the complete fixed-initialization matrix after independent checkpoint reproduction.','');q.write_text(t)
q=Path('paper/aistats_ace_2027/delivery_method_diagram.tex');t=q.read_text().replace('Save weights and fitting receipts','Save weights and training records').replace('clamps roots only and seals every fit','clamps roots only and saves every fit').replace('supervision therefore differ across recipes. This schematic supplies no\nextrapolation, identification or error-bound certification.','supervision therefore differ across recipes.');q.write_text(t)
q=Path('paper/aistats_ace_2027/delivery_theory.tex');t=q.read_text();t=t.replace('An intervention policy may depend on previous\nexperiments, but must choose its action before the current exogenous noises are\nrealized.','''The intervention policy is a function of previous experiments and auxiliary
randomness independent of those experiments and the current noise vector.
It chooses its action before the current exogenous noises are realized.''').replace('on the retained support, where $A$ is the current action','almost surely on the retained support, where $A$ is the current action');q.write_text(t)
print('C1-C6 and optional policy/accounting precision addressed, original numerical entries unchanged')
