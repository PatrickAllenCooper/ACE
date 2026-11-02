from pathlib import Path
import re,hashlib,json
root=Path('/Users/pat/code/ACE');p=root/'paper/aistats_ace_2027/paper.tex';s=p.read_text()
assert hashlib.sha256(p.read_bytes()).hexdigest()=='3cbdf644c8d48dda7b05b35b8c2d29c17dffa8988ca431fe53383b9f6d3c833a'
def section(start,end,new):
 global s
 a=s.index(start,s.index('\\begin{document}')); b=s.index(end,a);s=s[:a]+new.strip()+'\n\n'+s[b:]
def replace(old,new):
 global s
 assert old in s,old[:100];s=s.replace(old,new,1)
section(r'\begin{abstract}',r'\end{abstract}',r'''
\begin{abstract}
We assess post-collection refitting of structural causal model (SCM) surrogates
with known graphs and measured intermediates. A prospective study uses twenty
independent systems in each of two noise-disabled families, with five or thirty
nodes. Its registered analysis weights balanced and matched-random collection
histories equally within each system. At shared initialization zero, the pooled
results favor delivery over a flat refit in both families and over rolling
learning in the thirty-node family. Five-node delivery instead has 2.46 times
the rolling error, a reversal whose cause remains undiagnosed. The flat comparison
also depends strongly on collection history. For thirty-node balanced histories,
the delivery/flat error ratio is 0.912 and delivery improves only 10 of 20 systems,
compared with a ratio of 0.168 for matched-random histories. Across the three
fully trained arms and two families, median root-mean-square errors are about
0.5\% to 3.4\% of the training target's standard deviation. Separate archived
studies favor long fits on retained histories of one emulator but find that
Fourier regression beats delivery in all eleven physical conditions. Conditional
theory explains why eligible-row fitting, joint support and prediction through
a learned chain must be considered separately. These results support evaluating
a final refit against the rolling learner and simple controls, with explicit
attention to collection history, initialization and unequal training budgets.
''')
section(r'\paragraph{Contributions.}',r'\section{Setting and delivery procedure}',r'''
\paragraph{Contributions.}
Our main evidence is a frozen prospective comparison across forty independently
parameterized systems. The registered average over two collection histories
favors delivery over the selected flat refit in both families, but the
thirty-node balanced histories show mixed results. Delivery loses to rolling
in the five-node family and wins in the thirty-node family. All four registered
contrasts and every system are retained.

Two archived studies provide context. Repeated histories of one emulator and
an exploratory scratch-fit matrix compare retained data, paid observations
that never entered a live update, and fitting effort. Long fits on retained
histories improve over final-buffer and short fits. Adding the unused paid rows
has a smaller aggregate continuous-error effect. These contrasts do not explain
the prospective five-node reversal or partition the original online-to-delivery
gain. A separate physical case study finds that a domain-informed Fourier
predictor beats neural delivery in every condition.

Finally, elementary eligibility and propagation results clarify when a measured
row can train a mechanism and why good local fits need not give good downstream
predictions. They are not new identification theorems. The practical lesson is
to evaluate the delivered refit separately from the rolling learner, keep a
simple direct or domain-informed control, and score the complete prediction
chain. The evidence does not justify an automatic switch to a long SCM refit.
''')
replace(r'''Our central question is empirical: \emph{what does a completed experimental
history buy at delivery time, and what accounts for that benefit?} We investigate
three explanations:''',r'''Our central question is empirical: \emph{when does refitting a completed
experimental history improve the delivered predictor?} In an archived task,
we compare three changes:''')
replace('whose extra computation is spent after the\nexperimental responses have already been obtained.','whose computation is spent after the\nexperimental responses have already been obtained.')
section(r'\paragraph{Online learning and final prediction.}',r'\paragraph{Prediction-dependent inputs and physical validation.}',r'''
\paragraph{Online learning and final prediction.}
Online-to-batch results obtain predictors with risk bounds from online hypotheses
under specified sampling assumptions \citep{cesabianchi2004generalization}.
Our procedure instead fits anew on a retained interventional history and does
not inherit those bounds. \citet{ash2020warm} show that warm-starting a neural
network on growing data can generalize worse than training from scratch even
when training losses are similar. That work is a close precedent for separating
continued training from a fresh fit. We do not test their proposed remedy or
attribute our five-node reversal to the same cause.
GDumb also stores streaming examples and trains from scratch at test time
\citep{prabhu2020gdumb}. Experience replay retains earlier samples during
continual learning \citep{rolnick2019replay}. Our contribution is therefore not
a new fitting principle. It is an assessment of intervention eligibility,
measured-parent training versus predicted-parent deployment, and data-use and
training-budget differences in specified SCM recipes.
''')
replace('without exclusion or rescue tuning. The registered numerical floor is below every observed paired error and does\nnot alter these ratios.','without exclusion or rescue tuning. The registered numerical floor is below every observed paired error and does\nnot alter these ratios. An interrupted acquisition attempt for the final history\nwas retained and continued under an explicit amendment before scoring, as\ndetailed in Appendix~\\ref{app:accounting}.')
replace('The full registered matrix completed and passed an independent audit of source,\ninputs, eligibility masks, original artifacts, fitted-model hashes and all score\nfiles. With initialization zero,','The full registered matrix completed. With initialization zero,')
replace('training schedule, not the best prediction attainable by an optimally allocated\nflat compute budget. Table~\\ref{tab:attribution-histories} retains each history.','training schedule, not the best prediction attainable by an optimally allocated\nflat compute budget. In fact, its online-relative NMSE ratio is 1.234, worse\nthan 0.948 for the 30,000-update flat fit. Thus the matched-CPU comparison uses\nthe weaker of these two flat schedules. Table~\\ref{tab:attribution-histories} retains each history.')
replace('The protocol was frozen before collecting responses or opening held-out\noutcomes. The original audit, independent saved-checkpoint reproduction and\nboth registered and descriptive report exports are complete. The recipe,\nflat control and short-fit ablation were selected at the attribution gate and\nremain unchanged. The full matrix includes every registered system and fit.','We fixed the protocol before collecting responses or inspecting held-out\noutcomes. The delivery recipe, flat control and short-fit ablation were selected\nin the archived attribution study. No prospective result changed them. The\ncomplete study includes every registered system and fit. Section~\\ref{sec:reproducibility}\ndistinguishes our retained execution records from the evidence in the anonymous\nsupplement.')
section(r'\subsection{Comparators and separation of fitting from evaluation}',r'\subsection{Registered primary analysis}',r'''
\subsection{Comparators and separation of fitting from evaluation}
Delivery and flat both use 30,000 full-batch updates per fitted head. Rolling
processes all 400 arrivals after the shared calibration step, including the
first 50 rows. Each arrival triggers 20 fast updates on the current observation
and 100 consolidation updates on a buffer of up to 50 rows plus a second copy
of the current observation. This gives $400(20+100)=48{,}000$ updates per head,
with persistent Adam state. Rolling therefore receives more updates per head,
but on smaller batches. Update counts alone are not compute equivalence.

The structured learners fit three heads in the five-node family and 25 in the
thirty-node family. Flat fits one head. Thirty-node delivery therefore uses
750,000 head-updates against flat's 30,000, and it receives intermediate labels
that flat does not. The five-node parameter counts are 13,123 for the structured
model and 4,417 for flat. The thirty-node flat model has 4,609 parameters.
These are complete-recipe comparisons, not tests of architecture at matched
supervision or computation. Root-only histories also do not identify responses
to arbitrary internal interventions.

Initialization zero is fixed for the primary comparison. Head seeds depend on
the initialization index and topological position, not the system identity.
The same seed scheme is therefore shared across systems within each family.
The primary intervals describe variation across systems conditional on that
initialization. Delivery and flat also have fits at initializations one and two
for descriptive sensitivity. Rolling and the 100-update SCM ablation each have
one fit per history. No scored initialization is selected for deployment.
All 640 fits were saved before any held-out response was generated. All forty
systems completed without replacement or filtering.
''')
section(r'\subsection{Prospective results and boundaries}',r'\section{Archived physical prediction}',r'''
\subsection{Prospective results and boundaries}
Table~\ref{tab:prospective-primary} reports the registered equal-weight average
over the two history types. Delivery is worse than rolling in the five-node
family, with ratio \BFiveOnlineRatio{} and marginal interval
[\BFiveOnlineCILow{}, \BFiveOnlineCIHigh{}]. In the thirty-node family it is
better, with ratio \BThirtyOnlineRatio{} and interval
[\BThirtyOnlineCILow{}, \BThirtyOnlineCIHigh{}]. Against flat, the pooled ratios
are \BFiveFlatRatio{} and \BThirtyFlatRatio{}. Both flat contrasts and the
thirty-node rolling contrast meet the registered rule. The five-node result
favors rolling, rather than establishing equivalence. The $0.8$ point-estimate
criterion is not 95\% assurance of a 20\% population gain.

\input{delivery_prospective_primary_table}

\paragraph{Collection history changes the comparison.}
Against flat, delivery has balanced-history ratios of 0.793 and 0.912 in the
five- and thirty-node families, with improvements in 13 and 10 of 20 systems.
The matched-random ratios are 0.160 and 0.168, with improvements in 19 and 18
systems. Most of the pooled flat advantage therefore occurs on matched-random
histories. The thirty-node balanced result is mixed, not evidence of equivalence.
These are descriptive within-history comparisons, not tests of which collection
strategy is better. Longer fitting beats the 100-update ablation in both
families but does not remove the five-node loss to rolling. Its cause remains
undiagnosed. The archived propagation diagnostic concerns a different task and
cannot explain this prospective reversal.

\paragraph{Absolute error and initialization matter.}
At initialization zero, median RMSE as a percentage of the training target's
standard deviation is about 2.2\%, 1.6\% and 3.4\% for delivery, rolling and flat
in the five-node family. The corresponding thirty-node values are 0.53\%,
0.85\% and 0.87\%. These medians use the forty system/history cells per arm
and family. They describe prediction accuracy, not downstream decision value
or saved interventions.

The pooled comparisons keep the same direction at each registered delivery
initialization and when log errors are averaged over the three initializations.
This stability does not imply reliable behavior in every history. In system
5:08 under matched-random collection, initialization one gives delivery NMSE
about 0.0238, roughly 1,044 times the rolling error. Appendix~\ref{app:prospective-complete}
reports the initialization contrasts and this failure alongside all original
absolute errors, system ratios, history summaries and ablations. No primary
floor is activated, and initialization zero remains the deployment rule.
''')
section(r'\section{Reproducibility and compute accounting}',r'\section{Discussion and limitations}',r'''
\section{Reproducibility and compute accounting}
\label{sec:reproducibility}
The anonymous supplement contains the manuscript sources and complete saved
prospective error reports. These let readers reconstruct the reported
comparisons. We separately retain the protocols, response ledgers, checkpoints,
source versions and failed attempts. Our claims about protocol timing, fit
sealing and execution checks are based on those records. They are author-reported
process evidence, not independently established by the anonymous supplement.
Saved-checkpoint reproduction covered all 640 prospective fits without new
optimization or responses. The full executable archive retains required named
attribution and is withheld from anonymous upload.

The identified prospective execution and reporting chain used 36.9431 allocated
CPU-core-hours across 59 one-CPU allocations, including failures. Fit process
time and scheduler allocation overlap and are not added. Complete historical
confirmation CPU and whole-project preparation costs are unknown.
Appendices~\ref{app:accounting} and~\ref{app:provenance} give the scope and
reproduction details.
''')
section(r'\section{Discussion and limitations}',r'\section{Conclusion}',r'''
\section{Discussion and limitations}
\label{sec:discussion}
\paragraph{What the comparisons establish.}
A final refit should be evaluated as its own prediction procedure. In the
archived scratch-fit matrix, retaining a completed history and fitting longer
both matter. Retention also changes input scaling, while the online comparator
carries persistent weights and optimizer state. These contrasts therefore do
not isolate the causes of the online-to-delivery difference. The prospective
study tests the selected recipes on new systems and held-out actions. Its
five-node reversal remains unexplained, and its flat advantage is much larger
on matched-random histories than on balanced histories.

\paragraph{Why local fit is insufficient.}
A known graph specifies which observations can train each mechanism, but also
makes later predictions depend on earlier ones. A small error on measured
parents does not validate the complete learned chain. Nor does a coordinate
bounding box establish joint support. The theory states assumptions under
which reuse and propagation can be analyzed. It does not certify fitted
residuals, explain every empirical reversal, or identify arbitrary internal
interventions from root-only observations.

\paragraph{Scope of the evidence.}
The prospective results are conditional on two restricted, noise-disabled
generator families and a shared initialization scheme. The archived attribution
uses an exposed grid from one emulator. Its scored median across three refits
summarizes sensitivity, not a deployable selection rule. The physical study
uses one mechanism in one apparatus, where Fourier regression beats delivery
in every condition. Conditions and repeated readings are not independent
physical systems. Unequal intermediate supervision, parameter counts and
fitting work limit architecture claims. None of these studies measures
interventions saved by changing an acquisition policy.

\paragraph{Broader impact.}
An inaccurate surrogate can mislead a later experimental decision, especially
outside observed joint support. The archived emulator includes communication
and jamming variables, but its hand-coded scores do not validate an operational
system. Deployment would require application-specific validation, intervention
limits and human oversight. The adverse comparisons and support counterexamples
show why a fitted causal graph should not be treated as a safety certificate.
''')
section(r'\section{Conclusion}',r'\bibliographystyle',r'''
\section{Conclusion}
Post-collection refitting can improve a delivered predictor, but the result
depends on the comparison. In the registered equal-weight average over balanced
and matched-random histories, delivery beats flat in both prospective families.
Most of that advantage occurs on matched-random histories. On thirty-node
balanced histories, the ratio is 0.912 and only 10 of 20 systems improve.
Delivery also beats rolling in the thirty-node family but loses in the five-node
family, for an undiagnosed reason. Absolute errors are small, initialization
can produce severe individual failures, and the training budgets differ.
The archived studies add two lessons: retained histories can support better
scratch fits, and a simple domain-informed model can beat neural delivery.
Evaluate the final refit, the rolling learner and suitable simple controls on
the same held-out task. Check the composed prediction rather than relying on
local fit alone.
''')
replace(r'\section{Historical execution and release accounting}',r'\section{Historical execution and release accounting}'+'\n'+r'\label{app:accounting}')
# Narrative punctuation only. Style and retained generated evidence above the document are untouched.
a=s.index(r'\begin{document}');prefix=s[:a];body=s[a:]
body=re.sub(r';(\s+)([a-z])',lambda m:'.'+m.group(1)+m.group(2).upper(),body)
body=body.replace(';','.')
assert '—' not in body and '---' not in body
p.write_text(prefix+body)
# Add the directly relevant retraining precedent.
bib=p.parent/'delivery_references.bib'
with bib.open('a') as f:f.write('''\n@inproceedings{ash2020warm,\n  title={On Warm-Starting Neural Network Training},\n  author={Ash, Jordan T. and Adams, Ryan P.},\n  booktitle={Advances in Neural Information Processing Systems},\n  volume={33}, year={2020},\n  url={https://papers.nips.cc/paper/2020/hash/288cd2567953f06e460a33951f55daaf-Abstract.html}}\n''')
# Prose-only punctuation and a clearer existing proof step. Assumptions and claims are unchanged.
theory=p.parent/'delivery_theory.tex';t=theory.read_text()
t=t.replace('The policy precedes\nthe current realization, and the stated retention condition preserves\nconditional mean zero.','The policy precedes the current realization. Hence\n$\\mathbb E[U_i\\mid V_{\\mathrm{Pa}(i)},A,H]=0$ before terminal selection.\nThe stated retention condition preserves this conditional mean zero.')
t=re.sub(r';(\s+)([a-z])',lambda m:'.'+m.group(1)+m.group(2).upper(),t).replace(';','.')
theory.write_text(t)
for name in ['delivery_method_diagram.tex','delivery_evaluation_design.tex']:
 q=p.parent/name;v=q.read_text();v=v.replace(';',',') if 'design' in name else re.sub(r';(\s+)([a-z])',lambda m:'.'+m.group(1)+m.group(2).upper(),v).replace(';','.')
 q.write_text(v)
print('Narrative draft revised in the same manuscript. Awaiting independent evidence checks before final wording.')
