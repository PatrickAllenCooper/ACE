from pathlib import Path
import json
root=Path('/Users/pat/code/ACE');b=root/'results/delivery_panel_revision_20261102';d=json.loads((b/'numerical_evidence.json').read_text());p=root/'paper/aistats_ace_2027/paper.tex';s=p.read_text()
s=s.replace('Across the three\nfully trained arms and two families, median root-mean-square errors are about','Across delivery, rolling and flat in the two families, median root-mean-square\nerrors are about')
needle='''These are descriptive within-history comparisons, not tests of which collection
strategy is better.'''
replacement='''Against rolling, the balanced-history ratios are 1.614 and 0.695, with 4 and
12 of 20 systems improving. The matched-random ratios are 3.748 and 0.108, with
3 and 19 improving. These are descriptive within-history comparisons, not tests
of which collection strategy is better.'''
assert needle in s;s=s.replace(needle,replacement)
rows=[]
for size in (5,30):
 for control in ('online','simpler'):
  primary=next(v for v in d['primary_init0_pooled'] if v['graph_size']==size and v['control_code']==control)
  byinit={v['delivery_inits'][0]:v for v in d['init1_init2_descriptive'] if v['graph_size']==size and v['control_code']==control}
  avg=next(v for v in d['seed_averaged_descriptive'] if v['graph_size']==size and v['control_code']==control)
  values=[primary['geometric_ratio'],byinit[1]['geometric_ratio'],byinit[2]['geometric_ratio'],avg['geometric_ratio']]
  rows.append(str(size)+' & '+('rolling' if control=='online' else 'flat')+' & '+' & '.join(format(v,'.3f') for v in values)+r'\\')
table=r'''
\paragraph{Additional descriptive initialization summaries.}
Table~\ref{tab:initialization-contrasts} summarizes the saved fits without
changing the registered initialization-zero analysis. The log-average column
first averages each arm's log NMSE over its three registered initializations
within a system and history. Rolling retains its single fit. Each comparison
then averages the two histories and the twenty systems with equal weight.
This is a score summary, not an ensemble of predictions or a rule for selecting
a seed. It adds no significance test or confidence interval.

\begin{table}[ht]
\centering\small
\begin{tabular}{rlrrrr}
\toprule
Nodes & Control & Init 0 & Init 1 & Init 2 & Log average\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\caption{Descriptive delivery/control NMSE ratios from the recorded fits.
Delivery and flat use matching initialization indices. Every rolling comparison
uses its one recorded initialization-zero fit. All four directions persist,
but these averages do not describe the worst individual history.}
\label{tab:initialization-contrasts}
\end{table}

Within a history, the ratio of the worst to best delivery NMSE across the three
initializations has median 2.02 in the five-node family and 5.23 in the thirty-node
family. System 5:08 under matched-random collection is a severe case:
initialization-one delivery NMSE is 0.02384, flat NMSE is 0.04815, and the
single rolling fit has NMSE $2.2844\times10^{-5}$. Delivery/rolling is 1043.55.
The complete CSV retains this cell. Neither the primary fit nor the history
was replaced.
'''
needle='% END EDITORIAL TABLE delivery_prospective_sensitivity_table'
assert s.count(needle)==1;s=s.replace(needle,needle+'\n'+table)
p.write_text(s)
(b/'descriptive_table_rows.json').write_text(json.dumps({'basis':'numerical_evidence.json','status':'additional descriptive summaries, not new tests or registered-primary changes','latex_rows':rows},indent=2)+'\n')
print('Added paired seed sensitivity and complete history comparison context.')
