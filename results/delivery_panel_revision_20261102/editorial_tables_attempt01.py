"""Embed text-edited display tables in the manuscript while retaining accepted report files.
Only display labels, spacing and captions change. Numeric tabular fields must match exactly.
"""
from pathlib import Path
import re,hashlib,json
root=Path('/Users/pat/code/ACE');p=root/'paper/aistats_ace_2027/paper.tex';s=p.read_text()
files=['delivery_attribution_table','delivery_history_table','delivery_physical_table','delivery_prospective_primary_table','delivery_prospective_sensitivity_table','delivery_prospective_absolute_table','delivery_prospective_descriptive_tables']
# Confirm actual existing include names rather than guessing.
includes=re.findall(r'\\input\{([^}]+)\}',s[s.index(r'\begin{document}'):])
files=[n for n in includes if 'table' in n and n!='delivery_evaluation_design']
records=[]
def numeric_fields(text):
 blocks=re.findall(r'\\begin\{tabular\}.*?\\midrule(.*?)\\bottomrule',text,re.S)
 return [re.findall(r'(?<![A-Za-z])[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?',b) for b in blocks]
def caption(text):
 # Captions have nested math braces, so locate through their label rather than a shallow brace regex.
 a=text.index('\\caption{');b=text.index('\n\\label',a)
 before=text[:a];cap=text[a:b];after=text[b:]
 cap=re.sub(r';(\s+)([a-z])',lambda m:'.'+m.group(1)+m.group(2).upper(),cap).replace(';','.')
 cap=cap.replace('initialization0','initialization 0').replace('history124753321','history 124753321')
 return before+cap+after
for name in dict.fromkeys(files):
 path=p.parent/(name+'.tex');raw=path.read_text();edited=raw
 # These strings are labels. No score, system identifier or inferential rule is changed.
 if 'prospective' in name:
  edited=edited.replace(' & online &',' & rolling &').replace(' & simpler &',' & flat &')
  edited=edited.replace('delivery/online','delivery/rolling').replace('delivery/simpler','delivery/flat')
  edited=edited.replace('Marginal95','Marginal 95').replace('Ratio/init0','Ratio/init 0')
 # Edit each complete table independently when a source contains several tables.
 edited=re.sub(r'\\begin\{table\}.*?\\end\{table\}',lambda m:caption(m.group()),edited,flags=re.S)
 if name=='delivery_prospective_primary_table':
  a=edited.index('\\caption{');b=edited.index('\n\\label',a)
  edited=edited[:a]+r'\caption{Registered delivery/control NMSE ratios at shared initialization zero. Each system weights balanced and matched-random history log ratios equally. Intervals are marginal 95\% intervals across twenty systems. Holm adjustment covers all four tests. The superiority rule requires ratio $\leq0.8$, upper interval $<1$, and adjusted $p<0.05$.}' +edited[b:]
 assert numeric_fields(raw)==numeric_fields(edited),name
 assert edited.count('\\begin{table}')==raw.count('\\begin{table}')
 tag=r'\input{'+name+'}'
 assert s.count(tag)==1,(name,s.count(tag))
 display='% BEGIN EDITORIAL TABLE '+name+'\n'+edited.rstrip()+'\n% END EDITORIAL TABLE '+name
 s=s.replace(tag,display)
 records.append({'source':str(path.relative_to(root)),'sha256':hashlib.sha256(raw.encode()).hexdigest(),'display_sha256':hashlib.sha256(edited.encode()).hexdigest(),'tables':raw.count('\\begin{table}'),'numeric_fields_unchanged':True})
p.write_text(s)
(root/'results/delivery_panel_revision_20261102/editorial_table_sources.json').write_text(json.dumps({'scope':'Presentation-only derivatives embedded in paper.tex. Accepted table files remain byte-identical.','tables':records},indent=2)+'\n')
print('Embedded',len(records),'source groups with numeric fields unchanged.')
