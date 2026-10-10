"""Preserve original native chart XML and literal workbooks lost during import/export.
Only layout frames change. Chart values, categories, formulas and workbook bytes stay original.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as E
import json,hashlib
root=Path('/Users/pat/code/ACE');build=root/'.codex-artifacts/ace-mechanism-deck-v9/build'
source=root/'presentations/ACE_theory_and_evidence/ACE_mechanisms_and_evidence_v8.pptx'
with ZipFile(source) as z: old={n:z.read(n) for n in z.namelist()}
with ZipFile(build/'draft.pptx') as z: new={n:z.read(n) for n in z.namelist()}
ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
parts=[]
for i in range(1,3):
 name=f'ppt/slides/charts/chart{i}.xml'
 def values(data):
  return [[x.text for x in c.findall('c:pt/c:v',ns)] for c in E.fromstring(data).findall('.//c:numCache',ns)]
 assert values(old[name])==values(new[name]),name
 for n in [name,f'ppt/slides/charts/_rels/chart{i}.xml.rels',f'ppt/embeddings/chart-data-snapshot-{i:03}.xlsx']:
  new[n]=old[n];parts.append({'part':n,'source_sha256':hashlib.sha256(old[n]).hexdigest()})
ct='http://schemas.openxmlformats.org/package/2006/content-types'
E.register_namespace('',ct)
a=E.fromstring(new['[Content_Types].xml']); b=E.fromstring(old['[Content_Types].xml'])
for elem in b:
 if elem.get('Extension')=='xlsx' or elem.get('PartName','').startswith('/ppt/embeddings/'):
  key='Extension' if elem.tag.endswith('Default') else 'PartName'
  if not any(x.tag==elem.tag and x.get(key)==elem.get(key) for x in a):a.append(elem)
new['[Content_Types].xml']=E.tostring(a,encoding='utf-8',xml_declaration=True)
with ZipFile(build/'draft-preserved.pptx','w',ZIP_DEFLATED) as z:
 for n,data in new.items():z.writestr(n,data)
(build/'chart_source_preservation.json').write_text(json.dumps({'reason':'Imported charts dropped workbook relationships; exact source chart and workbook bytes restored, with unchanged numeric caches and retained slide layout frames.','parts':parts},indent=2)+'\n')
print('Two original curve charts, relationships and workbooks preserved exactly')
