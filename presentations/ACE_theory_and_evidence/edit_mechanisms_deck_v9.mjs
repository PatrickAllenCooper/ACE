import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import {resolvePresentationFont,finalizePresentation,applyPresentationChartFont} from '/Users/pat/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/artifact_tool_utils.mjs';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
const ROOT='/Users/pat/code/ACE',OUT=ROOT+'/presentations/ACE_theory_and_evidence',DIR=ROOT+'/.codex-artifacts/ace-mechanism-deck-v9/build';
const SKILL='/Users/pat/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const INPUT=OUT+'/ACE_mechanisms_and_evidence_v8.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(INPUT));
const FONT=resolvePresentationFont();
const C={paper:'#FFFFFF',navy:'#112B36',ink:'#16313A',muted:'#53666B',teal:'#087F83',coral:'#B95435',line:'#D3DDD9',white:'#FFFFFF',light:'#9DD4CD'};
function text(s,content,x,y,w,h,size=28,color=C.ink,bold=false,name=''){
 const sh=s.shapes.add({geometry:'textbox',name:name||content.slice(0,28),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=content;sh.text.style={typeface:FONT,fontSize:size,color,bold,insets:0,autoFit:'none',wrap:'square',verticalAlignment:'top'};return sh;
}
function newSlide(title,notes){const s=p.slides.add();s.background.fill=C.paper;text(s,title,64,52,1140,95,46,C.ink,true,'title');text(s,'00',1174,664,42,26,20,C.muted,false,'page');s.speakerNotes.text=notes;return s;}
function node(s,label,x,y,w=76,h=54,fill=C.teal){const n=s.shapes.add({geometry:'ellipse',name:'SCM node '+label,position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});n.text=label;n.text.style={typeface:FONT,fontSize:27,color:C.white,bold:true,alignment:'center',verticalAlignment:'middle',insets:0};return n;}
function edge(s,a,b,fromSide='right',toSide='left',color=C.teal,kind='straight',width=5){return s.shapes.connect(a,b,{fromSide,toSide,kind,line:{fill:color,width},tail:{type:'triangle',width:'lg',length:'lg'}});}
function box(s,label,x,y,w,h,color=C.teal,size=25){const q=s.shapes.add({geometry:'rect',name:label,position:{left:x,top:y,width:w,height:h},fill:C.paper,line:{fill:color,width:2}});q.text=label;q.text.style={typeface:FONT,fontSize:size,color,bold:true,alignment:'center',verticalAlignment:'middle',insets:12};return q;}
function table(s,values,x,y,w,h,widths,size=24){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({fill:C.line,width:1,style:'solid'});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const z=t.getCell(r,c);z.fill=r===0?C.navy:(r%2===1?'#EDF3EF':C.paper);z.text.style={typeface:FONT,fontSize:size,color:r===0?C.white:C.ink,bold:r===0,insets:8};}return t;}
function rule(s,x,y,w,h){return s.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:h},line:{fill:C.line,width:2}});}
const s=p.slides.items[3];
for(const sh of [...s.shapes.items])sh.delete();
for(const t of [...s.tables.items])s.tables.deleteById(t.id);
text(s,'Local mechanism tables vs one joint table',64,52,1140,95,44,C.ink,true,'title');
text(s,'Same system: 10 inputs, 5 settings each, 9 mechanisms with two parents',64,148,1150,40,27,C.muted);
text(s,'SCM LOCAL TABLES',64,204,530,40,28,C.teal,true);
text(s,'SCM-FREE JOINT TABLE',687,204,540,40,28,C.coral,true);
rule(s,630,204,0,416);
const a=node(s,'A',104,259,84,64),b=node(s,'B',468,259,84,64),m=node(s,'M',282,341,92,68);
edge(s,a,m,'bottom','left',C.teal,'straight',5);
edge(s,b,m,'bottom','right',C.teal,'straight',5);
text(s,'All input settings together\ndetermine the final output Y',687,282,516,83,29,C.muted);
const tl=table(s,[['A','B','M = f(A, B)'],['0','0','m₁'],['0','1','m₂'],['…','…','…'],['4','4','m₂₅']],64,436,535,175,[110,110,315],22);
const tr=table(s,[['X₁','X₂','…','X₁₀','Y'],['0','0','…','0','y₁'],['0','0','…','1','y₂'],['…','…','…','…','…'],['4','4','…','4','yₙ']],687,436,524,175,[93,93,74,116,148],22);
for(const t of [tl,tr])for(let r=0;r<t.rows.length;r++)for(let c=0;c<t.columns.items.length;c++){const cell=t.getCell(r,c);cell.fill=r===0?'#F0F6F6':C.white;cell.text.style={color:r===0?C.teal:C.ink,bold:r===0};}
text(s,'25 rows per mechanism × 9 = 225',64,626,547,38,27,C.teal,true);
text(s,'5¹⁰ = 9,765,625 joint rows',687,626,530,38,27,C.coral,true);
text(s,'Discrete illustration. Local maps compose through the graph. ACE fits functions instead of literal tables.',64,674,1090,30,18,C.muted);
text(s,'04',1174,664,42,26,20,C.muted,false,'page');
function clear(s){for(const q of [...s.shapes.items])q.delete();for(const t of [...s.tables.items])s.tables.deleteById(t.id);}
function heading(s,title,page){text(s,title,64,52,1140,95,44,C.ink,true,'title');text(s,String(page).padStart(2,'0'),1174,664,42,26,20,C.muted,false,'page');}
function round(s,label,x,y,d,color=C.teal,size=25){const n=node(s,'',x,y,d,d,color);const t=text(s,label,x+8,y+30,d-16,d-60,size,C.white,true);t.text.style={alignment:'center',verticalAlignment:'middle'};return n;}
function segment(s,x,y,w,h,arrowAtStart=false){return s.shapes.add({geometry:'connector',kind:'straight',position:{left:x,top:y,width:w,height:h},line:{fill:C.teal,width:5},head:{type:arrowAtStart?'triangle':'none',width:'lg',length:'lg'},tail:{type:'none'}});}
const loop=p.slides.items[5];clear(loop);heading(loop,'The ACE intervention loop',6);
text(loop,'How the LLM policy learns',64,149,456,37,27,C.coral,true);
text(loop,'1. Start with a pretrained LLM\n2. Learn teacher intervention commands\n3. Prefer higher-scoring candidates (DPO)',64,194,465,120,23,C.ink);
const llm=round(loop,'LLM\npolicy',562,159,146,C.coral,29);
text(loop,'Input: SCM state + history\nOutput: candidate interventions',787,168,428,85,26,C.ink);
text(loop,'Separate LLM implementation',787,270,424,35,23,C.coral,true);
const scm=round(loop,'SCM\nPredict\nuncertainty',130,370,180,C.teal,28);
const selector=round(loop,'Select an\nintervention',545,370,180,C.teal,28);
const environment=round(loop,'Environment\nApply +\nmeasure',960,370,180,C.teal,26);
edge(loop,llm,selector,'bottom','top',C.coral,'straight',5);
text(loop,'Proposed interface',653,324,230,32,21,C.coral);
edge(loop,scm,selector,'right','left',C.teal,'straight',5);
edge(loop,selector,environment,'right','left',C.teal,'straight',5);
text(loop,'Compare allowed\ninterventions',332,394,207,58,22,C.muted);
text(loop,'Chosen\nintervention',756,394,196,58,22,C.muted);
segment(loop,1050,550,0,64);segment(loop,220,614,830,0);segment(loop,220,550,0,64,true);
text(loop,'Measured response updates the natural mechanisms',342,574,672,34,24,C.teal,true);
text(loop,'Allowed interventions: drive X or speed M at −1, 0, +1. Stop when the budget ends.',64,640,1110,34,22,C.ink);
text(loop,'The plotted results use direct SCM scoring (PEV), without the LLM policy.',64,680,1060,26,18,C.muted);
loop.speakerNotes.text=`Implemented paths are distinguished. The lower SCM uncertainty/intervention/environment loop summarizes the measured PEV acquisition procedure. The upper LLM policy is a separate implemented historical path, not the source of the plotted PEV gains and not demonstrated superior by this slide. ace_experiments.py:HuggingFacePolicy loads a pretrained causal language model; supervised_pretrain_llm teaches teacher-generated legal intervention commands using graph/node losses; dpo_loss_llm increases preference for the winner over loser relative to a reference policy. Candidate scores include lookahead improvement and configured bonuses/scaffolding. Historical default lookahead can query the environment for candidate scoring; all such queries must be charged. --lookahead_on_student instead simulates from current learned mechanisms. This conceptual connection does not assert that the historical DPO learner used PEV scoring, that the two paths have been experimentally integrated, or that LLM training guarantees good intervention selection. The diagram proposes the interface between the LLM candidate channel and an SCM scorer while showing the numerical loop separately. The recent language mechanism-proposal screen is a different interface and had six invalid proposals; it does not qualify the LLM policy. Inputs to the policy include the SCM state, errors/history and legal command syntax. No new training was run.\nSources\n/Users/pat/code/ACE/ace_experiments.py (HuggingFacePolicy, supervised_pretrain_llm, dpo_loss_llm and winner/loser updates)\n/Users/pat/code/ACE/scripts/research/persistent_scm.py:campaign\n/Users/pat/code/ACE/baselines.py:PropagatedVariancePolicy\n/Users/pat/code/ACE/docs/development/guidance/ace_foundation_cycle_closeout_2026-10-09.md`;
// Plain-language takeaways precede the exact worked arithmetic.
for(const sh of p.slides.items[7].shapes.items){
 if(sh.name==='title')sh.text='One intervention can inform two mechanisms';
 if(String(sh.text).startsWith('Selected prediction:')){sh.text='Choose drive X = −1. Forecast: M̂ = −2, Ŷ = −6';sh.text.style={fontSize:31};}
}
for(const sh of p.slides.items[8].shapes.items){
 if(sh.name==='title')sh.text='One measured response improves both predictions';
 if(String(sh.text)==='After one illustrative gradient update')sh.text='Learn drive → speed and measured speed → pressure';
 if(String(sh.text).startsWith('Illustrative linear update'))sh.text='Both mechanisms stayed natural, so the same intervention response teaches both.';
}
// Use intervention terminology consistently throughout visible content.
for(const slide of p.slides.items)for(const sh of slide.shapes.items){const before=String(sh.text);let after=before.replace(/Actions:/g,'Interventions:').replace(/First selected action:/g,'First selected intervention:').replace(/first action in the list/g,'first intervention in the list').replace(/tied actions/g,'tied interventions').replace('Two experiments on the same compressor','Two interventions on the same compressor');if(after!==before)sh.text=after;}
const last=p.slides.items[11];clear(last);for(const c of [...last.charts.items])last.charts.deleteById(c.id);
heading(last,'64% fewer batches in the measured comparison',12);
text(last,'Two measured intervention policies alongside an analytic grid-coverage example',64,151,1150,44,26,C.muted);
text(last,'Recorded: matched prediction error',135,210,687,34,25,C.teal,true);
text(last,'Analytic: 95% grid coverage',879,210,333,67,24,C.coral,true);
const comparison=last.charts.add('bar',{
 position:{left:64,top:271,width:1140,height:309},
 categories:['SCM-scored ACE','Random + SCM','SCM-free grid'],
 series:[{name:'Query count',values:[500,1400,29255197],fill:C.teal,points:[{idx:0,fill:C.teal},{idx:1,fill:'#8BA5AD'},{idx:2,fill:C.coral}]}],
 hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:160,varyColors:false},
 dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'#,##0',textStyle:{fontSize:24,fill:C.ink,bold:true}},
 xAxis:{textStyle:{fontSize:22,fill:C.ink},line:{fill:C.line,width:1}},
 yAxis:{min:0,max:32000000,majorUnit:10000000,numberFormatCode:'0,,"m"',textStyle:{fontSize:19,fill:C.muted},majorGridlines:{fill:C.line,width:1}},
 chartFill:C.white,chartLine:{fill:'none',width:0},plotAreaFill:C.white,plotAreaLine:{fill:'none',width:0}
});
applyPresentationChartFont(comparison,{fontFamily:FONT});
text(last,'Counts of queries (linear scale)',64,251,648,30,20,C.muted);
text(last,'500 vs 1,400 intervention responses',64,603,767,44,30,C.teal,true);
text(last,'29.26 million grid draws',879,603,340,64,27,C.coral,true);
text(last,'First two: 10 vs 28 batches at MSE ≤ 0.04390. Third: a different task, not a measured ACE speedup.',64,671,1097,38,20,C.muted);
last.speakerNotes.text=`Three bars use counts of individual queries: measured PEV500 intervention responses at first mean-curve crossing versus random1400, and analytic29255197uniform random joint-grid draws for95%expected coverage. The third bar has a DIFFERENT endpoint/system/distribution, is not an empirical third arm, and cannot support a ratio of ACE savings relative to grid search. The shared linear axis deliberately leaves the measured bars nearly invisible; exact counts remain in data labels and the measured comparison is restated below. Measured values are retrospective first crossings of group mean curves across20synthetic30-node systems, MSEthreshold0.043898007078491626,50interventionresponses perbatch. Including observations, prefixes620and1760; both full campaigns ran2000totalresponses. The random arm is NonLeafRandomPolicy: uniform choice among graph-eligible nonleaf targets and random.uniform values, with the same SCM ensemble learner as PEV. It is not a deterministic direct policy; PEV is the direct SCM-scored policy. Third bar: ten five-valued inputs,9765625jointentries,uniform independent joint draws withreplacement. One queried configuration yields one deterministic lookup response in this analytic illustration. Other associative learners need not enumerate a grid. No new experiment has been run.\nSources\n/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/recorded_curves_2026-10-09.json\n/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/SCM_free_grid_comparison.json\n/Users/pat/code/ACE/scripts/research/persistent_scm.py:campaign\n/Users/pat/code/ACE/baselines.py:RandomPolicy,NonLeafRandomPolicy`;
await fs.mkdir(DIR+'/previews',{recursive:true});
await (await PresentationFile.exportPptx(p)).save(DIR+'/draft.pptx');
for(const [i,slide] of p.slides.items.entries()){const img=await p.export({slide,format:'png',scale:1});await fs.writeFile(DIR+'/previews/slide-'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await img.arrayBuffer()));}
await fs.writeFile(DIR+'/speaker_notes.md',p.slides.items.map((s,i)=>`# Slide ${i+1}\n\n${s.speakerNotes.text}`).join('\n\n'));
await fs.writeFile(DIR+'/edited-inspect.ndjson',(await p.inspect({kind:'slide,textbox,chart',maxChars:150000})).ndjson);
execFileSync('/Users/pat/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',[DIR+'/restore.py'],{stdio:'inherit'});
