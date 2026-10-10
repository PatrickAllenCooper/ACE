import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import {resolvePresentationFont,finalizePresentation} from '/Users/pat/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/artifact_tool_utils.mjs';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
const ROOT='/Users/pat/code/ACE',OUT=ROOT+'/presentations/ACE_theory_and_evidence',DIR=ROOT+'/.codex-artifacts/ace-mechanism-deck-v7/build';
const SKILL='/Users/pat/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const INPUT=OUT+'/ACE_mechanisms_and_evidence_v6.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(INPUT));
const FONT=resolvePresentationFont();
const C={paper:'#FAF8F3',navy:'#112B36',ink:'#16313A',muted:'#53666B',teal:'#087F83',coral:'#B95435',line:'#D3DDD9',white:'#FFFFFF',light:'#9DD4CD'};
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
const local=newSlide('Local mechanism tables vs one joint table',`Analytic representation example, not measured sample efficiency. Same discrete setup as the next slide: ten independent five-valued inputs, nine two-parent deterministic five-valued mechanisms in a binary reduction tree, all relevant variables observed. Each local table maps the two direct parent values to its child. Nine local maps compose to predict the final Y. Each local map has5^2=25input rows,225total. Joint lookup has one row per complete ten-input vector,5^10=9,765,625rows. Displayed outputs m1... and y1... are placeholders, not experimental observations. Learning all local entries requires every parent setting accessible with the child mechanism intact. Intermediate variables must be observed. A noncausal regressor need not enumerate this grid. Actual ACE learners fit functions (neural mechanisms in the historical results), not literal local lookup tables.\nSources\n${OUT}/SCM_free_grid_comparison.json\n${ROOT}/docs/development/guidance/ace_theoretical_ideation_2026-10-08.md`);
text(local,'Same system: 10 inputs, 5 settings each, 9 mechanisms with two parents',64,148,1150,40,27,C.muted);
text(local,'SCM LOCAL TABLES',64,211,530,40,28,C.teal,true);
text(local,'SCM-FREE JOINT TABLE',687,211,540,40,28,C.coral,true);
rule(local,630,207,0,392);
const a=node(local,'A',89,260,65,50),b=node(local,'B',89,323,65,50),m=node(local,'M',302,296,72,53);edge(local,a,m);edge(local,b,m);
text(local,'Only the direct\nparents of M',413,296,180,80,24,C.muted);
table(local,[['A','B','M = f(A, B)'],['0','0','m₁'],['0','1','m₂'],['…','…','…'],['4','4','m₂₅']],64,380,535,175,[110,110,315],22);
text(local,'25 rows per mechanism × 9 = 225',64,603,547,41,28,C.teal,true);
text(local,'All input settings together determine Y',687,279,525,62,27,C.muted);
table(local,[['X₁','X₂','…','X₁₀','Y'],['0','0','…','0','y₁'],['0','0','…','1','y₂'],['…','…','…','…','…'],['4','4','…','4','yₙ']],687,367,524,217,[93,93,74,116,148],23);
text(local,'5¹⁰ = 9,765,625 joint rows',687,603,530,41,28,C.coral,true);
text(local,'Discrete illustration. Local maps compose through the graph. ACE fits functions instead of literal tables.',64,647,1090,38,20,C.muted);
local.moveTo(3);
const mechanism=newSlide('ACE: models, actions and environmental feedback',`Overall conceptual architecture of the proposed foundation-model extension around the implemented ACE SCM experiment loop. The known graph, legal controls and budget are supplied; this is not graph discovery or unrestricted control of a physical device. The numerical foundation model consumes eligible parent/label examples and proposes alternative local mechanism predictors. Numerical and retained candidates remain available. Candidate validation/selection is required before incorporation; proposed integration has not demonstrated foundation-model acquisition efficiency. Component/retention pilots use fixed histories and show mixed candidate benefits. Foundation-model branch is explicitly marked proposed. Historical random/PEV curves later in this deck use neural SCM ensembles without this pretrained branch.\nSCM candidates simulate allowed interventions and supply forecasts plus ensemble uncertainty to the action selector. The PEV implementation scores covariance-based integrated variance reduction over descendant mechanisms, not a certified risk bound. The six-action toy legal menu is do(X=-1,0,+1) or do(M=-1,0,+1). The frozen tie rule selects do(X=-1) in the following example. The environment returns X=-1,M=-2.5,Y=-7.5 there. Eligible mechanisms use observed parents for fitting; the clamped mechanism is excluded. Candidate scoring is model-only; querying the environment charges a response. Budget exhaustion stops further paid actions. In an apparatus, access is limited to approved controllable settings. No compressor experiment or autonomous safety certification is claimed.\nSources\n${OUT}/mechanistic_demonstration_2026-10-09.json\n${ROOT}/docs/development/guidance/ace_foundation_cycle_closeout_2026-10-09.md\n${ROOT}/docs/development/guidance/ace_foundation_retention_design_2026-10-09.md\n${ROOT}/baselines.py`);
text(mechanism,'Foundation-model branch: proposed integration',64,145,1130,39,26,C.coral,true);
const inputs=box(mechanism,'Inputs\nGraph + eligible data\nControls + budget',64,198,283,105,C.muted,23);
const fm=box(mechanism,'Foundation model\nCandidate mechanisms',456,198,311,105,C.coral,27);
const check=box(mechanism,'Candidate validation\nCompare with numerical\nand retained mechanisms',873,198,338,105,C.coral,24);
edge(mechanism,inputs,fm,'right','left',C.coral);edge(mechanism,fm,check,'right','left',C.coral);
const scm=box(mechanism,'',64,375,311,130,C.teal);
text(mechanism,'SCM candidates',85,386,270,36,28,C.teal,true);
const sx=node(mechanism,'X',86,441,51,39),sm=node(mechanism,'M',188,441,51,39),sy=node(mechanism,'Y',290,441,51,39);edge(mechanism,sx,sm).bringToFront();edge(mechanism,sm,sy).bringToFront();
const select=box(mechanism,'Action selector\nClamp X or M\nto −1, 0 or +1',491,375,277,130,C.teal,26);
const env=box(mechanism,'Environment\nSimulator or apparatus',903,375,308,130,C.teal,27);
edge(mechanism,check,scm,'bottom','top',C.coral,'elbow');
text(mechanism,'validated candidate heads',406,307,426,30,23,C.coral);
edge(mechanism,scm,select);edge(mechanism,select,env);
text(mechanism,'predict +\nuncertainty',380,384,112,62,18,C.muted);
text(mechanism,'do(X = −1)',782,399,120,39,21,C.teal,true);
const row=box(mechanism,'Observed X, M, Y',903,565,308,62,C.teal,25);
const update=box(mechanism,'Update natural mechanisms\nusing measured parent values',64,565,525,62,C.teal,24);
edge(mechanism,env,row,'bottom','top');edge(mechanism,row,update,'left','right');edge(mechanism,update,scm,'top','bottom',C.teal,'elbow');
text(mechanism,'one paid response',640,550,246,36,23,C.muted);
text(mechanism,'Exclude the clamped mechanism. Stop when the budget ends. Measured curves use the neural SCM loop.',64,649,1100,38,20,C.muted);
mechanism.moveTo(5);
// Recompose the original closing slide, retaining its editable empirical chart and notes.
const last=p.slides.items[p.slides.items.length-1];
for(const sh of [...last.shapes.items])if(!['title','page'].includes(sh.name))sh.delete();
const chart=last.charts.items[0];chart.position={left:64,top:238,width:670,height:300};
// Original chart categories and numeric data are preserved.
text(last,'MEASURED INTERVENTION BATCHES',64,158,680,37,25,C.teal,true);
text(last,'First mean-curve crossing of MSE ≤ 0.04390',64,200,710,36,25,C.muted);
text(last,'64.3% fewer',64,556,378,64,45,C.teal,true);
text(last,'1,400 to 500\nintervention responses',430,554,345,76,27,C.ink,true);
text(last,'50 responses / batch. Including observations: 1,760 to 620.',64,639,700,29,20,C.muted);
rule(last,792,158,0,471);
text(last,'SCM-FREE JOINT TABLE',835,158,390,38,25,C.coral,true);
text(last,'10 inputs × 5 settings',835,205,375,34,25,C.muted);
text(last,'9.77 million',835,250,377,73,49,C.coral,true);
text(last,'entries for exhaustive coverage',835,323,378,63,23,C.ink);
table(last,[['Input settings','Y'],['0, 0, …, 0','y₁'],['…','…'],['4, 4, …, 4','yₙ']],835,391,376,140,[273,103],22);
text(last,'29.26 million random draws',835,550,378,72,29,C.coral,true);
text(last,'95% expected grid coverage',835,618,380,34,22,C.muted);
text(last,'Retrospective curve comparison',64,681,720,26,17,C.muted);
text(last,'Analytic illustration, separate units',835,658,332,41,18,C.muted);
last.speakerNotes.text+='\n\nAdded joint-table panel: analytic representation/coverage comparison under the same ten-input five-valued deterministic example as slide5. 9,765,625joint entries and29,255,197uniform random draws with replacement for95%expected coverage. These are not intervention batches at matched error and are not a third measured arm. Separate graphic and labels deliberately avoid a common axis. SCM local tables225entries under full parent-setting access. Original measured two-arm values and all retrospective limitations remain unchanged. Source: '+OUT+'/SCM_free_grid_comparison.json';
// Renumber only page markers on retained slides.
for(const [i,s]of p.slides.items.entries()){for(const sh of s.shapes.items)if(sh.name==='page')sh.text=String(i+1).padStart(2,'0');}
p.slides.items[0].speakerNotes.text+='\nRevision7 inserts the local-versus-joint explanation at4 and the proposed foundation-model/SCM overview at6. The worked example is7–9, archived graph10, empirical learning curves11 and closing comparison12.';
await fs.mkdir(DIR+'/previews',{recursive:true});
for(const [i,s]of p.slides.items.entries()){const b=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(DIR+'/previews/slide-'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await b.arrayBuffer()));}
await fs.writeFile(DIR+'/speaker_notes.md',p.slides.items.map((s,i)=>`# Slide ${i+1}\n\n${s.speakerNotes.text}`).join('\n\n'));
await (await PresentationFile.exportPptx(p)).save(DIR+'/draft.pptx');
await fs.writeFile(DIR+'/montage.webp',new Uint8Array(await(await p.export({format:'webp',montage:true})).arrayBuffer()));
// Import drops embedded workbook relationships. Restore exact original chart/workbook parts.
execFileSync('/Users/pat/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',[OUT+'/restore_chart_sources_v7.py'],{stdio:'inherit'});
if(process.env.FINALIZE==='1')console.log(JSON.stringify(await finalizePresentation({explicitTotalSlideCount:12,workspaceDir:ROOT,candidatePath:DIR+'/draft-preserved.pptx',finalPath:OUT+'/ACE_mechanisms_and_evidence_v7.pptx',pythonExecutable:'/Users/pat/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...([4,7,8,9,12].flatMap(n=>['--require-native-table-slide',String(n)]))],requiredNativeTableOwnerSlides:[4,7,8,9,12],requiredNativeChartOwnerSlides:[11,12],fontPolicy:{basis:'reference',families:[FONT],referencePath:INPUT,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(INPUT)).digest('hex')},verifyArtifactToolImport:true,receiptPath:DIR+'/validation-v7.json'})));
console.log(JSON.stringify({slides:p.slides.items.length,font:FONT,titles:p.slides.items.map(s=>String(s.shapes.items.find(sh=>sh.name==='title')?.text??''))}));
