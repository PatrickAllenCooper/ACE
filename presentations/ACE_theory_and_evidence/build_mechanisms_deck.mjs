import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
import { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } from '/Users/pat/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/artifact_tool_utils.mjs';
const DIR='/Users/pat/code/ACE/.codex-artifacts/ace-mechanism-deck/build';
const OUT='/Users/pat/code/ACE/presentations/ACE_theory_and_evidence';
const SKILL='/Users/pat/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const FONT=resolvePresentationFont();
const C={paper:'#FAF8F3',navy:'#112B36',ink:'#16313A',muted:'#53666B',teal:'#087F83',coral:'#B95435',line:'#D3DDD9',white:'#FFFFFF',light:'#9DD4CD'};
const p=Presentation.create({slideSize:{width:1280,height:720}});p.theme.defaultFont=FONT;
function text(s,content,x,y,w,h,size=28,color=C.ink,bold=false,name=''){
 const sh=s.shapes.add({geometry:'textbox',name:name||content.slice(0,28),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=content;sh.text.style={typeface:FONT,fontSize:size,color,bold,insets:0,autoFit:'none',wrap:'square',verticalAlignment:'top'};return sh;
}
function slide(title,notes,background=C.paper){const s=p.slides.add();s.background.fill=background;const n=p.slides.items.length;text(s,title,64,52,1140,95,46,background===C.navy?C.white:C.ink,true,'title');text(s,String(n).padStart(2,'0'),1174,664,42,26,20,background===C.navy?C.light:C.muted,false,'page');s.speakerNotes.text=notes;return s;}
function notes(body,sources){return body+'\n\nSources\n'+sources.join('\n');}
const SRC={theory:'/Users/pat/code/ACE/docs/development/guidance/ace_theoretical_ideation_2026-10-08.md',public:'/Users/pat/code/ACE/docs/ACE_evidence_and_public_claims_2026-10-08.md',claims:'/Users/pat/code/ACE/paper/aistats_ace_2027/claim_index.json',paper:'/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex',shift:'/Users/pat/code/ACE/results/research_pev_shift30_mean_confirmation/README.md',protocol:'/Users/pat/code/ACE/docs/development/guidance/protocol_pev_shift30_mean_confirmation.json',erratum:'/Users/pat/code/ACE/docs/development/guidance/erratum_shift_graph_provenance_2026-09-27.md'};
function chart(s,categories,values,pos,{max,format='0.000',points,labels}={}){
 const ch=s.charts.add('bar',{position:pos,categories,series:[{name:'Result',values,fill:C.teal,points:(points??values.map(()=>C.teal)).map((fill,idx)=>({idx,fill})),valuesFormatCode:format,...(labels?{dataLabelOverrides:labels.map((txt,idx)=>({idx,text:txt,showValue:false}))}:{})}],barOptions:{direction:'column',grouping:'clustered',gapWidth:90},hasLegend:false,chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},xAxis:{visible:true,textStyle:{fontSize:23,fill:C.ink},line:{fill:C.line,width:1},majorGridlines:null},yAxis:{visible:true,min:0,max,numberFormatCode:format,textStyle:{fontSize:20,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:25,bold:true,fill:C.ink}}});applyPresentationChartFont(ch,{fontFamily:FONT});return ch;
}
const DEMO='/Users/pat/code/ACE/presentations/ACE_theory_and_evidence/mechanistic_demonstration_2026-10-09.json';
const d=JSON.parse(await fs.readFile(DEMO,'utf8'));
const EVIDENCE=OUT+'/recorded_curves_2026-10-09.json';
const e=JSON.parse(await fs.readFile(EVIDENCE,'utf8'));
function node(s,label,x,y,w=112,h=82,fill=C.teal,color=C.white){let n=s.shapes.add({geometry:'ellipse',name:'SCM node '+label,position:{left:x,top:y,width:w,height:h},fill,line:{fill:C.line,width:1}});n.text=label;n.text.style={typeface:FONT,fontSize:w<70?18:(label.length>2?27:40),color,bold:true,insets:0,alignment:'center',verticalAlignment:'middle'};return n;}
function edge(s,a,b,color=C.teal,width=6){return s.shapes.connect(a,b,{kind:'straight',fromSide:'right',toSide:'left',line:{fill:color,width},tail:{type:'triangle',width:'lg',length:'lg'}});}
function chain(s,y,clamp=false){const a=node(s,'X',100,y),b=node(s,clamp?'M = 1':'M',490,y),c=node(s,'Y',880,y);if(!clamp)edge(s,a,b);edge(s,b,c);return[a,b,c];}
function curve(s,rows,pos,max,legend=true){const ch=s.charts.add('line',{position:pos,categories:rows.map(i=>String(i+1)),series:[{name:'Random',smooth:false,values:rows.map(i=>Number(e.curves.nonleaf_random_ens[i].mean_mse.toPrecision(12))),line:{fill:C.coral,width:3}},{name:'ACE / PEV',smooth:false,values:rows.map(i=>Number(e.curves.pev[i].mean_mse.toPrecision(12))),line:{fill:C.teal,width:3}}],hasLegend:legend,legend:{position:'bottom',textStyle:{fontSize:19}},chartFill:'none',plotAreaFill:'none',xAxis:{visible:true,textStyle:{fontSize:16},title:'Intervention batches (50 responses each)'},yAxis:{visible:true,min:0,max,numberFormatCode:max>1?'0.0':'0.00',textStyle:{fontSize:17},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:false}});applyPresentationChartFont(ch,{fontFamily:FONT});return ch;}
const mechanismNotes='Implementation: baselines.py, EnsembleStudentSCM / EnsembleLearner / PropagatedVariancePolicy, and scripts/research/persistent_scm.py. Historical PEV uses known graphs, independently initialized neural heads, model-only candidate contexts and covariance-based integrated variance reduction summed over descendant mechanisms. It does not identify the graph or optimize the proposed terminal-risk objective. Noise proxy is residual error, not a certified aleatoric estimate.';
function table(s,values,x,y,w,h,widths,size=27){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({fill:C.line,width:1,style:'solid'});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){let z=t.getCell(r,c);z.fill=r===0?C.navy:(r%2===1?'#EDF3EF':C.paper);z.text.style={typeface:FONT,fontSize:r===0?size-2:size,color:r===0?C.white:C.ink,bold:r===0,insets:10};}return t;}
function foot(s,t){text(s,t,64,620,1120,68,22,C.muted);}
{
const s=slide('',notes('ACE selects interventions using an assumed causal graph and uncertainty about local mechanisms. Compressor visual is a newly generated conceptual engineering illustration, not the apparatus used in the reported synthetic experiments. Slides1–4 introduce the theory; slides5–7 give precise illustrative inputs, scores and outputs; slide8 shows an archived graph; slides9–10 show historical evidence. '+mechanismNotes,[SRC.theory,SRC.public,DEMO] ),C.navy);
s.images.add({blob:new Uint8Array(await fs.readFile(OUT+'/ACE_compressor_concept.png')),contentType:'image/png',alt:'Conceptual cutaway compressor with teal airflow',fit:'cover',position:{left:0,top:0,width:1280,height:720}});
text(s,'ACE',62,60,550,130,100,C.white,true);
text(s,'Learn what to\nchange next.',66,223,530,170,56,C.white,true);
text(s,'Causal models for\nsmarter experiments',67,455,520,105,31,C.light);
text(s,'STRUCTURE  →  INTERVENE  →  LEARN',67,616,880,40,22,C.light,true);
text(s,'Compressor concept illustration',875,664,350,30,18,C.light);

}
{
const s=slide('The causal model inside ACE',notes('SCM definition: Xi=fi(Pa_i,Ui). Edges specify direct causes in the assumed graph, f describes the mechanism, U describes external disturbances. The toy is a fully observed, deterministic chain with dimensionless values. Its true rules are M=2.5X and Y=3M, with U=0. These truth equations explain the example to the audience; the policy receives only its candidate models and allowed actions, not oracle truth. Known graph does not imply known mechanisms. General SCMs may have dependent disturbances; identification and fitting require explicit assumptions.',[SRC.theory,DEMO]));
text(s,'Xᵢ = fᵢ(Paᵢ, Uᵢ)',64,171,1100,92,66,C.teal,true);
text(s,'Parents Paᵢ supply inputs. Uᵢ represents outside influences.',64,289,1100,78,31);
chain(s,400);text(s,'M = 2.5X',325,503,300,50,30,C.teal,true);text(s,'Y = 3M',735,503,300,50,30,C.teal,true);
foot(s,'Worked example: known graph, unknown slopes, zero disturbances and dimensionless values.');
}
{
const s=slide('Choose where to intervene',notes('do(M=m) fixes M independently of its usual parents and replaces M=fM(X,UM) with M=m. X→M is removed for this experiment, while M→Y remains. Row X=x,M=m,Y=y supplies a natural Y label with measured parent m, but is not a natural M=fM(X) label. Under do(X=x), both M and Y remain naturally generated and eligible under the stated retention/noise assumptions. The implementation excludes the intervened head and uses observed parents for eligible training. Candidate simulation propagates ensemble-mean parents. Historical acquisition-study evaluation instead predicts each mechanism from observed parents; the toy composed target forecast and delivery studies have different evaluation semantics.',[SRC.theory,DEMO]));
text(s,'Ordinary system',64,169,550,50,31,C.muted);
chain(s,231);
text(s,'Experiment: do(M = 1)',64,389,650,61,36,C.teal,true);
chain(s,459,true);text(s,'Incoming edge removed',178,554,490,48,26,C.coral);text(s,'Y still responds to M',761,554,424,48,26,C.teal);
foot(s,'One paid response can train several natural mechanisms. It is still one paid response.');
}
{
const s=slide('Exhaustive grids multiply the work',notes('Analytic representation-size illustration, not an ACE benchmark. Define a deterministic discrete SCM with10independently controllable five-valued input nodes and9two-parent mechanism nodes arranged as a binary reduction tree. Every endogenous output also has5values. A naive associative lookup table for the final response over the full joint input grid requires5^10=9,765,625entries. With the graph supplied, nine local mechanism tables require9*5^2=225entries, a43,402.78fold representation-count difference. This is not a measured intervention-count reduction. Learning local tables assumes each required parent configuration is accessible with the child mechanism intact and allvariablesobserved; without internal interventions, reachability can fail. Sharedparentsettings may reveal several labels. Noncausal regressors can exploit smoothness/sparsity and need not enumerate a grid; the comparator is specifically exhaustive lookup, not allassociativelearning. No empirical superiority over this new baseline has been measured. Current PEV scores candidates by descendant ensemble covariance reduction; details retained in the worked calculation.',[SRC.theory,DEMO]));
text(s,'10 inputs · 5 settings each · a deliberately naive joint-table baseline',64,152,1140,52,28,C.muted);
text(s,'SCM-FREE JOINT TABLE',64,236,710,40,24,C.coral,true);
text(s,'9.77 million',64,289,730,105,74,C.coral,true);
text(s,'configurations for exhaustive coverage',68,394,700,42,28);
text(s,'29.26 million random draws',67,474,760,57,36,C.coral,true);
text(s,'for 95% expected joint-grid coverage¹',68,540,745,42,27);
text(s,'SCM LOCAL TABLES',875,236,345,40,24,C.teal,true);
text(s,'225',870,310,340,122,92,C.teal,true);
text(s,'local entries',876,444,330,45,31,C.teal,true);
text(s,'9 mechanisms\n25 settings each',876,506,335,90,27,C.muted);
foot(s,'Analytic coverage / representation example, not a measured ACE speedup. ¹ Uniform draws with replacement.');
s.speakerNotes.text += '\nUniform random sampling baseline: independently draw one of N=5^10 joint configurations, with replacement and no SCM. Expected distinct fraction after n draws is 1−(1−1/N)^n. The smallest n for95% expected coverage is29,255,197. This is expected grid coverage, not a95% probability of complete coverage or a prediction-error target. The exhaustive/local entry ratio is43,402.78; do not present the random/local ratio as experimental intervention savings. Local access and determinism assumptions stated above remain required.';

}
{
const s=slide('Exact inputs to the three-node demonstration',notes('Illustrative configuration only. Three linear ensemble members: M slopes1,2,3, Y slopes2,3,4; ensemble means2X and3M. Candidates ordered X−1,X0,X1,M−1,M0,M1. Reference parent values−1,0,1. Population covariance denominator3, residual noise proxy1/20, deterministic mean-propagated parent contexts, no jitter or epsilon exploration. Preintervention observation history empty for this hand-constructed initial state. One response remaining. Oracle truth hidden from score: M=2.5X,Y=3M. Production historical confirmation uses neural heads and different grids/batches. All numbers are generated by explain_ace_mechanics.py with exact rational arithmetic.',[DEMO,'/Users/pat/code/ACE/scripts/research/explain_ace_mechanics.py']));
table(s,[['Mechanism','Member slopes','Mean prediction'],['M from X','1, 2, 3','M̂ = 2X'],['Y from M','2, 3, 4','Ŷ = 3M']],64,163,1136,216,[345,335,456],29);
text(s,'Actions: clamp X or M to −1, 0 or +1',64,428,1100,60,33,C.teal,true);
text(s,'References: −1, 0, +1     Noise proxy: 0.05\nBudget: one response     Ties: first action in the list',64,510,1100,94,29);
foot(s,'Simplified linear ensemble · six candidate actions · one response budget');
}
{
const s=slide('The score selects do(X = −1)',notes('Exact score for M context±1 is160/387≈0.413436693. Y context±2 is640/1467≈0.436264485. Their sum is≈0.849701178. Under do(M=±1) the clamped M mechanism contributes zero, leaving only Y≈0.413436693. Zero contexts give zero covariance and zero score. X+1 tiesX−1; first-maximum tie rule choosesX−1. At that action the ensemble mean predictsM−2,Y−6. No oracle outcome is queried during scoring. The Y slope ensemble is evaluated at the simulated mean parent; it is not a member-paired composed posterior.',[DEMO]));
const vals=[['Intervention','M contribution','Y contribution','Total']];for(const r of d.candidates)vals.push([`do(${r.node} = ${r.value.decimal})`,r.score_M.decimal.toFixed(6),r.score_Y.decimal.toFixed(6),r.score.decimal.toFixed(6)]);
table(s,vals,64,163,1136,378,[340,266,266,264],25);
text(s,'Selected prediction: M̂ = −2,  Ŷ = −6',64,559,1100,61,35,C.teal,true);
foot(s,'Score contribution: mean Cov² / (Variance + noise proxy) · tied actions follow the declared order');
}
{
const s=slide('The response updates eligible mechanisms',notes('Illustrative oracle row isX−1,M−2.5,Y−7.5. Eligible pairs M:(−1,−2.5),Y:(−2.5,−7.5); clamped X supplies no natural root-distribution update. For transparent arithmetic use one gradient step on half squared error, with etaM1/2 and etaY2/25. M slopes1,2,3 become1.75,2.25,2.75. Y slopes2,3,4 become2.5,3,3.5. At X−1, updated mean predictsM−2.25 andY−6.75. Signed final error drops from1.5 to.75 on this same illustrative row. This is not held-out improvement or production training: production uses neural heads, Adam and member-specific masks. Response budget becomes0. Do not associate this made-up row with the real archived trace.',[DEMO]));
text(s,'Observed row: X = −1, M = −2.5, Y = −7.5',64,162,1130,71,37,C.teal,true);
table(s,[['Natural mechanism','Training input','Training label'],['M','Measured X = −1','M = −2.5'],['Y','Measured M = −2.5','Y = −7.5']],64,254,1136,204,[350,396,390],28);
text(s,'After one illustrative gradient update',64,491,1100,53,29,C.muted);
text(s,'M̂ = −2.25       Ŷ = −6.75       Budget left: 0',64,551,1100,57,35,C.teal,true);
foot(s,'Illustrative linear update · both natural mechanisms learn from the same paid response');
}
{
const s=slide('A real scenario: 30 linked mechanisms',notes('Exact archived DAG for smallest seed5000, not chosen for favorable performance. Edges point parent to child. HighlightX7 first chosen intervention and its descendant edges. FirstactiondoX7=3.857086181640625 buys50responses. Eachrow supplieseligible natural-mechanism labels using observedparents. Fullcampaign32interventionbatches1600responses plus400observationalresponses. All20DAGs in thecomparison shareonefive-layergenerator, not20independentgraphfamilies. Rawrows/candidatescores notretained for thisaction.',[d.recorded_trace.path+'/system.json',d.recorded_trace.path+'/trajectory.csv',SRC.shift,EVIDENCE]));
const layers=[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25],[26,27,28,29,30]];
const nodes={};layers.forEach((arr,l)=>arr.forEach((i,j)=>{let y=190+(j+0.5)*(370/arr.length);nodes['X'+i]=node(s,'X'+i,90+l*205,y,60,32,i===7?C.coral:C.teal);}));
const desc=new Set(['X7']);for(let k=0;k<30;k++)for(const [child,parents]of Object.entries(e.graph))if(parents.some(p=>desc.has(p)))desc.add(child);
for(const [child,parents] of Object.entries(e.graph))for(const pa of parents)edge(s,nodes[pa],nodes[child],desc.has(pa)?C.coral:'#8BA8AE',desc.has(pa)?4.5:3);
text(s,'First selected action: do(X7 = 3.8571)  →  50 responses',64,568,1130,48,31,C.teal,true);
foot(s,'Arrows point from cause to effect. Coral follows the X7 intervention downstream · archived seed 5000.');
}
{
const s=slide('65% lower final prediction error',notes('All20shift30systemsseeds5000–5019. Arithmeticmean noise-free feasible nonroot mechanism MSE, measured-parent prediction. Full32checkpoints shown atleft; samecurves4–32zoomatright. FinalACE.0154302049,random.0438980071,64.85%lowerratioofmeans,19/20pairedwins. Secondarycontrastp.00750. Prespecifiedcoveragecontrastfails p.05108 andnaivevariancescoringunresolved p.62176. Earlyrandomadvantage remains visible. No Bresults,confidenceband,newfit ornewhypothesis.',[SRC.shift,EVIDENCE]));
text(s,'19 of 20 systems won · equal budgets · lower error is better',64,155,1120,45,28,C.muted);
text(s,'Full recorded curve',64,219,385,36,25,C.ink,true);
text(s,'Detail: batches 4–32',483,219,620,36,25,C.ink,true);
curve(s,Array.from({length:32},(_,i)=>i),{left:64,top:265,width:360,height:277},6.5,false);
curve(s,Array.from({length:29},(_,i)=>i+3),{left:464,top:265,width:740,height:277},0.13,true);
text(s,'Final MSE: 0.01543 ACE vs 0.04390 random  |  19 / 20 wins',64,561,1130,50,31,C.teal,true);
foot(s,'20 synthetic systems · 2,000 responses per arm · random action selection with the same SCM learner');
}
{
const s=slide('64% fewer intervention batches',notes('New descriptivepost-hocdisplayonly. TargetdefinedasrandomfinalgroupmeanMSE .043898007078491626. Firstobservedgroupmeancrossing,nointerpolation:ACEbatch10,500interventionresponses+120obs=620total,mean.04007927;randombatch28,1400interventionresponses+360obs=1760total,mean.04216014. 1−10/28=64.2857%fewerbatches/interventionresponses;1−620/1760=64.7727%fewertotalresponses. Randomcrossesbackabove atbatch31. Bothoriginalcampaignsactuallyran32batches2000totalresponses;thesearedescriptiveprefixcomparisons,notactualsavedcompute/measurements,norvalidatedprospectivestoppingrule. Groupmeanfirstcrossingnotmeanofindividualfirstcrossings. Noone-systemorpopulationguarantee. Originalstudiesshowfixedbudgetpredictionadvantage;prospectivesample-efficiencyconfirmationremainsfuture.',[EVIDENCE,SRC.shift]));
text(s,'First recorded crossing of MSE ≤ 0.04390',64,151,1130,48,31,C.muted);
chart(s,['Random','ACE / PEV'],[28,10],{left:70,top:225,width:605,height:344},{max:32,format:'0',points:[C.coral,C.teal]});
text(s,'Intervention batches · 50 responses each',74,579,622,38,24,C.muted);
text(s,'64.3% fewer',744,255,460,83,53,C.teal,true);
text(s,'1,400 → 500\nintervention responses',746,356,444,104,31,C.ink,true);
text(s,'Including observations:\n1,760 → 620 total responses',746,488,443,90,27,C.muted);
foot(s,'Retrospective comparison of mean learning curves across 20 synthetic systems · target: random’s final mean error');
}
await fs.mkdir(DIR+'/previews',{recursive:true});
for(const [i,s] of p.slides.items.entries()){const b=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(DIR+'/previews/slide-'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await b.arrayBuffer()));}
await (await PresentationFile.exportPptx(p)).save(DIR+'/draft.pptx');
await fs.writeFile(DIR+'/speaker_notes.md',p.slides.items.map((s,i)=>`# Slide ${i+1}\n\n${s.speakerNotes.text}`).join('\n\n'));
await fs.writeFile(DIR+'/montage.webp',new Uint8Array(await (await p.export({format:'webp',montage:true})).arrayBuffer()));
if(process.env.FINALIZE==='1'){
const suffix=process.env.DECK_SUFFIX||'v6';
const r=await finalizePresentation({explicitTotalSlideCount:10,workspaceDir:'/Users/pat/code/ACE',candidatePath:DIR+'/draft.pptx',finalPath:OUT+'/ACE_mechanisms_and_evidence_'+suffix+'.pptx',pythonExecutable:'/Users/pat/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...([5,6,7].flatMap(n=>['--require-native-table-slide',String(n)]))],requiredNativeTableOwnerSlides:[5,6,7],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:DIR+'/validation-'+suffix+'.json'});console.log(JSON.stringify(r));
}
console.log(JSON.stringify({slides:p.slides.items.length,font:FONT}));
