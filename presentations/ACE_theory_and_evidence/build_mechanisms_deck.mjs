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
function edge(s,a,b,color=C.muted){return s.shapes.connect(a,b,{kind:'straight',fromSide:'right',toSide:'left',line:{fill:color,width:2},tail:{type:'arrow',width:'sm',length:'sm'}});}
function chain(s,y,clamp=false){const a=node(s,'X',100,y),b=node(s,clamp?'M = 1':'M',490,y),c=node(s,'Y',880,y);if(!clamp)edge(s,a,b);edge(s,b,c);return[a,b,c];}
function curve(s,rows,pos,max,legend=true){const ch=s.charts.add('line',{position:pos,categories:rows.map(i=>String(i+1)),series:[{name:'Random',smooth:false,values:rows.map(i=>Number(e.curves.nonleaf_random_ens[i].mean_mse.toPrecision(12))),line:{fill:C.coral,width:3}},{name:'ACE / PEV',smooth:false,values:rows.map(i=>Number(e.curves.pev[i].mean_mse.toPrecision(12))),line:{fill:C.teal,width:3}}],hasLegend:legend,legend:{position:'bottom',textStyle:{fontSize:19}},chartFill:'none',plotAreaFill:'none',xAxis:{visible:true,textStyle:{fontSize:16},title:'Intervention batches (50 responses each)'},yAxis:{visible:true,min:0,max,numberFormatCode:max>1?'0.0':'0.00',textStyle:{fontSize:17},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:false}});applyPresentationChartFont(ch,{fontFamily:FONT});return ch;}
const mechanismNotes='Implementation: baselines.py, EnsembleStudentSCM / EnsembleLearner / PropagatedVariancePolicy, and scripts/research/persistent_scm.py. Historical PEV uses known graphs, independently initialized neural heads, model-only candidate contexts and covariance-based integrated variance reduction summed over descendant mechanisms. It does not identify the graph or optimize the proposed terminal-risk objective. Noise proxy is residual error, not a certified aleatoric estimate.';
function table(s,values,x,y,w,h,widths,size=27){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({fill:C.line,width:1,style:'solid'});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){let z=t.getCell(r,c);z.fill=r===0?C.navy:C.paper;z.text.style={typeface:FONT,fontSize:r===0?size-2:size,color:r===0?C.white:C.ink,bold:r===0,insets:10};}return t;}
function foot(s,t){text(s,t,64,620,1120,68,22,C.muted);}
{
const s=slide('ACE uses causal mechanisms to choose experiments',notes('Slides1–4 introduce SCMs and experiment selection. Slides5–7 are an illustrative three-node calculation, with the same covariance-score structure but simplified fixed candidates and linear heads. Slide8 is an archived action selected without performance screening. Slide9 distinguishes separate historical studies. Slide10 is a research proposal. '+mechanismNotes,[SRC.theory,SRC.public,DEMO]),C.navy);
text(s,'Structural causal model (SCM)',64,170,1100,60,42,C.light,true);
chain(s,267);text(s,'cause',269,266,150,45,26,C.light);text(s,'cause',659,266,150,45,26,C.light);
text(s,'Inputs',64,415,450,45,29,C.light,true);
text(s,'Observed responses\nKnown graph and legal actions\nExperiment budget',64,477,515,137,30,C.white);
text(s,'Outputs',700,415,490,45,29,C.light,true);
text(s,'The next intervention\nUpdated mechanism models\nPredictions for new actions',700,477,505,137,30,C.white);
}
{
const s=slide('Each SCM equation describes one mechanism',notes('SCM definition: Xi=fi(Pa_i,Ui). Edges specify direct causes in the assumed graph, f describes the mechanism, U describes external disturbances. The toy is a fully observed, deterministic chain with dimensionless values. Its true rules are M=2.5X and Y=3M, with U=0. These truth equations explain the example to the audience; the policy receives only its candidate models and allowed actions, not oracle truth. Known graph does not imply known mechanisms. General SCMs may have dependent disturbances; identification and fitting require explicit assumptions.',[SRC.theory,DEMO]));
text(s,'Xᵢ = fᵢ(Paᵢ, Uᵢ)',64,171,1100,92,66,C.teal,true);
text(s,'Parents Paᵢ supply inputs. Uᵢ represents outside influences.',64,289,1100,78,31);
chain(s,400);text(s,'M = 2.5X',325,503,300,50,30,C.teal,true);text(s,'Y = 3M',735,503,300,50,30,C.teal,true);
foot(s,'Worked example: known graph, unknown slopes, zero disturbances and dimensionless values.');
}
{
const s=slide('An intervention replaces one mechanism',notes('do(M=m) fixes M independently of its usual parents and replaces M=fM(X,UM) with M=m. X→M is removed for this experiment, while M→Y remains. Row X=x,M=m,Y=y supplies a natural Y label with measured parent m, but is not a natural M=fM(X) label. Under do(X=x), both M and Y remain naturally generated and eligible under the stated retention/noise assumptions. The implementation excludes the intervened head and uses observed parents for eligible training. Candidate simulation propagates ensemble-mean parents. Historical acquisition-study evaluation instead predicts each mechanism from observed parents; the toy composed target forecast and delivery studies have different evaluation semantics.',[SRC.theory,DEMO]));
text(s,'Ordinary system',64,169,550,50,31,C.muted);
chain(s,231);
text(s,'Experiment: do(M = 1)',64,389,650,61,36,C.teal,true);
chain(s,459,true);text(s,'Incoming edge removed',178,554,490,48,26,C.coral);text(s,'Y still responds to M',761,554,424,48,26,C.teal);
foot(s,'One paid response can train several natural mechanisms. It is still one paid response.');
}
{
const s=slide('Exhaustive grids multiply the work',notes('Analytic representation-size illustration, not an ACE benchmark. Define a deterministic discrete SCM with10independently controllable five-valued input nodes and9two-parent mechanism nodes arranged as a binary reduction tree. Every endogenous output also has5values. A naive associative lookup table for the final response over the full joint input grid requires5^10=9,765,625entries. With the graph supplied, nine local mechanism tables require9*5^2=225entries, a43,402.78fold representation-count difference. This is not a measured intervention-count reduction. Learning local tables assumes each required parent configuration is accessible with the child mechanism intact and allvariablesobserved; without internal interventions, reachability can fail. Sharedparentsettings may reveal several labels. Noncausal regressors can exploit smoothness/sparsity and need not enumerate a grid; the comparator is specifically exhaustive lookup, not allassociativelearning. No empirical superiority over this new baseline has been measured. Current PEV scores candidates by descendant ensemble covariance reduction; details retained in the worked calculation.',[SRC.theory,DEMO]));
text(s,'Illustration: 10 inputs × 5 settings per input',64,156,1130,56,32,C.muted);
text(s,'Naive joint response table',64,261,625,53,32,C.coral,true);
text(s,'9,765,625',64,344,650,112,76,C.coral,true);
text(s,'joint input configurations',64,463,650,48,30,C.ink);
text(s,'SCM mechanism tables',771,260,440,59,30,C.teal,true);
text(s,'225',771,347,440,99,72,C.teal,true);
text(s,'local parent configurations\n9 mechanisms × 25 each',771,462,444,99,28,C.ink);
foot(s,'Representation-size illustration · known graph · five-valued variables · accessible local parent settings');
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
for(const [child,parents] of Object.entries(e.graph))for(const pa of parents)edge(s,nodes[pa],nodes[child],desc.has(pa)?C.coral:C.line);
text(s,'First selected action: do(X7 = 3.8571)  →  50 responses',64,568,1130,48,31,C.teal,true);
foot(s,'Seed 5000 graph. Coral marks the intervention and outgoing descendant paths; the full study uses 20 systems.');
}
{
const s=slide('65% lower prediction error than random',notes('All20shift30systemsseeds5000–5019. Arithmeticmean noise-free feasible nonroot mechanism MSE, measured-parent prediction. Full32checkpoints shown atleft; samecurves4–32zoomatright. FinalACE.0154302049,random.0438980071,64.85%lowerratioofmeans,19/20pairedwins. Secondarycontrastp.00750. Prespecifiedcoveragecontrastfails p.05108 andnaivevariancescoringunresolved p.62176. Earlyrandomadvantage remains visible. No Bresults,confidenceband,newfit ornewhypothesis.',[SRC.shift,EVIDENCE]));
text(s,'Mean mechanism MSE · 20 systems · lower is better',64,155,1120,45,28,C.muted);
text(s,'Full recorded curve',64,219,385,36,25,C.ink,true);
text(s,'Detail: batches 4–32',483,219,620,36,25,C.ink,true);
curve(s,Array.from({length:32},(_,i)=>i),{left:64,top:265,width:360,height:277},6.5,false);
curve(s,Array.from({length:29},(_,i)=>i+3),{left:464,top:265,width:740,height:277},0.13,true);
text(s,'Final MSE: 0.01543 ACE vs 0.04390 random  |  19 / 20 wins',64,561,1130,50,31,C.teal,true);
foot(s,'20 synthetic causal systems · equal 2,000-response budgets · ACE / PEV versus non-leaf random selection');
}
{
const s=slide('64% fewer interventions at matched error',notes('New descriptivepost-hocdisplayonly. TargetdefinedasrandomfinalgroupmeanMSE .043898007078491626. Firstobservedgroupmeancrossing,nointerpolation:ACEbatch10,500interventionresponses+120obs=620total,mean.04007927;randombatch28,1400interventionresponses+360obs=1760total,mean.04216014. 1−10/28=64.2857%fewerbatches/interventionresponses;1−620/1760=64.7727%fewertotalresponses. Randomcrossesbackabove atbatch31. Bothoriginalcampaignsactuallyran32batches2000totalresponses;thesearedescriptiveprefixcomparisons,notactualsavedcompute/measurements,norvalidatedprospectivestoppingrule. Groupmeanfirstcrossingnotmeanofindividualfirstcrossings. Noone-systemorpopulationguarantee. Originalstudiesshowfixedbudgetpredictionadvantage;prospectivesample-efficiencyconfirmationremainsfuture.',[EVIDENCE,SRC.shift]));
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
const suffix=process.env.DECK_SUFFIX||'v5';
const r=await finalizePresentation({explicitTotalSlideCount:10,workspaceDir:'/Users/pat/code/ACE',candidatePath:DIR+'/draft.pptx',finalPath:OUT+'/ACE_mechanisms_and_evidence_'+suffix+'.pptx',pythonExecutable:'/Users/pat/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...([5,6,7].flatMap(n=>['--require-native-table-slide',String(n)]))],requiredNativeTableOwnerSlides:[5,6,7],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:DIR+'/validation-'+suffix+'.json'});console.log(JSON.stringify(r));
}
console.log(JSON.stringify({slides:p.slides.items.length,font:FONT}));
