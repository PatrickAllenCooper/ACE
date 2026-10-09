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
const mechanismNotes='Implementation: baselines.py, EnsembleStudentSCM / EnsembleLearner / PropagatedVariancePolicy, and scripts/research/persistent_scm.py. Historical PEV uses known graphs, independently initialized neural heads, model-only candidate contexts and covariance-based integrated variance reduction summed over descendant mechanisms. It does not identify the graph or optimize the proposed terminal-risk objective. Noise proxy is residual error, not a certified aleatoric estimate.';
function table(s,values,x,y,w,h,widths,size=27){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({fill:C.line,width:1,style:'solid'});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){let z=t.getCell(r,c);z.fill=r===0?C.navy:C.paper;z.text.style={typeface:FONT,fontSize:r===0?size-2:size,color:r===0?C.white:C.ink,bold:r===0,insets:10};}return t;}
function foot(s,t){text(s,t,64,620,1120,68,22,C.muted);}
{
const s=slide('ACE uses causal mechanisms to choose experiments',notes('Slides1–4 introduce SCMs and experiment selection. Slides5–7 are an illustrative three-node calculation, with the same covariance-score structure but simplified fixed candidates and linear heads. Slide8 is an archived action selected without performance screening. Slide9 distinguishes separate historical studies. Slide10 is a research proposal. '+mechanismNotes,[SRC.theory,SRC.public,DEMO]),C.navy);
text(s,'Structural causal model (SCM)',64,170,1100,60,42,C.light,true);
text(s,'X  →  M  →  Y',64,267,1100,95,76,C.white,true);
text(s,'Inputs',64,415,450,45,29,C.light,true);
text(s,'Observed responses\nKnown graph and legal actions\nExperiment budget',64,477,515,137,30,C.white);
text(s,'Outputs',700,415,490,45,29,C.light,true);
text(s,'The next intervention\nUpdated mechanism models\nPredictions for new actions',700,477,505,137,30,C.white);
}
{
const s=slide('Each SCM equation describes one mechanism',notes('SCM definition: Xi=fi(Pa_i,Ui). Edges specify direct causes in the assumed graph, f describes the mechanism, U describes external disturbances. The toy is a fully observed, deterministic chain with dimensionless values. Its true rules are M=2.5X and Y=3M, with U=0. These truth equations explain the example to the audience; the policy receives only its candidate models and allowed actions, not oracle truth. Known graph does not imply known mechanisms. General SCMs may have dependent disturbances; identification and fitting require explicit assumptions.',[SRC.theory,DEMO]));
text(s,'Xᵢ = fᵢ(Paᵢ, Uᵢ)',64,171,1100,92,66,C.teal,true);
text(s,'Parents Paᵢ supply inputs. Uᵢ represents outside influences.',64,289,1100,78,31);
text(s,'X  →  M  →  Y',64,407,600,88,60,C.ink,true);
text(s,'M = 2.5X\nY = 3M',822,390,365,138,42,C.teal,true);
foot(s,'Worked example: known graph, unknown slopes, zero disturbances and dimensionless values.');
}
{
const s=slide('An intervention replaces one mechanism',notes('do(M=m) fixes M independently of its usual parents and replaces M=fM(X,UM) with M=m. X→M is removed for this experiment, while M→Y remains. Row X=x,M=m,Y=y supplies a natural Y label with measured parent m, but is not a natural M=fM(X) label. Under do(X=x), both M and Y remain naturally generated and eligible under the stated retention/noise assumptions. The implementation excludes the intervened head and uses observed parents for eligible training. Candidate simulation propagates ensemble-mean parents. Historical acquisition-study evaluation instead predicts each mechanism from observed parents; the toy composed target forecast and delivery studies have different evaluation semantics.',[SRC.theory,DEMO]));
text(s,'Ordinary system',64,169,550,50,31,C.muted);
text(s,'X  →  M  →  Y',64,252,660,80,56,C.ink,true);
text(s,'Experiment: do(M = 1)',64,389,650,61,36,C.teal,true);
text(s,'X       M = 1  →  Y',64,478,708,83,50,C.teal,true);
text(s,'Learn Y from measured M\n\nExclude the clamped M label',825,255,379,226,30);
foot(s,'One paid response can train several natural mechanisms. It is still one paid response.');
}
{
const s=slide('Candidate experiments receive numerical scores',notes(mechanismNotes+' For descendant head j and simulated parent context x, the implemented contribution averages Cov_k(f_k(x),f_k(r))²/[Var_k(f_k(x))+noise_proxy_j] across reference contexts r and simulated contexts. Scores sum over descendants, not necessarily the final target alone. Candidates can be jittered and epsilon exploration can select a random candidate. Linear Gaussian conditioning motivates the covariance formula but does not certify neural ensemble calibration or optimal experimental design. A target-specific version and protection from wrong transferred priors are proposed work.',[SRC.theory,DEMO,'https://papers.nips.cc/paper/1011-active-learning-with-statistical-models.pdf']));
text(s,'1  Simulate each legal intervention with the ensemble',64,166,1110,66,32);
text(s,'2  Estimate how much it can reduce mechanism uncertainty',64,255,1110,70,32);
text(s,'3  Select an action, then pay for its observed response',64,345,1110,67,32);
text(s,'Score contribution = mean Cov² / (Variance + noise proxy)',64,460,1110,100,38,C.teal,true);
foot(s,'Current PEV sums descendant contributions. Neural ensemble scores are estimates, not certified gains.');
}
{
const s=slide('Exact inputs to the three-node demonstration',notes('Illustrative configuration only. Three linear ensemble members: M slopes1,2,3, Y slopes2,3,4; ensemble means2X and3M. Candidates ordered X−1,X0,X1,M−1,M0,M1. Reference parent values−1,0,1. Population covariance denominator3, residual noise proxy1/20, deterministic mean-propagated parent contexts, no jitter or epsilon exploration. Preintervention observation history empty for this hand-constructed initial state. One response remaining. Oracle truth hidden from score: M=2.5X,Y=3M. Production historical confirmation uses neural heads and different grids/batches. All numbers are generated by explain_ace_mechanics.py with exact rational arithmetic.',[DEMO,'/Users/pat/code/ACE/scripts/research/explain_ace_mechanics.py']));
table(s,[['Mechanism','Member slopes','Mean prediction'],['M from X','1, 2, 3','M̂ = 2X'],['Y from M','2, 3, 4','Ŷ = 3M']],64,163,1136,216,[345,335,456],29);
text(s,'Actions: clamp X or M to −1, 0 or +1',64,428,1100,60,33,C.teal,true);
text(s,'References: −1, 0, +1     Noise proxy: 0.05\nBudget: one response     Ties: first action in the list',64,510,1100,94,29);
foot(s,'Illustrative initialization. Fixed candidate grid. No exploration or candidate jitter in this example.');
}
{
const s=slide('The score selects do(X = −1)',notes('Exact score for M context±1 is160/387≈0.413436693. Y context±2 is640/1467≈0.436264485. Their sum is≈0.849701178. Under do(M=±1) the clamped M mechanism contributes zero, leaving only Y≈0.413436693. Zero contexts give zero covariance and zero score. X+1 tiesX−1; first-maximum tie rule choosesX−1. At that action the ensemble mean predictsM−2,Y−6. No oracle outcome is queried during scoring. The Y slope ensemble is evaluated at the simulated mean parent; it is not a member-paired composed posterior.',[DEMO]));
const vals=[['Intervention','M contribution','Y contribution','Total']];for(const r of d.candidates)vals.push([`do(${r.node} = ${r.value.decimal})`,r.score_M.decimal.toFixed(6),r.score_Y.decimal.toFixed(6),r.score.decimal.toFixed(6)]);
table(s,vals,64,163,1136,378,[340,266,266,264],25);
text(s,'Selected prediction: M̂ = −2,  Ŷ = −6',64,559,1100,61,35,C.teal,true);
foot(s,'The two endpoint X actions tie. Choosing −1 follows the declared order, not observed performance.');
}
{
const s=slide('The response updates eligible mechanisms',notes('Illustrative oracle row isX−1,M−2.5,Y−7.5. Eligible pairs M:(−1,−2.5),Y:(−2.5,−7.5); clamped X supplies no natural root-distribution update. For transparent arithmetic use one gradient step on half squared error, with etaM1/2 and etaY2/25. M slopes1,2,3 become1.75,2.25,2.75. Y slopes2,3,4 become2.5,3,3.5. At X−1, updated mean predictsM−2.25 andY−6.75. Signed final error drops from1.5 to.75 on this same illustrative row. This is not held-out improvement or production training: production uses neural heads, Adam and member-specific masks. Response budget becomes0. Do not associate this made-up row with the real archived trace.',[DEMO]));
text(s,'Observed row: X = −1, M = −2.5, Y = −7.5',64,162,1130,71,37,C.teal,true);
table(s,[['Natural mechanism','Training input','Training label'],['M','Measured X = −1','M = −2.5'],['Y','Measured M = −2.5','Y = −7.5']],64,254,1136,204,[350,396,390],28);
text(s,'After one illustrative gradient update',64,491,1100,53,29,C.muted);
text(s,'M̂ = −2.25       Ŷ = −6.75       Budget left: 0',64,551,1100,57,35,C.teal,true);
foot(s,'This transparent linear update differs from production neural training. It proves no test-error gain.');
}
{
const s=slide('A real ACE run records this first action',notes('Historical shifted-mechanism study, PEV arm, seed5000, first chronological row; selected by smallest study seed, not outcome. First targetX7,value3.857086181640625,cumulative samples50. Next rowX9,value5,cumulative100. Complete journal32interventioncalls1600responses plus10observationalcalls400responses,total42calls2000responses. Archived system.json supplies graph identity. Per-step raw response rows, candidate scores and ensembleweights are absent from this trace, so this slide does not fabricate them. This is a historical action illustration, not a new run or the toy state on slides5–7.',[d.recorded_trace.path+'/trajectory.csv',d.recorded_trace.path+'/query_budget.json',SRC.protocol,SRC.erratum]));
text(s,'do(X7 = 3.857086181640625)',64,177,1120,96,50,C.teal,true);
text(s,'50',64,327,450,116,94,C.ink,true);
text(s,'responses from the first intervention',64,454,540,92,30);
text(s,'Full campaign budget',750,330,450,55,29,C.muted);
text(s,'1,600 intervention responses\n400 observational responses\n2,000 total responses',750,413,455,140,29);
foot(s,'Seed 5000, first recorded step. The archive does not retain that step’s raw rows or candidate scores.');
}
{
const s=slide('The gains depend on the comparison',notes('Keep separate studies separate. Primary persistent confirmation:40 systems20eachhomogeneous/heterogeneous,PEV vscoverage paired differences−.01360 CI[−.02151,−.00569] Holm.00384 and−.10919 CI[−.18856,−.02981] Holm.00961.20/20 and16/20 wins. Shift30:20systems,PEVarithmeticmean.0154302049vsrandom.0438980071,19/20wins,descriptive64.85%lowergroupmean,secondaryp.00750;primarycoveragep.05108fails,scoringablationp.62176unresolved. Delivery12historiesoneemulatorgeometricexactlevelratio.183CI[.102,.327],11/12wins;exposedgrid,median3scoredinitializations,unequalfittingcompute. Retainedworsening124753321ratio2.015177. Physical7/11rolling3/11physics0/11Fourier,conditionsoneapparatus. No Stage B performance pending supplemental qualification. Numbers are historical accepted summaries, no recomputation of tests.',[SRC.public,SRC.claims,SRC.shift,'/Users/pat/code/ACE/results/research_persistent_confirmation_v1/README.md']));
table(s,[['Question','Supported result','Important boundary'],['Mechanism prediction','Both fresh primary\ncoverage contrasts pass','Known synthetic graphs\nScoring ablation unresolved'],['Mechanism vs random','19 / 20 wins\n64.85% lower mean MSE','Secondary comparison\nPrimary coverage gate failed'],['Better final fitting?','11 / 12 histories improve\nGeometric error ratio 0.183','One worsens 2.015×\nExtra fitting, scored inits']],64,160,1136,390,[285,410,441],27);
text(s,'Physical boundary: delivery beats Fourier in 0 of 11 conditions',64,572,1120,70,30,C.coral,true);
}
{
const s=slide('Foundation models can propose reusable mechanisms',notes('Research proposal, not current demonstrated ACE advantage. Numerical foundation model such asTabPFNv2 provides reusable prediction priors. Language model proposes typed mechanism forms or constraints, whose parameters and plausibility trustednumericalcode evaluates. SCM enforces interventionsemantics/eligibleupdates, numerical experimentselector evaluates candidatevalue/cost. Retain broad data-onlyalternative andtestwrongpriorrecovery/protection ofunchangedmechanisms. MDA alreadycombines LLMproposals,numericalBayesianinferenceanddesign,includingtask-awareVoI discussion. LLM-SR providesprogramstructureproposalswithnumericalparametersearch. Neither genericcomposition nortarget-riskformulaalone establishes novelty. Proposedclaimtest must isolatefoundationcomponent from estimator/acquisition/finalfitchanges. No universaladvantage or causalidentificationfrompredictionalone.',[SRC.theory,'https://arxiv.org/html/2608.09696','https://github.com/deep-symbolic-mathematics/LLM-SR','https://www.nature.com/articles/s41586-024-08328-6']));
text(s,'Pretrained numerical model or language proposer',64,172,1100,62,35,C.teal,true);
text(s,'SCM mechanisms with explicit intervention rules',64,274,1100,62,35,C.ink,true);
text(s,'Numerical checks and experiment selection',64,376,1100,62,35,C.ink,true);
text(s,'Test: fewer paid responses at matched target error',64,486,1110,69,36,C.teal,true);
foot(s,'Proposed work. Compare strong data-only controls and deliberately wrong priors. Prior art already combines these ingredients.');
}
await fs.mkdir(DIR+'/previews',{recursive:true});
for(const [i,s] of p.slides.items.entries()){const b=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(DIR+'/previews/slide-'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await b.arrayBuffer()));}
await (await PresentationFile.exportPptx(p)).save(DIR+'/draft.pptx');
await fs.writeFile(DIR+'/speaker_notes.md',p.slides.items.map((s,i)=>`# Slide ${i+1}\n\n${s.speakerNotes.text}`).join('\n\n'));
await fs.writeFile(DIR+'/montage.webp',new Uint8Array(await (await p.export({format:'webp',montage:true})).arrayBuffer()));
if(process.env.FINALIZE==='1'){
const suffix=process.env.DECK_SUFFIX||'v4';
const r=await finalizePresentation({explicitTotalSlideCount:10,workspaceDir:'/Users/pat/code/ACE',candidatePath:DIR+'/draft.pptx',finalPath:OUT+'/ACE_mechanisms_and_evidence_'+suffix+'.pptx',pythonExecutable:'/Users/pat/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...([5,6,7,9].flatMap(n=>['--require-native-table-slide',String(n)]))],requiredNativeTableOwnerSlides:[5,6,7,9],fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:DIR+'/validation-'+suffix+'.json'});console.log(JSON.stringify(r));
}
console.log(JSON.stringify({slides:p.slides.items.length,font:FONT}));
