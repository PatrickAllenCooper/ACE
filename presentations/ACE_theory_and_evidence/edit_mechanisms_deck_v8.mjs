import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import {resolvePresentationFont,finalizePresentation} from '/Users/pat/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/artifact_tool_utils.mjs';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
const ROOT='/Users/pat/code/ACE',OUT=ROOT+'/presentations/ACE_theory_and_evidence',DIR=ROOT+'/.codex-artifacts/ace-mechanism-deck-v8/build';
const SKILL='/Users/pat/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const INPUT=OUT+'/ACE_mechanisms_and_evidence_v7.pptx';
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
const sourceObjects=(await p.inspect({kind:'shape,textbox',maxChars:200000})).ndjson.split('\n').filter(Boolean).map(JSON.parse);
const sourceFills=new Map(sourceObjects.map(o=>[o.id,o.fillColor]));
// Whole-deck white design. Keep numerical charts and tables as editable evidence.
for(const s of p.slides.items){
 s.background.fill=C.white;
 for(const sh of s.shapes.items){
  if(sourceFills.get(sh.id)==='#FAF8F3')sh.fill=C.white;
  if(sh.name==='title'){sh.text.style={fontSize:44};}
 }
 for(const t of s.tables.items){
  t.styleOptions={headerRow:false,bandedRows:false};
  t.borders.assign({fill:'#DEE7E8',width:0.7,style:'solid'});
  for(let r=0;r<t.rows.length;r++)for(let c=0;c<t.columns.items.length;c++){
   const z=t.getCell(r,c);z.fill=r===0?'#F0F6F6':C.white;z.text.style={color:r===0?C.teal:C.ink,bold:r===0};
  }
 }
}
function clearSlide(s){for(const sh of [...s.shapes.items])sh.delete();for(const im of [...s.images.items])im.delete();}
function heading(s,title,page){text(s,title,64,52,1140,95,44,C.ink,true,'title');text(s,String(page).padStart(2,'0'),1174,664,42,26,20,C.muted,false,'page');}
function centered(s,value,x,y,w,h,size=28,color=C.ink,bold=false){const sh=text(s,value,x,y,w,h,size,color,bold);sh.text.style={alignment:'center'};return sh;}
const cover=p.slides.items[0];clearSlide(cover);
const buf=await fs.readFile(OUT+'/ACE_compressor_SCM_white_v8.png');
cover.images.add({blob:buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength),contentType:'image/png',alt:'Conceptual compressor cutaway on white with a faint structural causal graph in the background',fit:'contain',position:{left:0,top:0,width:1280,height:720}});
text(cover,'ACE',64,102,460,124,100,C.teal,true,'title');
text(cover,'Active Causal\nExperimentalism',66,264,525,150,48,C.ink,true,'technology');
text(cover,'Every experiment counts',67,465,530,65,33,C.ink,false,'tagline');
text(cover,'Compressor concept illustration',67,664,610,30,18,C.muted);
cover.speakerNotes.text+='\nVersion8: technology name Active Causal Experimentalism and user-provided tagline Every experiment counts. New white compressor/SCM concept cover edited with built-in image generation from the prior concept asset. Illustrative machine, not an experimental photograph. Prompt saved in cover_asset_v8.json. All measured acquisition results remain historical synthetic neural-SCM comparisons.';
const s2=p.slides.items[1];clearSlide(s2);heading(s2,'A compressor as a causal model',2);
text(s2,'Illustrative test rig with other operating conditions fixed',64,153,1130,50,28,C.muted);
const nx=node(s2,'X',146,286,124,95),nm=node(s2,'M',573,286,124,95),ny=node(s2,'Y',1000,286,124,95);
edge(s2,nx,nm,'right','left',C.teal,'straight',7);edge(s2,nm,ny,'right','left',C.teal,'straight',7);
centered(s2,'Drive command',64,409,290,47,32,C.ink,true);centered(s2,'Shaft speed',491,409,290,47,32,C.ink,true);centered(s2,'Pressure rise',918,409,290,47,32,C.ink,true);
centered(s2,'Input we choose',64,464,290,46,26,C.muted);centered(s2,'M = f(X)',491,464,290,46,32,C.teal);centered(s2,'Y = g(M)',918,464,290,46,32,C.teal);
text(s2,'An SCM models each local mechanism, then connects its predictions.',64,559,1130,60,29,C.ink);
text(s2,'Teaching values: M = 2.5X, Y = 3M. Normalized deviations around a reference condition.',64,639,1090,53,21,C.muted);
s2.speakerNotes.text+='\nCompressor analogy introduced in version8: X is normalized drive-command deviation, M normalized shaft-speed deviation, and Y normalized pressure-rise deviation. Other conditions are held fixed. The supplied graph is a simplified teaching chain with linear, zero-disturbance illustrative rules, not validated compressor dynamics. Negative teaching values mean below-reference deviations, not negative physical RPM or absolute pressure. Measured historical experiments later are synthetic systems. The real compressor can have additional direct paths, nonlinearities, operating limits and confounding, which this teaching model does not establish.';
const s3=p.slides.items[2];clearSlide(s3);heading(s3,'Two experiments on the same compressor',3);
text(s3,'Change the drive command',64,153,920,47,31,C.teal,true);
const x1=node(s3,'X',146,232,124,80),m1=node(s3,'M',573,232,124,80),y1=node(s3,'Y',1000,232,124,80);
edge(s3,x1,m1,'right','left',C.teal,'straight',7);edge(s3,m1,y1,'right','left',C.teal,'straight',7);
centered(s3,'Drive command',64,326,290,45,27);centered(s3,'Shaft speed responds',452,326,367,45,27);centered(s3,'Pressure responds',892,326,340,45,27);
text(s3,'Hold shaft speed with a test-rig controller: do(M = 1)',64,399,1110,49,31,C.teal,true);
const x2=node(s3,'X',146,479,124,80),m2=node(s3,'M = 1',573,479,124,80,C.coral),y2=node(s3,'Y',1000,479,124,80);
edge(s3,m2,y2,'right','left',C.teal,'straight',7);
centered(s3,'Drive-to-speed link bypassed',139,584,540,47,26,C.coral);centered(s3,'Pressure still responds',817,584,415,47,26,C.teal);
text(s3,'Learn only from mechanisms left natural. The imposed speed is not a training label for f(X).',64,651,1090,47,20,C.muted);
s3.speakerNotes.text+='\nVersion8 uses the same simplified compressor chain as slide2. Internal intervention assumes an approved independent test-rig controller can hold shaft speed, bypassing the usual drive-command mechanism. It is a legal action of this illustrative simulator/test rig, not a claim that arbitrary internal compressor quantities can be clamped. Unit1 means a normalized above-reference speed. The incoming X-to-M edge is absent during the clamp. M-to-Y remains. Numbers and eligibility rules are unchanged.';
// Keep the full mechanism slide but lighten its diagram frames and define physical labels.
const s6=p.slides.items[5];
for(const sh of s6.shapes.items){
 if(sh.geometry==='rect'&&sh.line?.width===2){sh.fill=C.white;sh.line.width=1.25;}
 if(sh.name==='title')sh.text='The ACE experiment loop';
 if(String(sh.text)==='Action selector\nClamp X or M\nto −1, 0 or +1')sh.text='Action selector\nDrive X or speed M\nLow, reference, high';
 if(String(sh.text)==='Environment\nSimulator or apparatus')sh.text='Environment\nSimulator or test rig';
}
for(const i of [6,7,8])p.slides.items[i].speakerNotes.text+='\nVersion8 physical reading: X drive-command deviation, M shaft-speed deviation, Y pressure-rise deviation. Normalized toy values, zero disturbances. These exact arithmetic results are a teaching example, not measured compressor data.';
const s7=p.slides.items[6];
for(const sh of s7.shapes.items){
 if(sh.name==='title')sh.text='Exact inputs for the compressor example';
 if(String(sh.text).startsWith('Actions: clamp')){sh.text='Actions: drive X or shaft speed M at −1, 0 or +1';sh.text.style={fontSize:30};}
 if(String(sh.text).startsWith('Simplified linear ensemble'))sh.text='Normalized deviations: low −1, reference 0, high +1. Y is pressure rise.';
}
// Minimal copy and consistent punctuation on retained empirical slides.
for(const sh of p.slides.items[9].shapes.items){if(sh.name==='title')sh.text='A recorded synthetic system: 30 mechanisms';}
for(const sh of p.slides.items[10].shapes.items){
 if(String(sh.text).startsWith('19 of 20 systems won'))sh.text='ACE wins on 19 of 20 systems at equal budgets';
 if(String(sh.text).startsWith('Final MSE:'))sh.text='Final MSE: 0.01543 ACE vs 0.04390 random';
 if(String(sh.text).startsWith('20 synthetic systems'))sh.text='20 synthetic systems. 2,000 responses per arm. Both use the same SCM learner.';
}
await fs.mkdir(DIR+'/previews',{recursive:true});
await fs.writeFile(DIR+'/speaker_notes.md',p.slides.items.map((s,i)=>`# Slide ${i+1}\n\n${s.speakerNotes.text}`).join('\n\n'));
await fs.writeFile(DIR+'/edited-inspect.ndjson',(await p.inspect({kind:'slide,textbox,table,chart',maxChars:150000})).ndjson);
await (await PresentationFile.exportPptx(p)).save(DIR+'/draft.pptx');
for(const [i,s]of p.slides.items.entries()){const b=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(DIR+'/previews/slide-'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await b.arrayBuffer()));}
execFileSync('/Users/pat/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',[OUT+'/restore_chart_sources_v8.py'],{stdio:'inherit'});
console.log(JSON.stringify({slides:p.slides.items.length,font:FONT}));
