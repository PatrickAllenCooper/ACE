from audit_common import *
import collections,struct
HERE=Path(__file__).resolve().parent;OUT=HERE/'records04';OUT.mkdir(exist_ok=False)
report=dict(status='incomplete',acceptance=False);began=time.monotonic();cpu=time.process_time()
def chain(path):
 b=raw(path);need(b.endswith(b'\n'),'truncated journal')
 values=[];previous=None
 for line in b.splitlines():
  v=decode(line);h=v.pop('sha256');need(v['previous_sha256']==previous and digest(v)==h,'chain byte pin '+str(path));v['sha256']=h;previous=h;values.append(v)
 return values
try:
 l=LOCAL/'remote/launch';b=l/'collection/session/worker/task/collection';packet=read(l/'inputs/collection-packet.json');proposal=read(l/'inputs/proposal.json')
 core_sources={k:v for k,v in packet['source_pins'].items() if k!='scripts/research/collect_llm_intervention_scientific_training_cpu.py'}
 schedule=read(HERE/'expected-public-schedule.json');term=read(b/'terminal.json');inv=read(b/'inventory.json');plan=read(b/'plan.json');intent=read(b/'intent.json')
 need(sha(raw(l/'inputs/collection-packet.json'))=='a86dfd705dba2ca213c3ac25884c3e3deca177e00d8ce8be72732a3baf878c35','unchanged inner scientific packet')
 need(digest(proposal)==packet['proposal_sha256']=='585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8','fixed proposal')
 worlds=proposal['training_worlds'];ids=[w['world_id'] for w in worlds];roster=[f'policy_train/{wid}/snapshot-{si:03d}' for wid in ids for si in range(5)]
 need(ids==list(range(202611070000,202611070032))==packet['world_roster']==term['world_roster']==plan['world_roster']==intent['world_roster'],'exact32 training roster')
 need(term['record_roster']==inv['record_roster']==roster and len(inv['records'])==len(plan['records'])==160,'full ordered160 records')
 need(intent['world_pins']==packet['world_pins'] and intent['configuration_pins']==packet['configuration_pins'] and intent['expected_source_pins']==core_sources==plan['source_file_sha256'],'source/world/config identity')
 need(set(core_sources)==set(proposal['source_pins'])|{'scripts/research/llm_intervention_training_collection.py'} and all(packet['source_pins'][k]==v for k,v in proposal['source_pins'].items()),'13 unchanged collection sources')
 need(intent['proposal_sha256']==digest(proposal) and intent['expected_roster_sha256']==digest(ids) and plan['intent_sha256']==sha(raw(b/'intent.json')),'intent/proposal/roster lineage')
 need(term['inventory_sha256']==sha(raw(b/'inventory.json')) and inv['collection_plan_sha256']==sha(raw(b/'plan.json')) and term['planned_records']==plan['records'],'complete batch inventory links')
 need(term['status']=='complete' and term['exposure_ready'] is True and term['accounting_complete'] is True and term['accounting_totals_are_lower_bounds'] is False,'full positive collection terminal')
 for k in ('partial_records_authorized_for_training','complete_case_filtering_allowed','policy_training_performed','scientific_evaluation_performed','retry_allowed'):need(term[k] is False,'collection bound '+k)
 need(plan['maximum_response_attempts']==9216 and plan['maximum_reserved_cost_units']==10240,'fixed distinct attempt/cost ceilings')
 census=read(HERE/'journals02/journal-census.json');need(census['errors']==[] and census['counts_are_lower_bounds'] is False and not census['missing_executor_journals'] and not census['unplanned_raw_returns'] and not census['saved_failure_files'],'complete journal census')
 need(census['accounting']==term['accounting'],'independent charge census equals terminal')
 all_exec={};all_reservations={};all_raw={};adapter_keys=set();expected_adapters=set();totalpairs=ties=strict=0;record_rows=[];world_rows=[];counts=collections.Counter()
 for wi,(world,scheduled,tw) in enumerate(zip(worlds,schedule['worlds'],term['worlds'])):
  wid=world['world_id'];key=str(wid);directory=b/f'world-{wi:03d}';wp=read(directory/'plan.json');wt=read(directory/'terminal.json');public=world['public'];training=world['training']
  need(scheduled['world_id']==wid and scheduled['world_sha256']==packet['world_pins'][key] and scheduled['configuration_sha256']==packet['configuration_pins'][key],'prepared audit schedule independent binding')
  supplied=read(b/f'input-{wi:03d}.json');need(digest(supplied['world'])==packet['world_pins'][key] and digest(supplied['configuration'])==packet['configuration_pins'][key],'sealed original input contents')
  need(digest(read(directory/'private-world.json'))==packet['world_pins'][key],'unchanged reused private world; no regeneration')
  need(wp['world_id']==wt['world_id']==wid and wp['world_sha256']==wt['world_sha256']==packet['world_pins'][key] and wp['scope']==wt['scope']=='policy_train','world identity')
  need(wp['journal_plan']==scheduled['journal_plan'] and wp['menu']==scheduled['menu']==public['menu'],'predeclared journal/menu roster')
  need(wp['task']==public['task'] and wp['task_sha256']==digest(public['task'])==public['task_sha256'],'fixed scientific task')
  need(wp['behavior_method']==training['behavior_method']==scheduled['behavior_method'] and wp['planning_draws']==training['planning_draws']==32 and wp['snapshots']==scheduled['snapshots'],'fixed behavior/snapshot/teacher recipes')
  need(wp['request_namespace']==training['request_namespace']==scheduled['namespace'] and wp['replicate_ids']==training['replicate_ids']==['r0','r1','r2','r3'],'namespace/replicates')
  need(wp['structural_scores']==training['structural_scores']==True and wp['tie_multiplier']==training['tie_multiplier']==2.0 and wp['minimum_gain_gap']==training['minimum_gain_gap']==0.0,'fixed training score/tie contract')
  need(wp['initial_actions']==public['initial_actions'] and wp['initial_cost_units']==8 and wp['total_trajectory_budget']==32,'fixed initial/acquisition cost')
  need(wt['status']=='complete' and wt['dataset_complete'] is True and wt['caller_learner_unchanged'] is True and wt['scoring_learner_updates']==0 and wt['stop_inference_performed'] is False and wt['unfinished_snapshot_indices']==[],'world completion/no inference on stopping')
  need(wt['plan_sha256']==digest(wp) and wt['actual_initial_cost_units']==8 and wt['actual_acquisition_cost_units']==24 and wt['training_pairs']==112,'complete world plan/cost/pairs')
  wc=read(HERE/'journals02'/f'world-{wi:03d}.json');need(wc['errors']==[] and wc['accounting']=={k:wt['accounting'][k] for k in wc['accounting']}=={k:tw['accounting'][k] for k in wc['accounting']},'independent world charges')
  need(wt['accounting']['accounting_complete'] is True and wt['accounting']['accounting_errors']==[] and tw['status']=='complete','world no incomplete attempts')
  for entry in scheduled['journal_plan']:
   rel=f'world-{wi:03d}/'+entry['path'];events=chain(b/rel);reservations={v['request_id']:v for v in events if v['event']=='reserve'};responses={v['request_id']:v for v in events if v['event']=='response'}
   need(events[0]['event']=='begin' and events[-1]['event'] in ('closed','stop') and len(reservations)==len(responses),'positive executor completion')
   need(set(reservations)==set(responses)==set(range(len(responses))),'complete sequential requests')
   need(events[-1]['spent_units']==sum(v['cost_units'] for v in reservations.values()),'executor terminal charges')
   for rid,reserve in reservations.items():
    response=responses[rid];ret=read((b/rel).with_name((b/rel).stem+f'.return-{rid:03d}.json'))
    need(ret['request_id']==rid and ret['journal']==Path(rel).name and ret['returned_to_wrapper'] is True and ret['raw_json_available'] is True,'original return witness')
    need(response['values']==ret['raw_json'] and response['action_id']==reserve['action']['action_id'] and (entry['kind']=='trajectory' or ret['action_id']==response['action_id']) and response['cost_units']==reserve['cost_units'],'reserve/return/accepted byte association')
    need(response['kind']==reserve['action']['kind'] and response['targets']==dict(reserve['action']['targets']),'accepted action target binding')
    all_reservations[(rel,rid)]=reserve;all_exec[(rel,rid)]=response;all_raw[(rel,rid)]=ret
   if entry['kind']=='trajectory':need(events[-1]['event']=='stop' and wt['trajectory_learner_updates']==len(responses),'explicit terminal stop and actual trajectory updates')
  # Adapter journal sources and namespaces are predeclared before any response.
  adapters=[('initial.world.jsonl',world['initial_binding'],False,8),('acquisition.world.jsonl',training['measurement_binding'],False,24)]
  for si,spec in enumerate(training['snapshots']):
   for ri,rid in enumerate(training['replicate_ids']):
    if spec['candidates']:
     for ci,_ in enumerate(spec['candidates']):adapters.append((f'snapshot-{si:03d}/world-journals/replicate-{ri:03d}.branch-{ci:03d}.world.jsonl',spec['counterfactual_bindings'][ri],False,1))
     adapters.append((f'snapshot-{si:03d}/world-journals/replicate-{ri:03d}.scoring.world.jsonl',spec['score_bindings'][ri],True,8))
  for rel,binding,structural,limit in adapters:
   expected_adapters.add(f'world-{wi:03d}/'+rel);events=chain(directory/rel);bind=events[0]
   need(bind['event']=='bind' and all(bind[k]==v for k,v in binding.items()) and bind['descriptor_sha256']==bind['expected_sha256']==packet['world_pins'][key] and bind['structural'] is structural and bind['max_requests']==limit,'adapter frozen binding')
   need(events[-1]['event']=='closed' and bind['reservation_authority']=='ChargedExecutor','positive adapter closure')
   maps=[v for v in events if v['event']=='mapping'];outcomes=[v for v in events if v['event']=='outcome'];attempts=[v for v in events if v['event']=='attempt']
   need(len(maps)==len(outcomes)==len(attempts)==events[-1]['responses_prepared']<=limit and len(events)==2+3*len(maps),'complete adapter event roster')
   exec_rel=bind['executor_journal'].removeprefix(read(l/'packet.json')['remote_root']+'/collection/session/worker/task/collection/')
   for mapping,outcome in zip(maps,outcomes):
    rid=mapping['request_id'];association=(exec_rel,rid);need(association in all_reservations and association not in adapter_keys,'unique adapter-to-reservation witness');adapter_keys.add(association)
    reserve=all_reservations[association];response=all_exec[association]
    need(mapping['reservation_sha256']==outcome['reservation_sha256']==reserve['sha256'] and mapping['action']==reserve['action'] and mapping['descriptor_sha256']==packet['world_pins'][key],'adapter reserved action and descriptor')
    need(mapping['stream']==bind['stream'] and mapping['structural'] is structural and outcome['request_id']==rid and outcome['returned_subset_ready'] is True and outcome['values']==response['values'],'adapter actual returned subset')
    action=reserve['action'];need(mapping['noise_key']==[binding['request_namespace'],binding['replicate_id'],rid,action['action_id'],digest({'kind':action['kind'],'targets':action['targets']})],'exact independent frozen stream key')
   counts['adapter_journals']+=1
  for si,spec in enumerate(training['snapshots']):
   idx=wi*5+si;rid=roster[idx];planned=plan['records'][idx];row=inv['records'][idx];path=b/row['path'];rec=read(path/'record.json');receipt=read(path/'receipt.json');env=read(path/'envelope.json');selection=read(path/'selection.json');teacher=read(path/'teacher.json');alignment=read(path/'alignment.json');labels=read(path/'numeric-labels.json');language=read(path/'language.json')
   need(row['record_id']==planned['record_id']==rid and row['status']=='complete' and all(row[k]==v for k,v in planned.items()),'planned record exact metadata')
   recordpin=sha(raw(path/'record.json'));receiptpin=sha(raw(path/'receipt.json'))
   need(recordpin==row['record_sha256']==inv['record_pins'][rid]==alignment['record_sha256'] and receiptpin==row['supervision_receipt_sha256']==inv['supervision_receipt_pins'][rid]==rec['supervision_receipt_sha256']==alignment['supervision_receipt_sha256'],'all original record/receipt pins')
   need(rec['envelope']==env and rec['world_id']==receipt['world_id']==alignment['world_id']==wid and rec['scope']==receipt['scope']==alignment['scope']=='policy_train','matched public state identity')
   need(sha(raw(path/'envelope.json'))==receipt['envelope_file_sha256'] and digest(env)==teacher['envelope_sha256']==alignment['envelope_sha256'] and digest(env['task'])==receipt['task_sha256']==alignment['task_sha256']==wp['task_sha256'],'envelope/task association')
   need(receipt['world_sha256']==packet['world_pins'][key] and receipt['plan_file_sha256']==sha(raw(directory/'plan.json')) and receipt['selection_file_sha256']==sha(raw(path/'selection.json')) and receipt['teacher_file_sha256']==sha(raw(path/'teacher.json')),'receipt plan/teacher/selection linkage')
   need(teacher['method']==spec['teacher_method'] and teacher['chosen_before_branch_responses'] is True and rec['supervised_action_id']==teacher['decision'].get('action_id'),'fixed independent teacher label')
   need(selection['plan_sha256']==digest(wp) and selection['snapshot_index']==si and all(selection[k]==v for k,v in scheduled['snapshots'][si].items()),'fixed snapshot selection')
   need(selection['trajectory_request_count']==env['state']['requests'] and selection['actual_acquisition_cost_units']==env['state']['spent_units']-8 and selection['selection_depends_on']=='fixed cost threshold only','observed fixed-cost milestone')
   pairs=[[a,z] for j,a in enumerate(spec['candidates']) for z in spec['candidates'][j+1:]]
   need(planned['candidate_ids']==receipt['candidate_ids']==spec['candidates'] and planned['pairs']==[[v['left'],v['right']] for v in rec['preferences']]==pairs and receipt['pair_count']==len(pairs),'all planned comparisons including ties')
   need(receipt['base_learner_unchanged'] is True and receipt['scoring_learner_updates']==0 and receipt['structural_scores'] is True,'no scoring leak into learner')
   action_ids=alignment['action_ids'];need(action_ids==([None] if si==4 else sorted(a['action_id'] for a in wp['menu'])+[None]) and alignment['feature_width']==8891,'complete numeric action roster/width')
   need(labels['chosen']==action_ids.index(rec['supervised_action_id']),'saved numeric teacher label')
   chosen_command=decode(language['supervised_completion'].encode());need(chosen_command==({'stop':True} if rec['supervised_action_id'] is None else {'action_id':rec['supervised_action_id']}),'saved language teacher label')
   tied=[v for v in rec['preferences'] if v['tie']];stricts=[v for v in rec['preferences'] if not v['tie']]
   need(alignment['strict_preferences']==len(stricts) and alignment['tied_pairs']==[[v['left'],v['right']] for v in tied],'ties retained in matched alignment')
   need(labels['pairs']==[[action_ids.index(v['chosen']),action_ids.index(v['rejected'])] for v in stricts],'complete strict numeric pair labels')
   need(len(language['preference_completions'])==len(stricts),'matched language preference count')
   need([[decode(x.encode()) for x in pair] for pair in language['preference_completions']]==[[{'action_id':v['chosen']},{'action_id':v['rejected']}] for v in stricts],'all strict language preference labels')
   # Validate only the non-object NPY header/extent; never load or recalculate arrays.
   npy=raw(path/'numeric-features.npy');need(npy[:8]==b'\x93NUMPY\x01\x00','saved numeric format')
   hs=struct.unpack('<H',npy[8:10])[0];nh=ast.literal_eval(npy[10:10+hs].decode('latin1'));need(nh=={'descr':'<f8','fortran_order':False,'shape':(len(action_ids),8891)} and len(npy)==10+hs+len(action_ids)*8891*8,'exact numeric feature shape/bytes, no tensor execution')
   if si==4:
    need(not pairs and rec['supervised_action_id'] is None and env['state']['terminal'] is True and env['state']['remaining_units']==0 and teacher['decision']=={'stop':True} and receipt['feedback_terminal_sha256'] is None and receipt['stop_label_basis']=='no affordable trajectory action','terminal32 stop records, no inferred early stop')
   else:
    feedback=read(path/'feedback/terminal.json');fp=read(path/'feedback/plan.json');fb=feedback['feedback']
    need(receipt['feedback_terminal_sha256']==sha(raw(path/'feedback/terminal.json')) and feedback['status']=='complete' and feedback['plan_sha256']==digest(fp) and feedback['base_state_sha256']==fp['base_state_sha256']==selection['base_state_sha256'],'fixed base feedback source')
    need(feedback['base_learner_unchanged'] is True and feedback['scoring_learner_updates']==0 and feedback['stop_inference_performed'] is False and feedback['accounting']['accounting_complete'] is True,'complete feedback no learner contamination')
    need(fp['candidates']==fb['candidate_ids']==spec['candidates'] and fp['replicate_ids']==fb['replicate_ids']==training['replicate_ids'] and fp['task']==fb['task']==env['task'],'complete replicated comparison scope')
    need(fp['tie_multiplier']==fb['tie_multiplier']==2.0 and fp['minimum_gain_gap']==fb['minimum_gain_gap']==0.0 and fb['replicate_count']==4 and fp['scale']==fb['scale']==2.0,'frozen feedback parameters')
    need(fp['base_state_sha256']==digest(fp['base_state']) and fb['base_predictions']==[fp['fixed_base_predictions']]*4,'fixed post-state shared base predictions')
    need([{k:v[k] for k in ('left','right','tie','chosen','rejected')} for v in fb['pairs']]==rec['preferences'],'record labels exactly saved full feedback')
    for v in fb['pairs']:
     delta=v['mean_gain_difference'];threshold=v['tie_threshold'];tie=abs(delta)<=threshold
     wanted=(None,None) if tie else ((v['left'],v['right']) if delta>0 else (v['right'],v['left']))
     need(v['tie'] is tie and (v['chosen'],v['rejected'])==wanted,'saved pair classification/threshold consistency')
    for ri,replicate in enumerate(training['replicate_ids']):
     prefix=f'replicate-{ri:03d}';score=read(path/'feedback'/(prefix+'.scoring.scoring.json'))
     need(score['replicate_id']==replicate and score['learner_updates']==0 and score['target']==env['task']['target'] and fb['scoring_truths'][ri]==score['truths'],'shared scoring truth lineage')
     exec_rel=f'world-{wi:03d}/snapshot-{si:03d}/feedback/'+prefix+'.scoring.jsonl'
     need(score['receipts']==[{k:v for k,v in all_exec[(exec_rel,j)].items() if k not in ('event','monotonic_ns','previous_sha256','sha256')} for j in range(8)],'shared scoring receipts match original accepted returns')
     for ci,candidate in enumerate(spec['candidates']):
      stem=prefix+f'.branch-{ci:03d}';branch=read(path/'feedback'/(stem+'.branch.json'));mapping=read(path/'feedback'/(stem+'.mapping.json'));entry=next(v for v in fp['journals'] if v['journal']==stem+'.jsonl')
      need(mapping==branch['mapping']=={**entry,'base_state_sha256':fp['base_state_sha256'],'status':'planned_before_response'},'per-branch fixed baseline/planned mapping')
      exec_rel=f'world-{wi:03d}/snapshot-{si:03d}/feedback/'+stem+'.jsonl';response=all_exec[(exec_rel,0)]
      need(branch['receipt']=={k:v for k,v in response.items() if k not in ('event','monotonic_ns','previous_sha256','sha256')} and branch['candidate_id']==candidate and branch['replicate_id']==replicate and branch['query_predictions']==fb['branch_predictions'][candidate][ri],'branch source/response/prediction association')
      need(branch['post_state']['request_ids']==fp['base_state']['request_ids']+[fp['next_learner_request_id']],'one branch update only, fixed base')
    counts['feedback_batches']+=1
   totalpairs+=len(pairs);ties+=len(tied);strict+=len(stricts);counts['records']+=1
   record_rows.append(dict(record_id=rid,record_sha256=recordpin,receipt_sha256=receiptpin,pairs=len(pairs),strict_pairs=len(stricts),ties=len(tied)))
  world_rows.append(dict(world_id=wid,accounting=wc['accounting'],behavior_method=wp['behavior_method'],trajectory_updates=wt['trajectory_learner_updates']))
 need(totalpairs==3584 and counts['records']==160 and counts['feedback_batches']==128 and counts['adapter_journals']==4672,'full record/feedback/adapter roster')
 need(adapter_keys==set(all_reservations)==set(all_exec)==set(all_raw) and len(adapter_keys)==9145,'every response positively mapped once through adapter/raw/executor')
 need(set(census['other_jsonl_for_adapter_audit'])==expected_adapters,'no extra unreviewed adapter journals')
 need(set(inv['record_pins'])==set(inv['supervision_receipt_pins'])==set(roster),'no record omission/filtering')
 report.update(status='verified_complete_saved_collection',counts=dict(counts),retained_pairs=totalpairs,strict_pair_labels=strict,tied_pair_labels=ties,accounting=census['accounting'],executor_journals=sum(len(w['journal_plan']) for w in schedule['worlds']),records=record_rows,worlds=world_rows,all_original_returns_linked=True,numerical_arrays_loaded=False,policy_fits_or_updates_reexecuted=0,world_draws_or_responses_reexecuted=0,tokenizer_calls=0,scope='all frozen collection records and saved metadata/receipts; no regeneration, fits, utility/scoring recomputation or D selection')
except BaseException as exc:report['error']=type(exc).__name__+': '+str(exc);raise
finally:report.update(process_cpu_seconds=time.process_time()-cpu,elapsed_seconds=time.monotonic()-began);save(OUT/'audit.json',report)
