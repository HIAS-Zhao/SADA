from pathlib import Path
import os,sys,json,importlib.util,copy,math
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
ROOT=Path('@@WORKSPACE@@/drift_lora_project');R=ROOT/'results';OUT=Path(__file__).parent
PREV=R/'three_models_light_period5_expanded_20260921';HIGH=R/'resnet18_heavier_inference_highload_20260921'
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
fair=load('seed_fair',ROOT/'scripts/run_remoteclip_fair_ewc_gpu4.py');rf=load('seed_rf',ROOT/'scripts/run_omniearth_remoteclip_formal_framework.py');h=load('seed_head',PREV/'head_with_update_budget.py')
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def acc(ids,p,y):return sum(p[u]==y[u] for u in ids)/len(ids)
for seed in [43,44]:
 for fam in ['remoteclip','resnet18']:
  dst=OUT/f'seed{seed}/three_models'/fam
  if (dst/'bundle.json').exists():continue
  original=HIGH/'bundle.json' if fam=='resnet18' else PREV/fam/'bundle.json'
  b=json.loads(original.read_text());b['seed']=seed;b['gammas']=[];b['ewc']=None
  src=R/f'all_model_three_seed_repeats_20260715/seed{seed}'/fam
  splits=rf.load_existing_splits(src/'splits');pseudo=rf.load_external_pseudo_labels(src/'teacher/pseudo_labels')
  for key in ['R_train','R_val','R_test']:
   assert [r['uid'] for r in splits[key]]==b[{'R_train':'train_ids','R_val':'val_ids','R_test':'test_ids'}[key]]
  assert all(pseudo[u]==b['pseudo'][u] for u in b['val_ids'])
  rows=rf.unique_rows(splits['history_no_drift'],splits['D_top80'],splits['R_train'],splits['R_val'],splits['R_test'],splits['R_no_drift'])
  oldlabels=fair.semantic_labels(rf,splits['history_no_drift'],letters=None);targets=fair.semantic_labels(rf,splits['R_train'],letters=pseudo)
  labels=sorted(set(oldlabels.values())|set(targets.values()));mapping={v:i for i,v in enumerate(labels)}
  y=fair.indices_for_rows(splits['R_train'],targets,mapping);oy=fair.indices_for_rows(splits['history_no_drift'],oldlabels,mapping)
  for l in b['lambdas']:
   lid=l['id'];cache=PREV/fam/'features'/f'{lid}.npz'
   high=not cache.exists()
   if high:cache=HIGH/'new_candidates'/lid/'features.npz'
   features=fair.load_feature_cache(cache);assert all(r['uid'] in features for r in rows)
   raw=fair.feature_matrix(features,splits['R_train']);scaler=h.fit_scaler(raw);initial=h.initialize_head(labels,raw.shape[1],scaler);x=h.transform(raw,scaler)
   paired=f"{fam}_light{seed}:{b['fixed_lambda'] if high else lid}"
   for name,updates in [('u8',8),('u16',16),('u32',32),('u64',64),('ep1',len(y)),('ep3',len(y)*3),('ep5',len(y)*5)]:
    def train():return h.train_head(x,y,initial,epochs=math.ceil(updates/len(y)),max_updates=updates,seed=paired)
    trained,runtime,times=fair.timed_deterministic(train,repeats=3,state_arrays=lambda z:(z[0].weight,z[0].bias));state,metrics=trained
    ps=fair.option_predictions(runner=rf,head_ewc=h,state=state,features=features,rows=rows);d=dst/'train'/lid/name;h.save_head(d/'head.npz',state)
    write(d/'metrics.json',{'seed':seed,'paired_seed':paired,'updates':updates,'runtime_s':runtime,'timing_repeats_s':times,'feature_source':str(cache),'metrics':metrics})
    write(d/'predictions.json',ps)
    b['gammas'].append({'id':f'head_{name}','lambda_id':lid,'train_time_s':runtime,'val_accuracy':acc(b['val_ids'],ps,pseudo),'predictions':ps,'updates':updates,'source':'new_seed_actual_head_training'})
   if lid==b['fixed_lambda']:
    ox=h.transform(fair.feature_matrix(features,splits['history_no_drift']),scaler);fisher,fr,fs=fair.timed_fisher(head_ewc=h,x=ox,y=oy,state=initial,repeats=3)
    def train_ewc():return h.train_head(x,y,initial,epochs=1,max_updates=16,seed=paired,anchor=initial,fisher=fisher,ewc_lambda=10.)
    trained,tr,ts=fair.timed_deterministic(train_ewc,repeats=3,state_arrays=lambda z:(z[0].weight,z[0].bias));state,metrics=trained
    ps=fair.option_predictions(runner=rf,head_ewc=h,state=state,features=features,rows=rows);d=dst/'ewc';h.save_head(d/'head.npz',state)
    write(d/'metrics.json',{'seed':seed,'updates':16,'train_s':tr,'fisher_s':fr,'paired_seed':paired,'metrics':metrics});write(d/'predictions.json',ps)
    b['ewc']={'id':'head_u16_ewc','lambda_id':lid,'train_time_s':tr+fr,'val_accuracy':acc(b['val_ids'],ps,pseudo),'predictions':ps,'updates':16}
   print(seed,fam,lid,'complete',flush=True)
  b['repeat_protocol']='Fixed splits, frozen backbone features/base predictions and inference hardware profiles; new seed for head SGD order; measured new head runtime. Same training and selection budgets as seed42.'
  write(dst/'bundle.json',b)
print('ALL COMPLETE',flush=True)
