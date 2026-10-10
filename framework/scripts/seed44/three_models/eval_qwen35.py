from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json,csv,sys,os,subprocess,importlib.util,statistics,time
ROOT=Path('@@WORKSPACE@@/drift_lora_project');OUT=Path(__file__).parent;PYTHON=Path('@@VLM_PYTHON@@')
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
q=load('light_qwen_eval_utils',ROOT/'scripts/run_qwen35_omniearth_formal_framework_augmented_v2.py')
# Preserve every recorded UID; rejected model outputs count as incorrect.
def complete_prediction_map(path):
 return {r['uid']:(q.normalize(r.get('pseudo_label')) if r.get('accepted') else '') for r in map(json.loads,Path(path).read_text().splitlines()) if r.get('uid')}
q.prediction_map=complete_prediction_map
source=ROOT/'results/all_model_three_seed_repeats_20260715/seed44/qwen35/framework'
protocol=json.loads((source/'framework_results.json').read_text())['protocol'];splitroot=Path(protocol['split_root']);pseudoroot=Path(protocol['pseudo_root'])
lambda_file=Path(protocol['lambda_root'])/'lambda_dtop80_by_config.csv';configs=[q.LambdaConfig(r['lambda_id'],r['model_precision'],r['dtype'],r['roi_strategy'],float(r['token_retention_ratio']),int(float(r['effective_input_resolution'])),int(r['batch_size']),int(r['image_resolution']),int(r['max_new_tokens'])) for r in q.read_csv(lambda_file)]
for side,keep in [(336,.25),(224,.1111)]:configs.append(q.LambdaConfig(f'qwen35_08b_bf16_input{side}_uniform','bf16','bfloat16','uniform_token',keep,side,1,side,16))
source_lambdas={r['lambda_id']:r for r in q.read_csv(lambda_file)}
base_rows,adapter_rows=q.maybe_load_eval_cache(source);bases={(r['group_id'],r['lambda_id']):r for r in base_rows};adapters={(r['group_id'],r['lambda_id'],r['gamma_id']):r for r in adapter_rows}
gprofiles=q.read_csv(Path(protocol['gamma_root'])/'gamma_smallstep_by_config.csv')
# Instrument a local copy only; no change to the shared evaluator.
code=Path('@@WORKSPACE@@/qwen_eval/code/candidate_vlm_eval.py').read_text().replace('import argparse\n','import argparse\nimport time\n')
code=code.replace('            results = labeler.predict_batch([sample for _, _, sample in entries])','            labeler.torch.cuda.synchronize()\n            started = time.perf_counter()\n            results = labeler.predict_batch([sample for _, _, sample in entries])\n            labeler.torch.cuda.synchronize()\n            prediction_elapsed_s = (time.perf_counter() - started) / len(entries)')
code=code.replace('                "allowed_labels": allowed_labels(sample),','                "allowed_labels": allowed_labels(sample),\n                "prediction_elapsed_s": prediction_elapsed_s,')
(OUT/'timed_candidate_eval.py').write_text(code)
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def accuracy(ids,pred,labels):return sum(pred.get(u,'')==labels[u] for u in ids)/len(ids)
def evaluate(g,tag,selected,rows,adapter,gpu):
 d=OUT/'qwen35/eval'/g/tag;d.mkdir(parents=True,exist_ok=True)
 source_data=OUT/'qwen35/input'/g/'all';write(source_data/'data.json',rows)
 allpred={};alltime={}
 for precision in sorted({c.dtype for c in selected}):
  cs=[c for c in selected if c.dtype==precision];dataset=d/precision/'dataset';combined=[]
  for c in cs:
   # Isolate transform caches by configuration: uniform inputs must not share resized images.
   transformed=q.prepare_dataset(source_dir=source_data,output_dir=OUT/'qwen35/transform'/g/c.lambda_id,dataset_key='val_test',config=c,overwrite=False)
   for item in json.loads((transformed/'data.json').read_text()):
    item['uid']=c.lambda_id+'|'+item['uid'];combined.append(item)
  write(dataset/'data.json',combined);ev=d/precision/'output';ev.mkdir(parents=True,exist_ok=True)
  raw=ev/'raw_predictions.jsonl';complete=ev/'complete.json'
  if not complete.exists():
   cmd=[str(PYTHON),str(OUT/'timed_candidate_eval.py'),'--dataset-dir',str(dataset),'--model-path','@@WORKSPACE@@/qwen_eval/models/Qwen3.5-0.8B','--model-name',tag,'--family','hf_image_text','--out-dir',str(ev),'--task-ids','930,931,932','--dtype',precision,'--batch-size','1','--enable-thinking','false','--answer-source','raw','--mcq-max-new-tokens','8','--multi-mcq-max-new-tokens','16','--resume']
   if adapter:cmd+=['--adapter-path',str(adapter)]
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu;env['OMP_NUM_THREADS']='4';env['TOKENIZERS_PARALLELISM']='false'
   print('EVAL',g,tag,precision,len(combined),flush=True)
   with (ev/'eval.log').open('w') as log:subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   actual=[json.loads(l) for l in raw.read_text().splitlines()];assert len(actual)==len(combined) and len({r['uid'] for r in actual})==len(combined)
   write(complete,{'expected':len(combined),'actual':len(actual)})
  for r in map(json.loads,raw.read_text().splitlines()):
   lid,uid=r['uid'].split('|',1);allpred.setdefault(lid,{})[uid]=q.normalize(r.get('pseudo_label')) if r['accepted'] else ''
   if r.get('prediction_elapsed_s'):alltime.setdefault(lid,[]).append(float(r['prediction_elapsed_s']))
 return allpred,alltime

def work(pair):
 g,gpu=pair;d=OUT/'qwen35'/g
 if (d/'bundle.json').exists():print('REUSE',g,flush=True);return
 gr=q.load_group_rows(splitroot,pseudoroot,g);rows=gr['R_val']+gr['R_test'];val_ids=[r['uid'] for r in gr['R_val']];test_ids=[r['uid'] for r in gr['R_test']]
 true={r['uid']:q.normalize(r.get('ground_truth') or r.get('gt')) for r in rows};pseudo={r['uid']:q.normalize(r.get('ground_truth') or r.get('gt')) for r in gr['R_val_pseudo']}
 train_ids=[r['uid'] for r in json.loads((splitroot/'groups'/g/'part3_retraining/R_train/data.json').read_text())]
 assert not(set(train_ids)&set(val_ids) or set(train_ids)&set(test_ids) or set(val_ids)&set(test_ids))
 new=[c for c in configs if c.lambda_id not in source_lambdas];newpred,newtime=evaluate(g,'base_new_inputs',new,rows,None,gpu)
 lambdas=[]
 for c in configs:
  lid=c.lambda_id
  if lid in source_lambdas:
   ps=q.prediction_map(Path(bases[g,lid]['raw_predictions']));lat=float(source_lambdas[lid]['latency_mean_s'])
  else:ps=newpred[lid];lat=statistics.mean(newtime[lid][1:])
  assert all(u in ps for u in val_ids+test_ids),(g,lid,'base missing')
  lambdas.append({'id':lid,'throughput':1/lat,'latency_s':lat,'val_accuracy':accuracy(val_ids,ps,pseudo),'predictions':ps,'config':asdict(c),'source':'retained_original' if lid in source_lambdas else 'new_lower_actual_input'})
 gammas=[]
 for rank,steps in [(4,5),(4,10),(8,10),(8,20)]:
  tag=f'plain_r{rank}_s{steps}';rt=json.loads((OUT/'qwen35/train'/g/tag/'runtime.json').read_text())
  ps,_=evaluate(g,tag,configs,rows,Path(rt['adapter']),gpu)
  for c in configs:gammas.append({'id':tag,'lambda_id':c.lambda_id,'train_time_s':rt['train_runtime_s'],'val_accuracy':accuracy(val_ids,ps[c.lambda_id],pseudo),'predictions':ps[c.lambda_id],'rank':rank,'steps':steps,'source':'new_light','adapter':rt['adapter']})
 # Retain genuine old 80-step checkpoints, without treating extrapolated targets as trained models.
 for rank in [8,16,32]:
  gid=f'qwen35_lora_r{rank}_a{rank*2}_profile80_target320';prof=next(r for r in gprofiles if r['group_id']==g and r['gamma_id']==gid)
  for c in configs:
   if (g,c.lambda_id,gid) not in adapters:continue
   ps=q.prediction_map(Path(adapters[g,c.lambda_id,gid]['raw_predictions']))
   gammas.append({'id':f'original_r{rank}_s80','lambda_id':c.lambda_id,'train_time_s':float(prof['microprofile_train_time_s']),'val_accuracy':accuracy(val_ids,ps,pseudo),'predictions':ps,'rank':rank,'steps':80,'source':'retained_measured80','adapter':prof['microprofile_adapter_path']})
 fixed=next(c for c in configs if c.model_precision=='bf16' and c.roi_strategy=='full')
 rt=json.loads((OUT/'qwen35/train'/g/'ewc_r4_s10/runtime.json').read_text());ps,_=evaluate(g,'ewc_r4_s10',[fixed],rows,Path(rt['adapter']),gpu)
 ewc={'id':'ewc_r4_s10','lambda_id':fixed.lambda_id,'train_time_s':rt['train_runtime_s']+rt['fisher_runtime_s'],'val_accuracy':accuracy(val_ids,ps[fixed.lambda_id],pseudo),'predictions':ps[fixed.lambda_id],'fisher_time_s':rt['fisher_runtime_s']}
 write(d/'bundle.json',{'family':'qwen35','group_id':g,'seed':44,'test_ids':test_ids,'val_ids':val_ids,'train_ids':train_ids,'true':true,'pseudo':pseudo,'fixed_lambda':fixed.lambda_id,'fixed_gamma':'plain_r4_s10','lambdas':lambdas,'gammas':gammas,'ewc':ewc,'upstream_delay_s':3.})
 print('COMPLETE',g,flush=True)
groups=sorted({r['group_id'] for r in gprofiles})
with ThreadPoolExecutor(max_workers=1) as pool:list(pool.map(work,zip(groups,[os.environ.get('EXPERIMENT_GPU','2')]*2)))
