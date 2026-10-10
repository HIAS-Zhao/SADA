from pathlib import Path
from collections import defaultdict
from argparse import Namespace
import json,csv,math,importlib.util,sys,statistics
ROOT=Path('@@WORKSPACE@@/drift_lora_project');OUT=Path(__file__).parent;R=ROOT/'results'
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
sel=load('repeat_sel',OUT/'protocol_snapshot/custom_selector.py');summ=load('repeat_summ',OUT/'protocol_snapshot/custom_summarizer.py')
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def csvwrite(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
cache={}
def pred(path):
 path=Path(path)
 if path not in cache:
  r=json.loads(path.read_text());p=Path(r['predictions']);p=p if p.is_absolute() else ROOT/p;cache[path]={x['uid']:x for x in map(json.loads,p.read_text().splitlines())}
 return cache[path]
def valid(path,us):return path is not None and Path(path).exists() and set(us)<=set(pred(path))
def count(p,us):assert set(us)<=set(p);return sum(bool(p[u].get('accepted')) and bool(p[u].get('correct')) for u in us)
mode=sys.argv[1] if len(sys.argv)>1 else 'plan';missing=[];details=[];selection=[];reference=sel.REFERENCE_THROUGHPUT*.25
for seed in ([int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else [42,43,44]):
 original=sel.SOURCE_ROOT/f'seed{seed}/qwen25/step6';base,adapters=summ.index_reports(original)
 for path in [original/'true_accuracy_full_r/ar0p05',sel.DEFAULT_EXPERIMENT_ROOT/f'seed{seed}/true_accuracy_target80_rtest',sel.DEFAULT_EXPERIMENT_ROOT/f'seed{seed}/true_accuracy_target80_selected_rtest',R/'qwen25_lower_load_reference_trial_20260920' if seed==42 else OUT/f'seed{seed}/qwen25/additional']:
  b,a=summ.index_reports(path)
  if path==original/'true_accuracy_full_r/ar0p05':base.update(b)
  else:
   for key,value in b.items():base.setdefault(key,value)
  adapters.update(a)
 lambdas=defaultdict(list);gammas=defaultdict(list)
 for r in sel.read_json(sel.LAMBDA_PROFILES):lambdas[r['group_id']].append(r)
 light=sel.read_json(sel.DEFAULT_EXPERIMENT_ROOT/f'seed{seed}/target80_gamma_profiles.json')
 for r in sel.read_json(sel.SOURCE_ROOT/f'seed{seed}/qwen25/step5/gamma_pareto/gamma_group_profiles_pareto.json')+light:gammas[r['group_id']].append(r)
 native={(r['组ID'],r['方法代码']):r for r in summ.closest_rate_rows(summ.source_detail_by_rate(seed),reference)}
 for factor in [.1,.25,.5,.75,1.,2.,4.]:
  rate=reference*factor
  for group in sorted(lambdas):
   us=[r['uid'] for r in json.loads((summ.TRUE_ROOT/'groups'/group/'R_test/data.json').read_text())];nr=native[group,'NoRetrain'];st=native[group,'StaticSplitContinuous'];lm={r['lambda_id']:r for r in lambdas[group]}
   c0=min(1,float(lm[nr['推理配置ID']]['throughput_samples_per_sec'])/rate);c=min(1,.5*float(lm[st['推理配置ID']]['throughput_samples_per_sec'])/rate)
   ch=sel.select_group(lambdas=lambdas[group],gammas=gammas[group],arrival_rate=rate,horizon_s=110/rate,fallback=False,upstream_delay_s=3,min_active_fraction=.2,min_gain=.005)
   selection.append({'seed':seed,'factor':factor,'group':group,**ch})
   for typ,key in [('base',(group,ch['lambda_id']))]+([('adapter',(group,ch['gamma_id'],ch['lambda_id']))] if ch['gamma_id'] else []):
    table=base if typ=='base' else adapters
    if not valid(table.get(key),us):
     dest=OUT/f'seed{seed}/qwen25/additional'/typ/Path(*key)/'strict_eval_report.json';table[key]=dest
     if not valid(dest,us):
      req={'seed':seed,'type':typ,'group':group,'lambda_id':ch['lambda_id'],'gamma_id':ch['gamma_id'] if typ=='adapter' else '', 'report':str(dest),'lambda_profile':lm[ch['lambda_id']]}
      if typ=='adapter':
       gr=next(g for g in gammas[group] if g['gamma_id']==ch['gamma_id']);req['adapter']=gr.get('checkpoint_path','')
      if req not in missing:missing.append(req)
   if mode!='score':continue
   pre=pred(nr['base报告']);bp=pred(st['base报告'])
   for method in ['NoRetrain','Periodic','Static','EWC','SADA']:
    di=min(110,math.ceil(3*rate));ri=110;ready=None;cov=c;inf=.5;lid=st['推理配置ID'];gid=''
    if method=='NoRetrain':value=c0*count(pre,us)/110;cov=c0;inf=1.;lid=nr['推理配置ID']
    elif method=='SADA':
     lid=ch['lambda_id'];gid=ch['gamma_id'];cov=ch['coverage'];inf=ch['I'];pbase=pred(base[group,lid])
     if gid:
      ready=3+ch['finish_time_s'];ri=min(110,math.ceil(rate*ready));post=pred(adapters[group,gid,lid]);value=(c0*count(pre,us[:di])+cov*(count(pbase,us[di:ri])+count(post,us[ri:])))/110
     else:value=cov*count(pbase,us)/110
    else:
     kind='ewc' if method=='EWC' else 'plain';d=(R/'qwen25_target120_period20_trial_20260921' if seed==42 else OUT/f'seed{seed}/qwen25')/kind/group
     rt=json.loads((d/'runtime.json').read_text());post=pred(d/'eval/strict_eval_report.json');ready=3+(rt['train_runtime_s']+rt['fisher_runtime_s'])/.5+(20/rate if method=='Periodic' else 0);ri=min(110,math.ceil(rate*ready));gid='qwen25vl3b_lora_r8_a16_target120'
     value=(c0*count(pre,us[:di])+cov*(count(bp,us[di:ri])+count(post,us[ri:])))/110
    details.append({'模型':'qwen25','种子':seed,'组ID':group,'负载倍数':factor,'方法':method,'窗口准确率':value,'测试样本数':110,'适配器评估样本数':110-ri,'就绪秒':ready,'推理资源份额':inf,'覆盖率':cov,'推理配置ID':lid,'重训练配置ID':gid})
write(OUT/'qwen25_missing_predictions.json',missing);write(OUT/'qwen25_selections.json',selection)
if mode=='score':csvwrite(OUT/'qwen25_分组结果.csv',details)
print('Missing prediction requests:',len(missing));print([(r['seed'],r['group'],r['gamma_id'],r['lambda_id']) for r in missing])
