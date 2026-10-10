from pathlib import Path
import csv,json,math,statistics,importlib.util,sys
ROOT=Path('@@WORKSPACE@@/drift_lora_project');OUT=Path(__file__).parent
FACTORS=[.1,.25,.5,.75,1.,2.,4.];PERIOD=5
REFS={'qwen35':3.718627525182433*.25,'remoteclip':11.092010981689922*.25,'resnet18':11.131380602097332*.25}
METHODS=['NoRetrain','Periodic','Static','EWC','SADA']
def write(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def accuracy(us,pred,truth):return sum(pred.get(u,'')==truth[u] for u in us)/len(us)
def choose(b,rate,horizon):
 # Coverage/readiness objective and admission thresholds from the Qwen2.5 pilot,
 # using measured, lambda-specific validation gains instead of step extrapolation.
 # Partial coverage remains admissible, as in the original OmniEarth schedulers.
 lam=b['lambdas'];gam=b['gammas'];delay=b['upstream_delay_s']
 candidates=[];grid={i/100 for i in range(1,101)}
 grid.update(min(1.,rate/l['throughput']) for l in lam)
 for l in lam:
  for inf in sorted(grid):
   cost=rate/l['throughput']
   cov=min(1.,inf/cost);base=l['val_accuracy']*cov
   candidates.append({'lambda_id':l['id'],'gamma_id':'','I':inf,'R':1-inf,'coverage':cov,'ready_s':None,'estimate':base,'future_estimate':base,'validation_base':l['val_accuracy'],'validation_post':l['val_accuracy']})
   if inf>=1:continue
   for g in gam:
    if g['lambda_id']!=l['id']:continue
    finish=g['train_time_s']/(1-inf);active=max(0.,1-(finish+delay)/horizon)
    estimate=cov*(min(1.,finish/horizon)*l['val_accuracy']+max(0.,1-finish/horizon)*g['val_accuracy'])
    if active<.2 or estimate-base<.005:continue
    candidates.append({'lambda_id':l['id'],'gamma_id':g['id'],'I':inf,'R':1-inf,'coverage':cov,'ready_s':delay+finish,'estimate':estimate,'future_estimate':cov*g['val_accuracy'],'validation_base':l['val_accuracy'],'validation_post':g['val_accuracy']})
 if not candidates:
  # Coverage-aware fallback, only if no configuration serves every arrival at full resource.
  for l in lam:
   cov=min(1.,l['throughput']/rate);candidates.append({'lambda_id':l['id'],'gamma_id':'','I':1.,'R':0.,'coverage':cov,'ready_s':None,'estimate':cov*l['val_accuracy'],'future_estimate':cov*l['val_accuracy'],'validation_base':l['val_accuracy'],'validation_post':l['val_accuracy']})
 return max(candidates,key=lambda c:(c['estimate'],c['future_estimate'],c['R'],c['lambda_id'],c['gamma_id']))
def evaluate(b,choice,rate,method):
 us=b['test_ids'];n=len(us);true=b['true'];lm={l['id']:l for l in b['lambdas']};fixed=lm[b['fixed_lambda']];l=lm[choice['lambda_id']]
 pre=fixed['predictions'];base=l['predictions'];post=base
 if choice['gamma_id']:
  g=b['ewc'] if method=='EWC' else next(g for g in b['gammas'] if g['lambda_id']==l['id'] and g['id']==choice['gamma_id'])
  post=g['predictions']
 delay=b['upstream_delay_s'] if method!='NoRetrain' and choice['gamma_id'] else 0.
 di=min(n,math.ceil(delay*rate));ready=choice['ready_s'];ri=min(n,math.ceil(ready*rate)) if ready is not None else n
 ri=max(di,ri);c0=min(1.,fixed['throughput']/rate);cov=choice['coverage']
 correct=lambda ps,ids:sum(ps.get(u,'')==true[u] for u in ids)
 score=(c0*correct(pre,us[:di])+cov*(correct(base,us[di:ri])+correct(post,us[ri:])))/n
 return score,n-ri,ri
paths=list((OUT/'qwen35').glob('*/bundle.json'))+[OUT/f/'bundle.json' for f in ['remoteclip','resnet18']]
bundles=[json.loads(p.read_text()) for p in paths if p.exists()]
assert len(bundles)==4,[(p,p.exists()) for p in paths]
details=[];choices=[];configuration=[];audits=[]
for b in bundles:
 fam=b['family'];g=b['group_id'];lm={l['id']:l for l in b['lambdas']};fixed=lm[b['fixed_lambda']]
 train=set(b['train_ids']);val=set(b['val_ids']);test=set(b['test_ids']);assert not(train&val or train&test or val&test)
 for l in b['lambdas']:
  assert set(b['test_ids'])<=set(l['predictions'])
  configuration.append({'模型':fam,'组ID':g,'类型':'推理','配置ID':l['id'],'推理配置ID':l['id'],'耗时秒':l['latency_s'],'验证准确率':l['val_accuracy'],'来源':l['source']})
 for t in b['gammas']+[b['ewc']]:
  assert t['train_time_s']>0 and test<=set(t['predictions'])
  measured=accuracy(b['val_ids'],t['predictions'],b['pseudo']);assert abs(measured-t['val_accuracy'])<1e-12
  configuration.append({'模型':fam,'组ID':g,'类型':'重训练','配置ID':t['id'],'推理配置ID':t['lambda_id'],'耗时秒':t['train_time_s'],'验证准确率':t['val_accuracy'],'来源':t.get('source','ewc')})
 for f in FACTORS:
  rate=REFS[fam]*f;horizon=len(test)/rate
  for method in METHODS:
   if method=='SADA':c=choose(b,rate,horizon)
   elif method=='NoRetrain':c={'lambda_id':fixed['id'],'gamma_id':'','I':1.,'R':0.,'coverage':min(1.,fixed['throughput']/rate),'ready_s':None}
   else:
    t=b['ewc'] if method=='EWC' else next(x for x in b['gammas'] if x['id']==b['fixed_gamma'] and x['lambda_id']==fixed['id'])
    c={'lambda_id':fixed['id'],'gamma_id':t['id'],'I':.5,'R':.5,'coverage':min(1.,.5*fixed['throughput']/rate),'ready_s':b['upstream_delay_s']+t['train_time_s']/.5+(PERIOD/rate if method=='Periodic' else 0)}
   score,post,ri=evaluate(b,c,rate,method)
   details.append({'模型':fam,'种子':42,'组ID':g,'负载倍数':f,'方法':method,'窗口准确率':score,'测试样本数':len(test),'适配器评估样本数':post,'到达率每秒':rate,'窗口秒':horizon,'就绪秒':c['ready_s'],'推理配置ID':c['lambda_id'],'重训练配置ID':c['gamma_id'],'推理资源份额':c['I'],'覆盖率':c['coverage'],'等待样本数':PERIOD if method=='Periodic' else 0})
   if method=='SADA':choices.append({'模型':fam,'组ID':g,'负载倍数':f,**{'推理配置ID':c['lambda_id'],'重训练配置ID':c['gamma_id'],'推理资源份额':c['I'],'验证估计效用':c['estimate'],'窗口准确率':score,'适配器评估样本数':post}})
 audits.append({'family':fam,'group':g,'train':len(train),'val':len(val),'test':len(test),'lambda_count':len(b['lambdas']),'gamma_pairs':len(b['gammas']),'fixed_gamma':b['fixed_gamma'],'periodic_samples':5,'split_overlap':0})
write('分组结果.csv',details);write('SADA配置选择.csv',choices);write('候选配置及实测耗时.csv',configuration)
table=[];active=[]
for fam in ['qwen35','remoteclip','resnet18']:
 for method in METHODS:
  row={'模型':fam,'方法':method};ar=dict(row)
  for f in FACTORS:
   rs=[r for r in details if r['模型']==fam and r['方法']==method and r['负载倍数']==f]
   row[f'{f:g}×']=sum(r['窗口准确率']*r['测试样本数'] for r in rs)/sum(r['测试样本数'] for r in rs)
   ar[f'{f:g}×']=sum(r['适配器评估样本数']>0 for r in rs)
  table.append(row);active.append(ar)
write('结果表.csv',table);write('窗口内启用组数.csv',active)
(OUT/'validation.json').write_text(json.dumps({'seed':42,'periodic_wait_samples':PERIOD,'reference_scale':.25,'load_factors':FACTORS,'groups':audits,'group_method_load_rows':len(details),'no_test_labels_in_selection':True,'new_training_and_prediction_artifacts':True,'runtime_scope':'measured shared-hardware profiles plus simulated resource partition; not concurrent flight hardware execution'},indent=2))
for r in table:print(r['模型'],r['方法'],' | '.join(f'{r[f"{f:g}×"]:.3f}' for f in FACTORS))
