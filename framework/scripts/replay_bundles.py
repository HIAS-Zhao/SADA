from pathlib import Path
import json,csv,copy,statistics
OUT=Path(__file__).parent;R=OUT.parent;PREV=R/'three_models_light_period5_expanded_20260921';HIGH=R/'resnet18_heavier_inference_highload_20260921'
ns={'__file__':str(PREV/'replay.py')};exec((OUT/'protocol_snapshot/omniearth_replay.py').read_text().split('paths=list')[0],ns)
choose,evaluate=ns['choose'],ns['evaluate'];factors=ns['FACTORS'];refs={**ns['REFS'],'resnet18':11.131380602097332};methods=ns['METHODS']
def write(p,rows):
 if not rows:return
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
rows=[];audits=[]
for seed in [42,43,44]:
 root=PREV if seed==42 else OUT/f'seed{seed}/three_models'
 paths=list((root/'qwen35').glob('*/bundle.json'))+[root/'remoteclip/bundle.json',HIGH/'bundle.json' if seed==42 else root/'resnet18/bundle.json']
 for path in paths:
  if not path.exists():continue
  b=json.loads(path.read_text());fam=b['family'];train,val,test=[set(b[k]) for k in ['train_ids','val_ids','test_ids']];assert not(train&val or train&test or val&test)
  if fam=='qwen35':
   required=set(b['test_ids']+b['val_ids']);rejected_audit=[]
   for g in b['gammas']:
    absent=required-set(g['predictions'])
    if not absent:continue
    assert g['id'].startswith('original_r'),(path,g['id'],'unexpected missing new predictions')
    rank=int(g['id'].split('_r')[1].split('_')[0]);gid=f'qwen35_lora_r{rank}_a{rank*2}_profile80_target320'
    raw=R/f'all_model_three_seed_repeats_20260715/seed{seed}/qwen35/framework/eval_cache/adapter'/b['group_id']/g['lambda_id']/gid/'raw_predictions.jsonl'
    recorded={r['uid']:r for r in map(json.loads,raw.read_text().splitlines())}
    assert absent<=set(recorded),(raw,'unrecorded UIDs')
    assert all(not recorded[u].get('accepted') for u in absent),(raw,'missing accepted predictions')
    for u in absent:g['predictions'][u]=''
    rejected_audit.append({'gamma':g['id'],'lambda':g['lambda_id'],'explicit_rejected_count':len(absent),'raw':str(raw)})
   if rejected_audit:
    b['rejected_output_audit']=rejected_audit
    if seed!=42:path.write_text(json.dumps(b,ensure_ascii=False,indent=2))
  lm={l['id']:l for l in b['lambdas']};fixed=lm[b['fixed_lambda']]
  for l in b['lambdas']:
   assert test|val<=set(l['predictions']);assert abs(ns['accuracy'](b['val_ids'],l['predictions'],b['pseudo'])-l['val_accuracy'])<1e-12
  for g in b['gammas']+[b['ewc']]:
   assert test|val<=set(g['predictions']);assert g['train_time_s']>0;assert abs(ns['accuracy'](b['val_ids'],g['predictions'],b['pseudo'])-g['val_accuracy'])<1e-12
  fake=copy.deepcopy(b);fake['true']={u:'INVALID' for u in fake['true']}
  for f in factors:
   rate=refs[fam]*f;horizon=len(test)/rate;assert choose(fake,rate,horizon)==choose(b,rate,horizon)
   for method in methods:
    if method=='SADA':c=choose(b,rate,horizon)
    elif method=='NoRetrain':c={'lambda_id':fixed['id'],'gamma_id':'','I':1.,'R':0.,'coverage':min(1.,fixed['throughput']/rate),'ready_s':None}
    else:
     g=b['ewc'] if method=='EWC' else next(g for g in b['gammas'] if g['id']==b['fixed_gamma'] and g['lambda_id']==fixed['id'])
     c={'lambda_id':fixed['id'],'gamma_id':g['id'],'I':.5,'R':.5,'coverage':min(1.,.5*fixed['throughput']/rate),'ready_s':b['upstream_delay_s']+g['train_time_s']/.5+(5/rate if method=='Periodic' else 0)}
    score,post,ri=evaluate(b,c,rate,method)
    rows.append({'模型':fam,'种子':seed,'组ID':b['group_id'],'负载倍数':f,'方法':method,'窗口准确率':score,'测试样本数':len(test),'适配器评估样本数':post,'就绪秒':c['ready_s'],'覆盖率':c['coverage'],'推理配置ID':c['lambda_id'],'重训练配置ID':c['gamma_id'],'推理资源份额':c['I']})
  audits.append({'seed':seed,'family':fam,'group':b['group_id'],'test':len(test),'lambdas':len(lm),'gammas':len(b['gammas']),'split_disjoint':True,'test_label_invariance':True})
write(OUT/'omniearth_分组结果.csv',rows)
errors=[]
for fam,orig in [('qwen35',PREV/'结果表.csv'),('remoteclip',PREV/'结果表.csv'),('resnet18',HIGH/'结果表.csv')]:
 old=list(csv.DictReader(orig.open()))
 for method in methods:
  expected=next(r for r in old if r['方法']==method and (fam=='resnet18' or r['模型']==fam))
  for f in factors:
   rs=[r for r in rows if r['模型']==fam and r['种子']==42 and r['方法']==method and r['负载倍数']==f]
   actual=sum(r['窗口准确率']*r['测试样本数'] for r in rs)/sum(r['测试样本数'] for r in rs);errors.append(abs(actual-float(expected[f'{f:g}×'])))
assert max(errors)<1e-12
(OUT/'bundle_validation.json').write_text(json.dumps({'seed42_reproduction_cases':len(errors),'max_error':max(errors),'audits':audits},indent=2))
print('Completed bundles:',[(a['seed'],a['family'],a['group']) for a in audits]);print('seed42 reproduction:',len(errors),max(errors))
