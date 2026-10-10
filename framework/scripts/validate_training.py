from pathlib import Path
import json,hashlib
R=Path(__file__).parent;audits=[]
for s in [43,44]:
 paths=list((R/f'seed{s}/qwen25').glob('*/*/runtime.json'));assert len(paths)==10
 for path in paths:
  rt=json.loads(path.read_text());a=json.loads((Path(rt['adapter'])/'args.json').read_text());saved=json.loads((path.parent/'training_config.json').read_text());old=json.loads(Path(saved['source_80step_args']).read_text())
  assert a['seed']==s and a['max_steps']==120 and a['lora_rank']==8 and a['lora_alpha']==16
  assert a['per_device_train_batch_size']*a['gradient_accumulation_steps']==8
  ref_rt=json.loads((R.parent/'qwen25_target120_period20_trial_20260921'/path.parent.relative_to(R/f'seed{s}/qwen25')/'runtime.json').read_text());ref=json.loads((Path(ref_rt['adapter'])/'args.json').read_text())
  for key in ['lora_dropout','learning_rate','max_length','attn_impl','per_device_train_batch_size','gradient_accumulation_steps','warmup_ratio','weight_decay','target_modules','freeze_llm','freeze_vit','freeze_aligner']:
   assert a.get(key)==ref.get(key),(path,key,'differs from seed42')
  for k in ['lora_dropout','learning_rate','gradient_checkpointing','max_length','attn_impl','per_device_train_batch_size','gradient_accumulation_steps']:
   assert a.get(k)==old.get(k),(path,k,a.get(k),old.get(k))
  for k in ['dataset','val_dataset']:
   for p,q in zip(a[k],old[k]):assert hashlib.sha256(Path(p).read_bytes()).digest()==hashlib.sha256(Path(q).read_bytes()).digest(),(path,k)
  report=json.loads((path.parent/'eval/strict_eval_report.json').read_text());assert report['total']==110
  audits.append({'family':'qwen25','seed':s,'path':str(path),'steps':120,'rank':8,'test_count':report['total'],'accepted':report['accepted'],'runtime_s':rt['train_runtime_s']})
 paths=list((R/f'seed{s}/three_models/qwen35/train').glob('*/*/runtime.json'));assert len(paths)==10
 for p in paths:
  rt=json.loads(p.read_text());a=json.loads((Path(rt['adapter'])/'args.json').read_text());assert a['seed']==s and a['max_steps']==rt['steps'] and a['lora_rank']==rt['rank'] and a['lora_alpha']==2*rt['rank'];assert a['per_device_train_batch_size']*a['gradient_accumulation_steps']==8
  audits.append({'family':'qwen35','seed':s,'path':str(p),'steps':rt['steps'],'rank':rt['rank'],'runtime_s':rt['train_runtime_s']})
 for fam in ['remoteclip','resnet18']:
  b=json.loads((R/f'seed{s}/three_models'/fam/'bundle.json').read_text());assert b['seed']==s and len(b['lambdas'])=={'remoteclip':8,'resnet18':11}[fam];assert len(b['gammas'])==7*len(b['lambdas'])
  for p in (R/f'seed{s}/three_models'/fam/'train').glob('*/*/metrics.json'):
   m=json.loads(p.read_text());assert m['seed']==s and f'light{s}:' in m['paired_seed'];assert (p.parent/'head.npz').exists() and (p.parent/'predictions.json').exists()
(R/'training_validation.json').write_text(json.dumps({'new_qwen_checkpoints':len(audits),'seeds':[43,44],'audits':audits,'specialist_training_seed_checks':True},indent=2));print('Training validation passed:',len(audits),'new Qwen checkpoints')
