from pathlib import Path
from argparse import Namespace
import importlib.util,sys,os,json,csv,subprocess,time,fcntl
ROOT=Path('@@WORKSPACE@@/drift_lora_project');OUT=Path(__file__).parent
PYTHON=Path('@@QWEN25_PYTHON@@');MODEL=Path('@@WORKSPACE@@/qwen_eval/models/Qwen2.5-VL-3B-Instruct')
OLD=ROOT/'results/qwen25_light_fixed_trial_20260920';LIGHT=ROOT/'results/qwen25_target80_admission_gate_3s_20260914/seed43'
TRUE=ROOT/'results/v2_teacher_closed_loop_20260630_181348/step5_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924/inputs/true_5group_rwindow_root'
MODE=sys.argv[1];GPU=sys.argv[2];assert MODE in ['plain','ewc']
profiles=sorted([r for r in json.loads((LIGHT/'target80_gamma_profiles.json').read_text()) if r['lora_rank']==8],key=lambda r:r['group_id'])
spec=importlib.util.spec_from_file_location('ewc_util',ROOT/'scripts/run_qwen_student_ewc_baselines.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
eargs=Namespace(ewc_mode='ba_kv',ewc_anchor_mode='initial_zero',ewc_lambda=10.,ewc_ba_target_modules='k_proj,v_proj',ewc_normalize='mean',ewc_reduction='mean')
keys=['model_type','tuner_type','torch_dtype','freeze_llm','freeze_vit','freeze_aligner','target_modules','lora_rank','lora_alpha','lora_dropout','learning_rate','per_device_train_batch_size','per_device_eval_batch_size','gradient_accumulation_steps','gradient_checkpointing','warmup_ratio','lr_scheduler_type','weight_decay','max_grad_norm','max_length','attn_impl','dataloader_num_workers','dataset_num_proc','optim','adam_beta1','adam_beta2','adam_epsilon','max_pixels']
for prof in profiles:
 g=prof['group_id'];d=OUT/MODE/g;d.mkdir(parents=True,exist_ok=True)
 lock=(d/'.run.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  lock.close();print('CLAIMED BY OTHER WORKER',MODE,g,flush=True);continue
 if (d/'runtime.json').exists() and (d/'eval/strict_eval_report.json').exists():
  lock.close();print('ALREADY COMPLETE',MODE,g,flush=True);continue
 print(MODE,g,'start GPU',GPU,flush=True)
 if MODE=='plain': source_args=Path(prof['checkpoint_path'])/'args.json';rt0=None
 else:rt0=json.loads((OLD/'ewc'/g/'runtime.json').read_text());source_args=Path(rt0['adapter'])/'args.json'
 old=json.loads(source_args.read_text());old['dataset']=[p.replace('/seed42/', '/seed43/') for p in old['dataset']];old['val_dataset']=[p.replace('/seed42/', '/seed43/') for p in old['val_dataset']];assert old['max_steps']==80 and old['lora_rank']==8
 assert old['per_device_train_batch_size']*old['gradient_accumulation_steps']==8
 command=[str(PYTHON),'-m','swift.cli.sft'] if MODE=='plain' else [str(PYTHON),str(ROOT/'scripts/run_swift_sft_ewc.py')]
 command+=['--model',str(MODEL),'--dataset',*old['dataset'],'--val_dataset',*old['val_dataset'],'--output_dir',str(d/'swift_output')]
 for key in keys:
  value=old.get(key)
  if value is None:continue
  command+=['--'+key]+([str(v) for v in value] if isinstance(value,list) else [str(value).lower() if isinstance(value,bool) else str(value)])
 command+=['--max_steps','120','--eval_steps','80','--save_steps','120','--save_total_limit','1','--logging_steps','10','--report_to','none','--seed','43']
 fisher=Path(rt0['fisher_source']) if rt0 else None

 if rt0:
  plans={r['group_id']:r for r in csv.DictReader((ROOT/'results/all_model_same_start_ewc_gpu4_20260722/seed43/ewc/qwen25/ewc_training_plan_and_results.csv').open())}
  fisher=Path(plans[g]['fisher']);rt0['fisher_runtime_s']=float(plans[g]['fisher_runtime_s'])
 env=mod.ewc_env(os.environ.copy(),eargs,gpu_id=GPU,fisher=fisher)
 if MODE=='plain':
  metadata=json.loads((Path(old['dataset'][0]).parent/'experiment_metadata.json').read_text());env['MAX_PIXELS']=str(metadata['recommended_max_pixels'])
 elif old.get('max_pixels'):env['MAX_PIXELS']=str(old['max_pixels'])
 env['PYTHONPATH']=str(ROOT/'remote_edit')+':'+str(ROOT)+':'+env.get('PYTHONPATH','')
 (d/'training_config.json').write_text(json.dumps({'mode':MODE,'seed':43,'target_steps':120,'source_80step_args':str(source_args),'command':command,'gpu':GPU,'MAX_PIXELS':env.get('MAX_PIXELS'),'fisher':str(fisher) if fisher else None},indent=2))
 adapter=mod.latest_checkpoint(d/'swift_output',120)
 if adapter is None:
  (d/'gpu_before_train.csv').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv'],text=True))
  with (d/'train.log').open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  adapter=mod.latest_checkpoint(d/'swift_output',120);assert adapter is not None
 runtime=mod.parse_train_runtime(d/'swift_output');assert runtime>0
 (d/'runtime.json').write_text(json.dumps({'group_id':g,'mode':MODE,'adapter':str(adapter),'train_runtime_s':runtime,'fisher_runtime_s':rt0['fisher_runtime_s'] if rt0 else 0.,'fisher_source':str(fisher) if fisher else None,'seed':43,'target_steps':120,'effective_batch_size':8,'gpu':GPU},indent=2))
 ev=d/'eval';ev.mkdir(exist_ok=True)
 if not (ev/'strict_eval_report.json').exists():
  cmd=[str(PYTHON),'-m','remote_sensing_adaptation.eval_qwen_lora_strict','--dataset-dir',str(TRUE/'groups'/g/'R_test'),'--model-path',str(MODEL),'--adapter-path',str(adapter),'--out-dir',str(ev),'--task-ids','1,3,4,5,6,7,8,9','--max-new-tokens','16','--dtype','float32','--cuda-visible-devices',GPU,'--seed','43','--answer-source','raw','--batch-size','1','--progress-every','40']
  with (ev/'eval.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(MODE,g,'complete',runtime,flush=True)
 lock.close()
print(MODE,'ALL COMPLETE',flush=True)
