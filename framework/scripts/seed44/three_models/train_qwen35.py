from pathlib import Path
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
import csv,json,os,sys,importlib.util,subprocess
ROOT=Path('@@WORKSPACE@@/drift_lora_project');OUT=Path(__file__).parent
PYTHON=Path('@@VLM_PYTHON@@')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
u=load('ewc_utils',ROOT/'scripts/run_qwen_student_ewc_baselines.py')
profiles=list(csv.DictReader((ROOT/'results/all_model_three_seed_repeats_20260715/seed44/qwen35/gamma/gamma_smallstep_by_config.csv').open()))
ewc={r['group_id']:r for r in csv.DictReader((ROOT/'results/all_model_same_start_ewc_gpu4_20260722/seed44/ewc/qwen35/qwen35_ewc_actual_execution.csv').open())}
args=Namespace(ewc_mode='ba_kv',ewc_anchor_mode='initial_zero',ewc_lambda=10.,ewc_ba_target_modules='k_proj,v_proj',ewc_normalize='mean',ewc_reduction='mean')
keys=['model_type','tuner_type','torch_dtype','freeze_llm','freeze_vit','freeze_aligner','target_modules','lora_dropout','learning_rate','gradient_checkpointing','warmup_ratio','lr_scheduler_type','weight_decay','max_grad_norm','max_length','attn_impl','optim','adam_beta1','adam_beta2','adam_epsilon','max_pixels']
def work(pair):
 prof,gpu=pair;g=prof['group_id'];old=json.loads((Path(prof['microprofile_adapter_path'])/'args.json').read_text());old['max_pixels']=262144
 for kind,rank,steps in [('plain',4,5),('plain',4,10),('plain',8,10),('plain',8,20),('ewc',4,10)]:
  d=OUT/'qwen35/train'/g/f'{kind}_r{rank}_s{steps}';d.mkdir(parents=True,exist_ok=True)
  cmd=[str(PYTHON),'-m','swift.cli.sft'] if kind=='plain' else [str(PYTHON),str(ROOT/'scripts/run_swift_sft_ewc.py')]
  cmd+=['--model','@@WORKSPACE@@/qwen_eval/models/Qwen3.5-0.8B','--dataset',*old['dataset'],'--val_dataset',*old['val_dataset'],'--output_dir',str(d/'swift_output')]
  for k in keys:
   v=old.get(k)
   if v is not None:cmd+=['--'+k]+([str(x) for x in v] if isinstance(v,list) else [str(v).lower() if isinstance(v,bool) else str(v)])
  cmd+=['--lora_rank',str(rank),'--lora_alpha',str(rank*2),'--per_device_train_batch_size','4','--gradient_accumulation_steps','2','--per_device_eval_batch_size','1','--max_steps',str(steps),'--save_steps',str(steps),'--eval_steps',str(steps),'--save_total_limit','1','--logging_steps','5','--report_to','none','--seed','44','--dataloader_num_workers','1']
  fisher=Path(ewc[g]['fisher_path']) if kind=='ewc' else None
  env=u.ewc_env(os.environ.copy(),args,gpu_id=gpu,fisher=fisher);env['MAX_PIXELS']=str(old['max_pixels']);env['TOKENIZERS_PARALLELISM']='false';env['OMP_NUM_THREADS']='4'
  (d/'command.json').write_text(json.dumps({'command':cmd,'gpu':gpu,'fisher':str(fisher) if fisher else None,'seed':44},indent=2))
  print('TRAIN',g,kind,steps,'GPU',gpu,flush=True)
  adapter=u.latest_checkpoint(d/'swift_output',steps)
  if adapter is None:
   with (d/'train.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
   adapter=u.latest_checkpoint(d/'swift_output',steps)
  runtime=u.parse_train_runtime(d/'swift_output');assert runtime>0 and adapter
  (d/'runtime.json').write_text(json.dumps({'group_id':g,'kind':kind,'steps':steps,'rank':rank,'alpha':rank*2,'max_pixels':262144,'batch_size':4,'gradient_accumulation':2,'effective_batch_size':8,'adapter':str(adapter),'train_runtime_s':runtime,'fisher_runtime_s':float(ewc[g]['fisher_runtime_s']) if fisher else 0.,'fisher_source':str(fisher) if fisher else None,'train_data':old['dataset'],'val_data':old['val_dataset']},indent=2))
  print('DONE',g,kind,steps,runtime,flush=True)
profs=sorted([r for r in profiles if r['gamma_id']=='qwen35_lora_r8_a16_profile80_target320'],key=lambda r:r['group_id'])
with ThreadPoolExecutor(max_workers=1) as pool:list(pool.map(work,zip(profs,[os.environ.get('EXPERIMENT_GPU','2')]*2)))
