from pathlib import Path
from argparse import Namespace
import importlib.util,sys,json,fcntl
ROOT=Path('@@WORKSPACE@@/drift_lora_project');OUT=Path(__file__).parent
s=importlib.util.spec_from_file_location('evalutil',ROOT/'remote_edit/remote_sensing_adaptation/evaluate_v2_scheduler_true_accuracy.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
TRUE=ROOT/'results/v2_teacher_closed_loop_20260630_181348/step5_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924/inputs/true_5group_rwindow_root'
pending=[]
for req in json.loads((OUT/'qwen25_missing_predictions.json').read_text()):
 seed=req['seed'];d=Path(req['report']).parent;d.mkdir(parents=True,exist_ok=True)
 lock=(d/'.run.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  lock.close();pending.append(req);continue
 if Path(req['report']).exists():
  lock.close();continue
 data=TRUE/'groups'/req['group']/'R_test';uidfile=d/'uids.jsonl';uidfile.parent.mkdir(parents=True,exist_ok=True);m.write_uid_filter(uidfile,data)
 adapter=Path(req['adapter']) if req['type']=='adapter' else None
 if adapter:assert (adapter/'adapter_model.safetensors').exists()
 args=Namespace(force=False,python_bin=Path('@@QWEN25_PYTHON@@'),model_path=Path('@@WORKSPACE@@/qwen_eval/models/Qwen2.5-VL-3B-Instruct'),task_ids='1,3,4,5,6,7,8,9',cuda_visible_devices=sys.argv[1] if len(sys.argv)>1 else '4',seed=seed,answer_source='raw',strict_eval_batch_size=1)
 print('EVAL MISSING',seed,req['group'],req['gamma_id'],req['lambda_id'],flush=True)
 l=req['lambda_profile'];m.run_strict_eval(args=args,dataset_dir=data,out_dir=d,uid_filter=uidfile,max_new_tokens=int(l.get('max_new_tokens') or 16),dtype=m.dtype_from_precision(l.get('model_precision',l.get('precision')),'bfloat16'),adapter_path=adapter)
 lock.close()
for req in pending:
 with (Path(req['report']).parent/'.run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert Path(req['report']).exists(),('Other evaluator failed',req['report'])
print('ALL COMPLETE',flush=True)
