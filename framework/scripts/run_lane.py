from pathlib import Path
import subprocess,sys,os,json,time
R=Path(__file__).parent
lane=sys.argv[1]
for seed in [43,44]:
 if lane=='qwen25':
  commands=[[sys.executable,str(R/f'seed{seed}/qwen25/run.py'),mode,'4'] for mode in ['plain','ewc']]
 else:
  commands=[['@@VLM_PYTHON@@',str(R/f'seed{seed}/three_models'/name)] for name in ['train_qwen35.py','eval_qwen35.py']]
 for cmd in commands:
  print(time.ctime(),cmd,flush=True)
  subprocess.run(cmd,check=True,env=os.environ.copy())
print(lane,'ALL SEEDS COMPLETE',flush=True)
