from pathlib import Path
import csv,json,statistics,collections,subprocess,sys
OUT=Path(__file__).parent
FACTORS=[.1,.25,.5,.75,1.,2.,4.];FAMILIES=['qwen25','qwen35','remoteclip','resnet18'];METHODS=['NoRetrain','Periodic','Static','EWC','SADA']
subprocess.run([sys.executable,str(OUT/'validate_training.py')],check=True)
rows=[]
for name in ['qwen25_分组结果.csv','omniearth_分组结果.csv']:rows+=list(csv.DictReader((OUT/name).open()))
keys=[(r['模型'],int(r['种子']),r['组ID'],float(r['负载倍数']),r['方法']) for r in rows];assert len(keys)==len(set(keys))
seedrows=[];stats=[]
for fam in FAMILIES:
 for m in METHODS:
  for f in FACTORS:
   values=[]
   for s in [42,43,44]:
    rs=[r for r in rows if r['模型']==fam and r['方法']==m and int(r['种子'])==s and float(r['负载倍数'])==f]
    assert len(rs)=={'qwen25':5,'qwen35':2,'remoteclip':1,'resnet18':1}[fam],(fam,m,f,s,len(rs))
    v=sum(float(r['窗口准确率'])*int(r['测试样本数']) for r in rs)/sum(int(r['测试样本数']) for r in rs);values.append(v)
    seedrows.append({'模型':fam,'方法':m,'负载倍数':f,'种子':s,'窗口准确率':v})
   stats.append({'模型':fam,'方法':m,'负载倍数':f,'均值':statistics.mean(values),'样本标准差':statistics.stdev(values),'种子42':values[0],'种子43':values[1],'种子44':values[2]})
def write(p,rows):
 with p.open('w') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(OUT/'逐种子结果.csv',seedrows);write(OUT/'三种子统计.csv',stats)
table=[];text=['# 三种子完整框架实验结果','', '随机种子：42、43、44。单元格为窗口准确率均值 ± 样本标准差（n=3，ddof=1）；按各模型每档负载的最高均值加粗，三位小数下相同的值不自动视为并列。', '', '所有种子使用固定负载与配置。小模型复用固定主干特征和推理耗时，按种子训练分类头；Qwen轻量检查点通过实际训练与推理获得。','']
titles={'qwen25':'(a) Qwen2.5，自建数据集，上游等待 3 s','qwen35':'(b) Qwen3.5，OmniEarth，上游等待 3 s','remoteclip':'(c) RemoteCLIP-RN50，OmniEarth，上游等待 0.5 s','resnet18':'(d) ResNet18，OmniEarth，上游等待 0.5 s'}
for fam in FAMILIES:
 text+=['**'+titles[fam]+'**','','| 方法 | '+' | '.join(f'{f:g}×' for f in FACTORS)+' |','|---|'+'---:|'*7]
 for m in METHODS:
  name={'Periodic':f"Periodic（{20 if fam=='qwen25' else 5} 样本）",'EWC':'EWC-LoRA' if fam.startswith('qwen') else 'EWC'}.get(m,m)
  if fam=='qwen25' and m in ['Static','EWC']:name+='（轻量）'
  r={'模型':fam,'方法':name};formatted=[]
  for f in FACTORS:
   v=next(r for r in stats if r['模型']==fam and r['方法']==m and r['负载倍数']==f);s=f"{v['均值']:.3f} ± {v['样本标准差']:.3f}";r[f'{f:g}×']=s
   best=max(r['均值'] for r in stats if r['模型']==fam and r['负载倍数']==f);formatted.append('**'+s+'**' if abs(v['均值']-best)<1e-12 else s)
  text+=['| '+name+' | '+' | '.join(formatted)+' |'];table.append(r)
 text+=['']
write(OUT/'论文同格式结果表.csv',table);(OUT/'结果预览.md').write_text('\n'.join(text))
validation={'seed_count':3,'seeds':[42,43,44],'group_rows':len(rows),'per_seed_cells':len(seedrows),'aggregate_cells':len(stats),'sample_std_ddof':1}
assert len(rows)==945 and len(seedrows)==420 and len(stats)==140
(OUT/'aggregate_validation.json').write_text(json.dumps(validation,indent=2))
print('\n'.join(text))
