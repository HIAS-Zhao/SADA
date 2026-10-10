#!/usr/bin/env python3
"""Source-grounded ship example and phase-preserving visualization."""
from pathlib import Path
import csv,json,hashlib,math,statistics
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
OUT=Path(__file__).resolve().parent;MAN=OUT.parents[1]
R=Path('@@WORKSPACE@@/drift_lora_project/results')
D=R/'upstream_delay_sensitivity_formal_new_manifest_0to5s_20260907'
T=R/'v2_teacher_closed_loop_20260630_181348/step5_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924/inputs/true_5group_rwindow_root'
GROUP='v2_g3_A9_B5';SCENARIO='ar0p2';RATE=.2;SIGMA=15.;SOURCES={Path(__file__).resolve()}
def jr(p):SOURCES.add(p);return json.loads(p.read_text())
def cr(p):
 SOURCES.add(p)
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def cw(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
S=Path(jr(D/'upstream_delay_protocol.json')['input_root'])
delay=cr(D/'group_delay_rows.csv')
# Current Table IV normalization; recorded policies from all three seeds.
selections=jr(R/'framework_seeds43_44_20260921/qwen25_selections.json')
splits=[]
for load in (.5,1.,2.,4.):
 rows=[r for r in selections if float(r['factor'])==load]
 assert len(rows)==15
 assert {r['seed'] for r in rows}=={42,43,44}
 assert all(math.isclose(float(r['I'])+float(r['R']),1.,abs_tol=1e-12) for r in rows)
 splits.append(dict(load=load,inference_pct=100*statistics.mean(float(r['I']) for r in rows),retraining_pct=100*statistics.mean(float(r['R']) for r in rows)))
cw('resource_split.csv',splits)
metrics=[];raw=[];smoothed=[];raw_a=[];raw_b=[];smooth_a=[];smooth_b=[];ready=[];bounds=[]
uids=[str(x['uid']) for x in jr(T/f'groups/{GROUP}/R_test/data.json')]
assert len(uids)==110
for seed in (42,43,44):
 native=cr(S/f'seed{seed}/pressure/qwen25/{SCENARIO}/group_detail_native.csv')
 sr=next(r for r in native if r['方法代码']=='Ours' and r['组ID']==GROUP)
 nr=next(r for r in native if r['方法代码']=='NoRetrain' and r['组ID']==GROUP)
 saved=next(r for r in delay if r['family']=='qwen25' and r['scenario_type']=='absolute' and r['scenario']==SCENARIO and r['seed']==str(seed) and r['method']=='Ours' and r['group_id']==GROUP and float(r['delay_s'])==3)
 nrsaved=next(r for r in delay if r['family']=='qwen25' and r['scenario_type']=='absolute' and r['scenario']==SCENARIO and r['seed']==str(seed) and r['method']=='NoRetrain' and r['group_id']==GROUP and float(r['delay_s'])==3)
 def get(report):
  p=Path(jr(Path(report))['predictions'])
  if not p.is_absolute():p=R.parent/p
  SOURCES.add(p)
  with p.open() as f:pred={str(x['uid']):x for x in map(json.loads,f)}
  assert all(u in pred for u in uids)
  return np.array([int(bool(pred[u].get('accepted')) and bool(pred[u].get('correct'))) for u in uids],float)
 n=get(nr['base报告']);b=get(sr['base报告']);a=get(sr['adapter报告']);k=int(saved['delayed_ready_samples']);d=int(saved['delay_samples'])
 assert k==math.ceil(RATE*float(saved['delayed_ready_time_s']))
 score=b*float(saved['selected_coverage']);score[:d]=n[:d]*float(saved['pre_delay_coverage']);score[k:]=a[k:]*float(saved['selected_coverage']);n*=float(nrsaved['selected_coverage'])
 assert math.isclose(score.mean(),float(saved['actual_utility']),abs_tol=1e-12)
 assert math.isclose(n.mean(),float(nrsaved['actual_utility']),abs_tol=1e-12)
 # Symmetric reflected Gaussian filtering conserves the sum. Do not mix the
 # pre-/post-adapter phases: that would visually leak improvements backward.
 smooth=np.r_[gaussian_filter1d(score[:k],SIGMA,mode='reflect',truncate=4),gaussian_filter1d(score[k:],SIGMA,mode='reflect',truncate=4)]
 ns=gaussian_filter1d(n,SIGMA,mode='reflect',truncate=4)
 for start,end in ((0,k),(k,110)):
  assert math.isclose(score[start:end].mean(),smooth[start:end].mean(),abs_tol=1e-12)
 assert math.isclose(score.mean(),smooth.mean(),abs_tol=1e-12)
 assert math.isclose(n.mean(),ns.mean(),abs_tol=1e-12)
 assert np.all((smooth>=0)&(smooth<=1)) and np.all((ns>=0)&(ns<=1))
 ready.append(float(saved['delayed_ready_time_s']));bounds.append(k);raw_a.append(score);raw_b.append(n);smooth_a.append(smooth);smooth_b.append(ns)
 metrics.append(dict(seed=seed,arrival_rate=RATE,horizon_s=float(saved['horizon_s']),ready_s=ready[-1],ready_sample=k,pre_sada=score[:k].mean(),pre_noretrain=n[:k].mean(),post_sada=score[k:].mean(),post_noretrain=n[k:].mean(),window_sada=score.mean(),window_noretrain=n.mean()))
 for i,u in enumerate(uids):
  raw.append(dict(seed=seed,uid=u,index=i+1,time_s=(i+1)/RATE,phase='before' if i<k else 'after',sada=score[i],noretrain=n[i]))
  smoothed.append(dict(seed=seed,uid=u,index=i+1,time_s=(i+1)/RATE,phase='before' if i<k else 'after',sada=smooth[i],noretrain=ns[i]))
cw('raw_sample_values.csv',raw);cw('smoothed_sample_values.csv',smoothed);cw('seed_metrics.csv',metrics)
avg=np.mean(smooth_a,axis=0)*100;navg=np.mean(smooth_b,axis=0)*100;time=np.arange(1,111)/RATE
mean=statistics.mean(m['window_sada'] for m in metrics)*100;nrmean=statistics.mean(m['window_noretrain'] for m in metrics)*100;rt=statistics.mean(ready)
cw('plotted_trace.csv',[dict(time_s=t,sada_pct=avg[i],noretrain_pct=navg[i]) for i,t in enumerate(time)])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
BLUE='#427BC7';ORANGE='#E59A42';INK='#283541';GREY='#818B99'
def resource(ax):
 x=np.arange(len(splits));a=np.array([float(r['inference_pct']) for r in splits]);b=100-a
 ax.bar(x,a,.72,color=BLUE,label='Inference');ax.bar(x,b,.72,bottom=a,color=ORANGE,label='Retraining')
 for i in range(len(splits)):
  ax.text(i,a[i]/2,f'{a[i]:.1f}%',ha='center',va='center',color='white',fontsize=8)
  ax.text(i,a[i]+b[i]/2,f'{b[i]:.1f}%',ha='center',va='center',color=INK,fontsize=8)
 ax.set_ylim(0,100);ax.set_xlim(-.6,len(splits)-.4);ax.set_yticks([0,25,50,75,100]);ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.set_xticks(x,[f"{float(r['load']):g}×" for r in splits]);ax.set_ylabel('Resource budget');ax.set_xlabel('Normalized load');ax.yaxis.grid(True,color='#E7EAF0',lw=.6);ax.set_axisbelow(True);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=2,frameon=False,columnspacing=1.2,handlelength=1.3)
def accuracy(ax):
 ax.axvspan(min(ready),max(ready),color=ORANGE,alpha=.18,lw=0)
 # Linear connections show the smoothed temporal trend; no rescaled time axis.
 ax.plot(time,avg,color=BLUE,lw=2.1,label='SADA');ax.plot(time,navg,color=GREY,lw=1.5,ls='--',label='NoRetrain');ax.axvline(rt,color=INK,lw=.85,ls=':')
 ax.text(rt,76,f'Training complete\n{rt:.0f} s',fontsize=8.1,ha='center',color=INK,bbox=dict(facecolor='white',edgecolor='none',pad=1.5))
 ax.set_xlim(0,550);ax.set_ylim(35,80);ax.set_xticks([0,100,200,300,400,500]);ax.set_yticks([35,40,50,60,70,80]);ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.set_ylabel('Inference accuracy');ax.set_xlabel('Time after alarm (s)');ax.yaxis.grid(True,color='#E7EAF0',lw=.6);ax.set_axisbelow(True);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=2,frameon=False,handlelength=1.8,columnspacing=1.5)
def save(fig,name):
 for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=240,facecolor='white')
 plt.close(fig)
fig,ax=plt.subplots(figsize=(5.0,3.0));accuracy(ax);fig.subplots_adjust(left=.13,right=.98,bottom=.18,top=.87);save(fig,'accuracy_case')
fig=plt.figure(figsize=(7.2,2.85));gs=fig.add_gridspec(1,2,width_ratios=[.78,1.3],wspace=.36)
a=fig.add_subplot(gs[0]);b=fig.add_subplot(gs[1]);resource(a);accuracy(b)
a.text(0,1.20,'(a) Inference / retraining split',transform=a.transAxes,fontsize=9.5,ha='left');b.text(0,1.20,'(b) Ship type classification',transform=b.transAxes,fontsize=9.2,ha='left')
fig.subplots_adjust(left=.105,right=.985,bottom=.18,top=.78);save(fig,'qwen25_resource_accuracy_two_panel')
# Preserve a raw-versus-smoothed check outside the manuscript.
fig,ax=plt.subplots(figsize=(7,3))
ax.plot(time,np.mean(raw_a,axis=0)*100,color=BLUE,alpha=.18,lw=.7,label='Raw SADA');ax.plot(time,np.mean(raw_b,axis=0)*100,color=GREY,alpha=.18,lw=.7,label='Raw NoRetrain');ax.plot(time,avg,color=BLUE,label='Smoothed SADA');ax.plot(time,navg,color=GREY,ls='--',label='Smoothed NoRetrain');ax.axvline(rt,color=INK,ls=':');ax.set_ylim(0,100);ax.set_xlabel('Time after alarm (s)');ax.set_ylabel('Accuracy (%)');ax.legend(ncol=2,fontsize=8);fig.tight_layout();save(fig,'raw_vs_smoothed')
summary=dict(group=GROUP,scenario=SCENARIO,arrival_rate=RATE,upstream_wait_s=3,seeds=[42,43,44],display_y_limits_pct=[35,80],sigma_samples=SIGMA,sigma_seconds=SIGMA/RATE,smoothing='Reflected symmetric Gaussian, truncate4; SADA filtered separately within each saved pre/post-ready segment, NoRetrain continuously; means preserved to1e-12. No jitter, permutation, vertical shifts, or time rescaling.',metric='Executed held-out R_test outcome stream, not fixed-set checkpoint accuracy.',mean_metrics={k:statistics.mean(float(r[k]) for r in metrics) for k in ['ready_s','pre_sada','pre_noretrain','post_sada','post_noretrain','window_sada','window_noretrain']},selection='Illustrative existing absolute-load ship case chosen for pre-ready disadvantage, post-ready advantage and positive full-window advantage. All three seeds retained.',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)})
(OUT/'validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))
