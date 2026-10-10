#!/usr/bin/env python3
"""Read frozen experiment outputs; verify, export and visualize without retraining."""
from pathlib import Path
from collections import defaultdict
import csv, json, hashlib, statistics, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
MANUSCRIPT = OUT.parents[1]
RESULTS = Path('@@WORKSPACE@@/drift_lora_project/results')
CANON = RESULTS/'model_specific_delay_normalized_20260914/normalized_load_mean_std.csv'
TARGET = RESULTS/'qwen25_target80_admission_gate_3s_20260914'
DELAY = RESULTS/'upstream_delay_sensitivity_formal_new_manifest_0to5s_20260907'
SOURCE = RESULTS/'dtop80_current_pool_recompute_20260726/all_model_rtest_only_formal_profiles_new_manifest'
TRUTH = RESULTS/'v2_teacher_closed_loop_20260630_181348/step5_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924/inputs/true_5group_rwindow_root'
LOADS = (.25,.5,.75,1,1.25,1.5,2,3,4)
METHODS = ('NoRetrain','PeriodicFixedRetrain','StaticSplitContinuous','EWC','Ours')
LABELS = dict(zip(METHODS,('NoRetrain','Periodic','Static','EWC-LoRA','SADA')))
SOURCES = {Path(__file__).resolve()}

def read_csv(path):
    SOURCES.add(path)
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def read_json(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_csv(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

cache={}
def prediction(report):
    report=Path(report)
    if report not in cache:
        path=Path(read_json(report)['predictions'])
        if not path.is_absolute(): path=RESULTS.parent/path
        SOURCES.add(path)
        with path.open() as f: cache[report]={str(r['uid']):r for r in map(json.loads,f) if r.get('uid') is not None}
    return cache[report]

def correct(report, uids):
    p=prediction(report)
    assert all(u in p for u in uids), (report,'missing UID')
    return np.array([int(bool(p[u].get('accepted')) and bool(p[u].get('correct'))) for u in uids],float)

canonical=read_csv(CANON)
assert len(canonical)==4*9*5
assert all(int(r['seed_count'])==3 for r in canonical)
write_csv('all_families_all_methods.csv', canonical)
q25=[r for r in canonical if r['model_family']=='qwen25']
lookup={(r['method'],float(r['load_factor_vs_noretrain_throughput'])):r for r in q25}
assert len(lookup)==45
frozen=read_csv(DELAY/'group_delay_rows.csv')
baseline_rows=[r for r in frozen if r['family']=='qwen25' and r['scenario_type']=='normalized' and float(r['delay_s'])==3 and r['method'] in METHODS[:-1]]
assert len(baseline_rows)==9*3*5*4
by=defaultdict(list)
for r in baseline_rows: by[(r['method'],r['scenario'],int(r['seed']))].append(r)
checks=[]
for m in METHODS[:-1]:
    for load in LOADS:
        scenario=f'load{load:g}x'.replace('.','p')
        values=[]
        for seed in (42,43,44):
            rr=by[m,scenario,seed]; assert len(rr)==5
            values.append(sum(float(r['actual_utility'])*int(r['total_samples']) for r in rr)/sum(int(r['total_samples']) for r in rr))
        ref=lookup[m,load]
        assert math.isclose(statistics.mean(values),float(ref['actual_long_mean']),abs_tol=1e-12)
        assert math.isclose(statistics.stdev(values),float(ref['actual_long_std']),abs_tol=1e-12)
        checks.append(dict(method=m,load=load,mean=statistics.mean(values),std=statistics.stdev(values),groups=15,adapter_active=sum(r['adapter_active_with_delay']=='True' for seed in (42,43,44) for r in by[m,scenario,seed])))
write_csv('baseline_reproduction.csv',checks)
# Reconstruct every SADA sample exactly as the published scorer does.
detail=read_csv(TARGET/'target80_group_actual_results.csv')
assert len(detail)==135
selection={}
source_detail={}
for seed in (42,43,44):
    for r in read_csv(TARGET/f'seed{seed}/admission_policy_selections.csv'):
        selection[seed,float(r['load_factor']),r['group_id']]=r
    for manifest in (SOURCE/f'seed{seed}/pressure/normalized_load_runs/qwen25').glob('*/pressure_recompute_manifest.json'):
        rate=float(read_json(manifest)['arrival_rate'])
        source_detail[seed,rate]=read_csv(manifest.parent/'group_detail_native.csv')
traces=defaultdict(list); resources=[]; samples=[]; policy=[]; baseline_traces=defaultdict(list)
audit_index={(int(r['seed']),r['scenario'],r['group_id'],r['method']):r for r in baseline_rows}
for r in detail:
    seed=int(r['seed']);load=float(r['load_factor']);group=r['group_id'];rate=float(r['arrival_rate'])
    s=selection[seed,load,group]
    uids=[str(x['uid']) for x in read_json(TRUTH/f'groups/{group}/R_test/data.json')]
    assert len(uids)==110
    matched=min((k for k in source_detail if k[0]==seed),key=lambda k:abs(k[1]-rate))
    assert abs(matched[1]-rate)<1e-5
    nr=next(x for x in source_detail[matched] if x['方法代码']=='NoRetrain' and x['组ID']==group)
    baseline=correct(nr['base报告'],uids)*float(audit_index[seed,r['scenario'],group,'NoRetrain']['selected_coverage'])
    for method in METHODS[:-1]:
        saved=audit_index[seed,r['scenario'],group,method]
        native=next(x for x in source_detail[matched] if x['方法代码']==method and x['组ID']==group)
        count=int(saved['delay_samples']); end=int(saved['delayed_ready_samples'])
        pre=correct(nr['base报告'],uids)*float(saved['pre_delay_coverage'])
        bs=correct(native['base报告'],uids)*float(saved['selected_coverage'])
        bs[:count]=pre[:count]
        assert end==110, 'Current fixed baselines have no adapter active in-window'
        assert math.isclose(bs.mean(),float(saved['actual_utility']),abs_tol=1e-12)
        baseline_traces[load,method].append(bs)
    base=correct(r['base_report'],uids); score=base*float(r['coverage'])
    delay=int(r['delay_samples']);ready=int(r['ready_samples'])
    if r['gamma_id']:
        adap=correct(r['adapter_report'],uids)
        score[:delay]=correct(nr['base报告'],uids)[:delay]
        score[ready:]=adap[ready:]*float(r['coverage'])
        assert score[:delay].sum()==int(r['pre_correct'])
        assert base[delay:ready].sum()==int(r['base_correct'])
        assert adap[ready:].sum()==int(r['adapter_correct'])
    assert math.isclose(float(score.mean()),float(r['actual_utility']),abs_tol=1e-12)
    traces[load].append((score,baseline))
    inference=float(r['I']); residual=float(r['R']); training=residual if r['gamma_id'] else 0
    resources.append(dict(seed=seed,load=load,group=group,inference=inference,training_budget=training,unused_budget=residual-training,adapter_active=int(r['adapter_samples'])>0))
    policy.append(dict(seed=seed,load=load,group=group,lambda_id=r['lambda_id'],gamma_id=r['gamma_id'],inference=inference,residual=residual,coverage=float(r['coverage']),ready_s=3+float(s['finish_time_s']) if r['gamma_id'] else '',ready_samples=ready,delay_samples=delay,horizon_s=len(uids)/rate))
    for i,u in enumerate(uids):samples.append(dict(seed=seed,load=load,group=group,uid=u,sample=i+1,time_s=(i+1)/rate,sada=score[i],noretrain=baseline[i]))
for load,rr in traces.items():
    values=[np.mean([float(r['actual_utility']) for r in detail if int(r['seed'])==seed and float(r['load_factor'])==load]) for seed in (42,43,44)]
    assert math.isclose(statistics.mean(values),float(lookup['Ours',load]['actual_long_mean']),abs_tol=1e-12)
    assert math.isclose(statistics.stdev(values),float(lookup['Ours',load]['actual_long_std']),abs_tol=1e-12)
write_csv('sada_resource_decisions.csv',resources)
write_csv('sada_policy_decisions.csv',policy)
write_csv('sada_sample_traces.csv',samples)
# A transparent diagnostic: no trajectory selection by observed gain.
# Pool all five groups and all three seeds at each load in original R_test order.
for load in (.25,.5,1):
    a=np.mean([r[0] for r in traces[load]],axis=0)
    b=np.mean([r[1] for r in traces[load]],axis=0)
    print('LOAD',load,'blocks10 SADA',np.round(a.reshape(11,10).mean(1),3),'NR',np.round(b.reshape(11,10).mean(1),3),'MEAN',a.mean(),b.mean())
# Table V with no change to the protocol.
lines=[r'\begin{table*}[!t]',r'\centering',r'\small',r'\caption{End-to-end results on the custom dataset at all nine normalized loads with a fixed \(3\,\mathrm{s}\) upstream wait. Entries are three-seed means of realized \(\mathrm{Acc}_{\rm win}\); nonzero sample standard deviations appear in parentheses. Bold indicates the highest mean at each load, including ties.}',r'\label{tab:framework_custom}',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}',r'\toprule','Method & '+' & '.join(r'\('+f'{l:g}'+r'\times\)' for l in LOADS)+r' \\',r'\midrule']
for m in METHODS:
    vals=[]
    for l in LOADS:
        r=lookup[m,l];mean=float(r['actual_long_mean']);std=float(r['actual_long_std']);v=f'{mean:.3f}'
        if std>1e-12:v+=f'({std:.3f})'
        if abs(mean-max(float(lookup[x,l]['actual_long_mean']) for x in METHODS))<1e-12:v=r'\textbf{'+v+'}'
        vals.append(v)
    lines.append(LABELS[m]+' & '+' & '.join(vals)+r' \\')
lines += [r'\bottomrule',r'\end{tabular*}',r'\end{table*}']
(OUT/'table_v.tex').write_text('\n'.join(lines)+'\n')
write_csv('table_v_all_loads.csv',[{'method':LABELS[m],**{f'{l:g}x':lookup[m,l]['actual_long_mean'] for l in LOADS}} for m in METHODS])
(OUT/'validation.json').write_text(json.dumps(dict(canonical_rows=len(canonical),baseline_group_rows=len(baseline_rows),sada_group_rows=len(detail),baseline_mean_std_checks=36,sada_mean_std_checks=9,per_sample_baseline_checks=540,per_sample_sada_checks=135,protocol='Frozen current manuscript outputs; Qwen2.5 upstream wait 3 s; seeds 42/43/44; five groups; 110 ordered R_test samples/group. No training or reselection.',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}),indent=2))
print('VERIFIED complete baseline and SADA records')

# Reproduce publication graphics directly from the audited arrays.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
BLUE='#427BC7'; ORANGE='#E59A42'; GREY='#E3E7ED'; GREEN='#2E8B66'; RED='#C64C57'; INK='#26323F'
fig=plt.figure(figsize=(7.2,6.15))
gs=fig.add_gridspec(3,3,height_ratios=[1.35,1,1.25],hspace=.63,wspace=.23)
ax=fig.add_subplot(gs[0,:])
xx=np.arange(len(LOADS)); summaries=[]
for l in LOADS:
    rr=[r for r in resources if r['load']==l]
    summaries.append(dict(load=l,inference=np.mean([r['inference'] for r in rr]),training=np.mean([r['training_budget'] for r in rr]),unused=np.mean([r['unused_budget'] for r in rr]),active=sum(r['adapter_active'] for r in rr)))
a=np.array([s['inference'] for s in summaries])*100
b=np.array([s['training'] for s in summaries])*100
c=np.array([s['unused'] for s in summaries])*100
ax.bar(xx,a,.65,color=BLUE,label='Inference')
ax.bar(xx,b,.65,bottom=a,color=ORANGE,label='Retraining')
ax.bar(xx,c,.65,bottom=a+b,color=GREY,edgecolor='#A9B2BE',linewidth=.4,label='Idle / released budget')
for i,s in enumerate(summaries):
    if a[i]>10: ax.text(i,a[i]/2,f'{a[i]:.0f}%',ha='center',va='center',color='white',fontsize=8)
    if b[i]>10: ax.text(i,a[i]+b[i]/2,f'{b[i]:.0f}%',ha='center',va='center',color=INK,fontsize=8)
    ax.text(i,106,f"{s['active']}/15",ha='center',va='center',fontsize=7.5,color=INK)
ax.text(-.53,118,'Adapters active within window:',ha='left',fontsize=7.5,color=INK)
ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100]);ax.set_xlim(-.65,8.65)
ax.set_xticks(xx,[f'{l:g}×' for l in LOADS]);ax.set_xlabel('Normalized load');ax.set_ylabel('GPU budget (%)')
ax.set_title('(a) SADA resource allocation at all nine loads',loc='left',pad=27)
ax.legend(loc='lower right',bbox_to_anchor=(1,1.1),ncol=3,frameon=False,handlelength=1.2,columnspacing=1.3)
ax.yaxis.grid(True,alpha=.15);ax.set_axisbelow(True)

# Resources are modeled from the selected policy; no post-ready reassignment is invented.
# Inference remains at selected I after readiness; completed retraining share becomes idle.
plot_export=[];timeline_export=[]
for col,(l,method,title) in enumerate(((.25,'fixed','Fixed baselines · 0.25×'),(.25,'sada','SADA · 0.25×'),(.5,'sada','SADA · 0.5×'))):
    rr=[r for r in policy if r['load']==l]
    H=rr[0]['horizon_s']; rate=110/H
    breaks=sorted(set([0.,3.,H]+[min(H,float(r['ready_s'])) for r in rr if r['ready_s']!='']))
    if method=='fixed':breaks=[0.,3.,H]
    parts=[]
    for t in breaks:
        if method=='fixed':parts.append([1.,0.,0.] if t<3 else [.5,.5,0.])
        else:
            inf=np.mean([1. if r['ready_s']!='' and t<3 else r['inference'] for r in rr])
            train=np.mean([r['residual'] if r['ready_s']!='' and 3<=t<float(r['ready_s']) else 0 for r in rr])
            parts.append([inf,train,1-inf-train])
        timeline_export.append(dict(load=l,method=method,time_s=t,inference=parts[-1][0],retraining=parts[-1][1],idle=parts[-1][2]))
    p=np.array(parts)*100
    axr=fig.add_subplot(gs[1,col])
    axr.stackplot(breaks,p.T,colors=[BLUE,ORANGE,GREY],step='post',linewidth=0)
    axr.set_xlim(0,H);axr.set_ylim(0,103);axr.set_yticks([0,50,100]);axr.set_title(f'({chr(98+col)}) {title}',loc='left',pad=8)
    if col==0:axr.set_ylabel('GPU budget (%)')
    axr.set_xticks([0,100,200,H] if l==.25 else [0,50,100,H]);axr.set_xticklabels(['0','100','200','286'] if l==.25 else ['0','50','100','143'])
    axr.text(.97,.90,'No adapter ready' if method=='fixed' else f"{sum(r['ready_s']!='' and float(r['ready_s'])<H for r in rr)}/15 adapters ready",transform=axr.transAxes,ha='right',va='top',fontsize=7.3,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1.8))
    axr.set_xlabel('Time after alarm (s)')
    axr.axvline(3,color=INK,lw=.65,ls=':')
    # Mean over five groups and three seeds in disjoint blocks of 10 samples/group.
    sa=np.mean([r[0] for r in traces[l]],axis=0)
    nr=np.mean([r[1] for r in traces[l]],axis=0)
    sb=sa.reshape(11,10).mean(1)*100;nb=nr.reshape(11,10).mean(1)*100
    edges=np.linspace(0,H,12)
    axc=fig.add_subplot(gs[2,col])
    if method=='fixed':
        fixed=np.mean(baseline_traces[l,'StaticSplitContinuous'],axis=0)
        assert all(np.allclose(np.mean(baseline_traces[l,m],axis=0),fixed) for m in METHODS[1:4])
        axc.stairs(fixed.reshape(11,10).mean(1)*100,edges,color=BLUE,lw=1.6,label='Fixed baseline',baseline=None)
        mean=fixed.mean()*100
    else:
        axc.stairs(nb,edges,color='#8994A1',lw=1,ls='--',label='NoRetrain',baseline=None)
        axc.stairs(sb,edges,color=BLUE,lw=1.6,label='SADA',baseline=None)
        for j in range(11):
            axc.fill_between(edges[j:j+2],nb[j],sb[j],color=GREEN if sb[j]>=nb[j] else RED,alpha=.20,lw=0)
        mean=sa.mean()*100
        readies=[float(r['ready_s']) for r in rr if r['ready_s']!='' and float(r['ready_s'])<H]
        axc.plot(readies,[4]*len(readies),'|',color=ORANGE,ms=6,mew=1.2,clip_on=True)
        if col==1:axc.text(.03,.08,'| adapter ready',color='#A86D21',transform=axc.transAxes,ha='left',fontsize=7)
        axc.legend(loc='upper left',bbox_to_anchor=(0,.88),frameon=False,fontsize=7,handlelength=1.6,ncol=2,columnspacing=.8,borderaxespad=.2)
    axc.axhline(mean,color=ORANGE,ls=':',lw=1.4)
    axc.text(.98,.95,f'Window mean: {mean:.1f}%',transform=axc.transAxes,ha='right',va='top',fontsize=7.3,color=INK)
    axc.set_ylim(0,100);axc.set_yticks([0,25,50,75,100]);axc.set_xlim(0,H)
    axc.set_title(f'({chr(101+col)}) Accuracy over time',loc='left',pad=8)
    if col==0:axc.set_ylabel('Inference accuracy (%)')
    axc.set_xlabel('Time after alarm (s)')
    axc.set_xticks([0,100,200,H] if l==.25 else [0,50,100,H]);axc.set_xticklabels(['0','100','200','286'] if l==.25 else ['0','50','100','143'])
    axc.grid(axis='y',alpha=.15)
    for j in range(11):plot_export.append(dict(method=method,load=l,start_s=edges[j],end_s=edges[j+1],sada_percent=sb[j],noretrain_percent=nb[j],samples_per_group=10))
fig.subplots_adjust(left=.08,right=.985,top=.90,bottom=.075)
for ext in ('pdf','png','svg'):
    fig.savefig(OUT/f'qwen25_resource_accuracy.{ext}',dpi=240,facecolor='white')
plt.close(fig)
write_csv('resource_load_summary.csv',summaries)
write_csv('resource_timeline.csv',timeline_export)
write_csv('accuracy_time_bins.csv',plot_export)
# Optional companion: all nine time trajectories and cumulative means, full precision data.
fig,axes=plt.subplots(3,3,figsize=(8.6,7.2),sharey=True)
for ax,l in zip(axes.flat,LOADS):
    sa=np.mean([r[0] for r in traces[l]],axis=0);nr=np.mean([r[1] for r in traces[l]],axis=0)
    H=next(r['horizon_s'] for r in policy if r['load']==l);t=np.arange(1,111)*H/110
    ax.plot(t,np.cumsum(sa)/np.arange(1,111)*100,color=BLUE,label='SADA',lw=1.5)
    ax.plot(t,np.cumsum(nr)/np.arange(1,111)*100,color='#8994A1',label='NoRetrain',lw=1.1,ls='--')
    ax.set_title(f'{l:g}× load · final {sa.mean()*100:.1f}% / {nr.mean()*100:.1f}%',loc='left',fontsize=9)
    ax.set_ylim(0,100);ax.set_xlim(0,H);ax.grid(axis='y',alpha=.15)
    ax.set_xlabel('Time after alarm (s)');ax.set_ylabel('Cumulative accuracy (%)')
axes[0,0].legend(frameon=False,loc='lower right')
fig.tight_layout()
for ext in ('pdf','png'):fig.savefig(OUT/f'cumulative_accuracy_all_loads.{ext}',dpi=200)
plt.close(fig)
print('FIGURES',OUT/'qwen25_resource_accuracy.png')
