"""Regenerate figure: python -m pip install matplotlib; python plot_history.py."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
p=Path(__file__).resolve().parent
d=json.loads((p/'six_epoch_evidence.json').read_text())
series={k:[e['seconds'] for e in v['AUDIT']['epochs']] for k,v in d.items()}
a,b,c=series['reference1'],series['reference2'],series['candidate']
r=[(x+y)/2 for x,y in zip(a,b)]
whole=((d['reference1']['RESULT']['process_seconds']+d['reference2']['RESULT']['process_seconds'])/2)/d['candidate']['RESULT']['process_seconds']
warm=sum(r[1:])/sum(c[1:])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
red='#a50808';ink='#292626';muted='#817575';light='#faf5f5'
fig=plt.figure(figsize=(14,11),facecolor='white')
fig.text(.065,.945,'Faster-ProteinMPNN',color=red,size=17,weight='bold')
fig.text(.065,.895,'Agent Optimization History',color=ink,size=30,weight='bold')
fig.text(.065,.852,'Implementation milestones + measured training outcomes',color=muted,size=15)
fig.text(.065,.805,'1 × A100 40 GB  ·  Official full dataset  ·  6 epochs / 8,970 updates per arm',color=muted,size=13)
fig.text(.68,.945,f'{whole:.2f}×',color=red,size=32,weight='bold',ha='center')
fig.text(.68,.916,'whole process',color=muted,size=11,ha='center')
fig.text(.87,.945,f'{warm:.2f}×',color=red,size=32,weight='bold',ha='center')
fig.text(.87,.916,'epochs 2–6',color=muted,size=11,ha='center')
ax=fig.add_axes([.10,.405,.82,.335]);x=list(range(1,7))
ax.fill_between(x,[min(u,v) for u,v in zip(a,b)],[max(u,v) for u,v in zip(a,b)],color='#d9d3d3',alpha=.7)
ax.plot(x,r,color='#857c7c',lw=2.3,marker='s',markersize=6,label='Reference mean (two runs)')
ax.plot(x,c,color=red,lw=3,marker='o',markerfacecolor='white',markeredgewidth=2,markersize=8,label='Optimized candidate')
for xx,yy in zip(x,c):ax.annotate(f'{yy:.1f}s',(xx,yy),xytext=(0,-25),textcoords='offset points',ha='center',size=11,color=red,weight='bold')
ax.set_ylim(0,550);ax.set_xlim(.7,6.3);ax.set_xticks(x);ax.set_xlabel('Training epoch (not optimization round)',labelpad=12,color=muted)
ax.set_ylabel('Native epoch time (seconds)',labelpad=12,color=muted)
ax.grid(axis='y',color='#e8dede');ax.set_axisbelow(True)
for s in ax.spines.values():s.set_visible(False)
ax.tick_params(length=0,labelcolor=muted,pad=10)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),frameon=False,ncol=2,fontsize=12)
ax.annotate('Cold first epoch',xy=(1,c[0]),xytext=(1.45,345),color=muted,fontsize=11,arrowprops={'arrowstyle':'->','color':muted})
fig.text(.065,.315,'Implementation milestones',size=15,weight='bold',color=ink)
cards=[('01','Remove checkpoint recomputation','Keep original layer forward computation'),('02','Dynamic compilation + rank mask','Handle variable lengths; reduce dense mask work'),('03','Preserve RNG and cast boundaries','Retain failures; repair compiler precision behavior'),('04','Group feature transfers by dtype','Combined candidate followed by R–C–R validation')]
for i,(n,title,sub) in enumerate(cards):
 col=i%2;row=i//2;xx=.065+col*.455;yy=.208-row*.09
 box=FancyBboxPatch((xx,yy),.43,.075,boxstyle='round,pad=0.007',transform=fig.transFigure,facecolor=light,edgecolor='none');fig.patches.append(box)
 fig.text(xx+.013,yy+.043,n,size=18,color=red,weight='bold')
 fig.text(xx+.064,yy+.044,title,size=12.4,color=ink,weight='bold')
 fig.text(xx+.064,yy+.018,sub,size=9.3,color=muted)
fig.text(.065,.065,'Milestones summarize implementation history; no isolated speedup is assigned to each change.',size=10.5,color=muted)
fig.text(.065,.043,'Open markers: observed timing only. No frozen four-gate PASS or full 150-epoch convergence claim.',size=10.5,color=muted)
fig.text(.065,.021,'Source: six_epoch_evidence.json. Reference band shows two-run range, not a confidence interval.',size=10.5,color=muted)
fig.savefig(p/'optimization-history.png',dpi=160,facecolor='white')
fig.savefig(p/'optimization-history.svg',facecolor='white')
print('Whole process:',whole,'Epochs 2-6:',warm)
