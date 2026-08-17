#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'analysis/canonical_cohorts/reference_outputs/figure_interpolated_mean_sd_v18.csv'
OUT = ROOT / 'manuscript/figures'
FAMILIES = ['learned','sinusoidal','rope','alibi']
FLABEL = {'learned':'Learned','sinusoidal':'Sinusoidal','rope':'RoPE','alibi':'ALiBi'}
DATASETS = ['imagenet','cifar']
DLABEL = {'imagenet':'ImageNet-100','cifar':'CIFAR-100'}
YLIMS = {'imagenet':(0.0,1.03),'cifar':(0.84,1.01)}
PRIMARY_ENDPOINT=0.09
DISPLAY_MAX=0.14

def main():
    df=pd.read_csv(CSV)
    plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':13,
                         'legend.fontsize':11,'xtick.labelsize':11,'ytick.labelsize':11})
    fig,axes=plt.subplots(2,4,figsize=(15.5,7.4),sharex=True,sharey=False)
    for ri,ds in enumerate(DATASETS):
        for ci,fam in enumerate(FAMILIES):
            ax=axes[ri,ci]
            for regime,ls,label in [('noise','-','Random noise'),('adversarial','--','Adversarial PGD')]:
                q=df.query('dataset == @ds and family == @fam and regime == @regime').sort_values('rho')
                line,=ax.plot(q.rho,q.mean_accuracy,linewidth=2.0,linestyle=ls,label=label,zorder=3)
                ax.fill_between(q.rho,q.low_accuracy,q.high_accuracy,color=line.get_color(),alpha=.18,zorder=1)
            ax.axvline(PRIMARY_ENDPOINT,linewidth=1.2,linestyle=':',alpha=.9,zorder=2)
            ax.set_title(FLABEL[fam])
            ax.set_xlim(0,DISPLAY_MAX)
            ax.set_ylim(*YLIMS[ds])
            ax.grid(True,alpha=.25,linewidth=.6)
            if ci==0:
                ax.set_ylabel(DLABEL[ds]+'\nNormalized accuracy')
            if ri==1:
                ax.set_xlabel(r'Achieved $\rho_{\mathrm{rel}}$')
    handles=[
        Line2D([0],[0],linewidth=2.0,linestyle='-',label='Random noise'),
        Line2D([0],[0],linewidth=2.0,linestyle='--',label='Adversarial PGD'),
        Line2D([0],[0],linewidth=1.2,linestyle=':',label=r'Primary endpoint $\rho_{\max}=0.09$'),
        Line2D([0],[0],linewidth=6.0,alpha=.18,label=r'Mean $\pm$ 1 sample SD, $n=6$'),
    ]
    fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,.995))
    fig.subplots_adjust(left=.065,right=.995,bottom=.105,top=.895,wspace=.13,hspace=.24)
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/'fig_primary_robustness_curves.pdf',bbox_inches='tight')
    fig.savefig(OUT/'fig_primary_robustness_curves.png',dpi=300,bbox_inches='tight')
    fig.savefig(OUT/'fig_primary_robustness_curves.svg',bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__': main()
