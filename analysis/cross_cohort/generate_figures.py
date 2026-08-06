#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS=[42,123,456,789,1011,1213]
FAMILIES=['learned','sinusoidal','rope','alibi']
FLABEL={'learned':'Learned','sinusoidal':'Sinusoidal','rope':'RoPE','alibi':'ALiBi'}
ARCHES=['vitb','vits']
ALABEL={'vitb':'ViT-B/16','vits':'ViT-S/16'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True, type=Path)
    ap.add_argument('--d031-point-csv', required=True, type=Path)
    args=ap.parse_args()
    out=args.output_dir
    curves=pd.read_csv(out/'v19_imagenet_curves_long.csv')
    rep=json.loads((out/'V19_SUPPORT_AWARE_BUILD_REPORT.json').read_text())
    end=rep['rho_common']['cross_architecture']
    grid=np.linspace(0,end,301)
    fig,axes=plt.subplots(4,2,figsize=(10.2,11.0),sharex=True,sharey=True)
    for i,fam in enumerate(FAMILIES):
        for j,arch in enumerate(ARCHES):
            ax=axes[i,j]
            for regime,label in [('noise','Random noise'),('adversarial','Adversarial')]:
                vals=[]
                for seed in SEEDS:
                    q=curves.query('architecture==@arch and family==@fam and seed==@seed and regime==@regime').sort_values('rho')
                    vals.append(np.interp(grid,q.rho.to_numpy(),q.normalized_accuracy.to_numpy()))
                arr=np.asarray(vals); mean=arr.mean(0); sd=arr.std(0,ddof=1)
                line,=ax.plot(grid,mean,label=label,linewidth=1.7,linestyle='-' if regime=='noise' else '--')
                ax.fill_between(grid,mean-sd,mean+sd,alpha=.16,color=line.get_color(),linewidth=0)
            if i==0: ax.set_title(ALABEL[arch])
            if j==0: ax.set_ylabel(f'{FLABEL[fam]}\nNormalized accuracy')
            if i==3: ax.set_xlabel(r'Achieved $\rho_{\mathrm{rel}}$')
            ax.set_xlim(0,end); ax.set_ylim(0,1.04); ax.grid(alpha=.22)
            if i==0 and j==1: ax.legend(loc='lower left',frameon=False)
    fig.suptitle(f'ImageNet-100 cross-architecture robustness on common support [0, {end:.5f}]',y=.995,fontsize=12)
    fig.tight_layout(rect=[0,0,1,.985])
    for ext in ['pdf','png','svg']:
        fig.savefig(out/f'fig_cross_architecture_robustness.{ext}',dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)

    df=pd.read_csv(args.d031_point_csv)
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.8),sharey=True)
    for ax,budget in zip(axes,[.0095,.02]):
        subset=df.query('architecture=="vits" and training_seed==42 and budget==@budget')
        for objective,label,selection in [('task_loss','Task loss','loss_selected'),('direct_rho',r'Direct-$\rho$','')]:
            q=subset[subset.objective.eq(objective)]
            if objective=='task_loss': q=q[q.selection.eq(selection)]
            r=q.iloc[0]
            vals=[r[f'c_L{i}'] for i in range(1,13)]
            ax.plot(range(1,13),vals,marker='o',label=label,linewidth=1.6)
        ax.set_title(f'Budget {budget:g}')
        ax.set_xlabel('Transformer layer')
        ax.set_xticks(range(1,13)); ax.set_ylim(bottom=0); ax.grid(alpha=.22)
    axes[0].set_ylabel('Share of squared layer displacement')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    for ext in ['pdf','png','svg']:
        fig.savefig(out/f'fig_layerwise_displacement_profiles.{ext}',dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)
    print('figures generated')

if __name__=='__main__':
    main()
