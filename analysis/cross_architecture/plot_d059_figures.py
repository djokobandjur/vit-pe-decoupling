#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FAMILIES=["learned","sinusoidal","rope"]
LABELS={"learned":"Learned","sinusoidal":"Sinusoidal","rope":"RoPE"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--analysis',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    agg=pd.read_csv(args.analysis/'D059_PRIMARY_NAUC_AGGREGATE_v1.csv'); x=np.arange(3); w=.34
    b=[agg[(agg.architecture=='vitb_amp')&(agg.family==f)].iloc[0] for f in FAMILIES]
    s=[agg[(agg.architecture=='vits_amp')&(agg.family==f)].iloc[0] for f in FAMILIES]
    plt.figure(figsize=(7.5,4.6)); plt.bar(x-w/2,[r.attack_nauc_mean for r in b],w,yerr=[r.attack_nauc_sd for r in b],capsize=4,label='ViT-B/16 AMP-trained'); plt.bar(x+w/2,[r.attack_nauc_mean for r in s],w,yerr=[r.attack_nauc_sd for r in s],capsize=4,label='ViT-S/16 AMP-trained'); plt.xticks(x,[LABELS[f] for f in FAMILIES]); plt.ylim(.6,1.02); plt.ylabel('Adversarial nAUC'); plt.title(r'AMP-matched cross-architecture comparison ($\rho_{\max}=0.08735$)'); plt.legend(); plt.tight_layout(); plt.savefig(args.out/'fig_D059_attack_nauc_amp_matched.pdf'); plt.savefig(args.out/'fig_D059_attack_nauc_amp_matched.png',dpi=300); plt.close()
    con=pd.read_csv(args.analysis/'D059_SEED_LEVEL_ARCHITECTURE_CONTRASTS_v1.csv'); con=con[con.metric=='attack_nauc']
    plt.figure(figsize=(7.5,4.6))
    for i,f in enumerate(FAMILIES):
        vals=con[con.family==f].delta_vits_minus_vitb.to_numpy(); plt.scatter(np.full(len(vals),i)+np.linspace(-.08,.08,len(vals)),vals,label=LABELS[f])
    plt.axhline(0,linewidth=1); plt.xticks(range(3),[LABELS[f] for f in FAMILIES]); plt.ylabel('Delta adversarial nAUC (ViT-S minus ViT-B)'); plt.title('Seed-aligned architecture effects'); plt.tight_layout(); plt.savefig(args.out/'fig_D059_seed_attack_nauc_deltas.pdf'); plt.savefig(args.out/'fig_D059_seed_attack_nauc_deltas.png',dpi=300); plt.close()

if __name__=='__main__': main()
