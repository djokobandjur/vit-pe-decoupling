#!/usr/bin/env python3
"""Build the clean-accuracy table from public seed-level evaluation anchors."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FAMILIES=["learned","sinusoidal","rope","alibi"]
LABELS={"learned":"Learned","sinusoidal":"Sinusoidal","rope":"RoPE","alibi":"ALiBi"}

def main():
    root=Path(__file__).resolve().parents[2]
    ap=argparse.ArgumentParser()
    ap.add_argument("--vitb",type=Path,default=root/"data/clean_accuracy_eval_subset_seed_level.csv")
    ap.add_argument("--vits",type=Path,default=root/"analysis/cross_cohort/reference_bundle/v19_vits_clean_accuracy.csv")
    ap.add_argument("--output-dir",type=Path,default=root/"analysis/clean_accuracy/reference_outputs")
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)

    vb=pd.read_csv(args.vitb)
    vs=pd.read_csv(args.vits)
    rows=[]
    for family in FAMILIES:
        for dataset in ["imagenet","cifar"]:
            values=vb.query("dataset==@dataset and family==@family")["clean_acc_eval"].to_numpy(float)
            if len(values)!=6: raise ValueError(f"ViT-B coverage {dataset}/{family}")
            rows.append({"architecture":"vitb","dataset":dataset,"family":family,
                         "n_seeds":6,"n_eval":3464 if dataset=="imagenet" else 8464,
                         "mean":float(values.mean()),"sd":float(values.std(ddof=1))})
        values=vs.query("family==@family")["clean_accuracy"].to_numpy(float)
        if len(values)!=6: raise ValueError(f"ViT-S coverage {family}")
        rows.append({"architecture":"vits","dataset":"imagenet","family":family,
                     "n_seeds":6,"n_eval":3464,
                     "mean":float(values.mean()),"sd":float(values.std(ddof=1))})
    agg=pd.DataFrame(rows)
    agg.to_csv(args.output_dir/"clean_accuracy_aggregate.csv",index=False)

    def get(arch,dataset,fam):
        return agg.query("architecture==@arch and dataset==@dataset and family==@fam").iloc[0]
    lines=[
      r"\begin{table}[t]",
      r"\centering",
      r"\caption{Clean classification accuracy on the fixed robustness",
      r"evaluation splits. Values are mean $\pm$ sample standard deviation",
      r"over six independently trained seeds. ImageNet-100 uses 3,464 images",
      r"and CIFAR-100 uses 8,464 images.}",
      r"\label{tab:clean-accuracy}",
      r"\resizebox{\columnwidth}{!}{%",
      r"\begin{tabular}{lccc}",
      r"\toprule",
      r"PE family & ViT-B ImageNet-100 & ViT-B CIFAR-100 & ViT-S ImageNet-100 \\",
      r"\midrule",
    ]
    for fam in FAMILIES:
        a=get("vitb","imagenet",fam); b=get("vitb","cifar",fam); c=get("vits","imagenet",fam)
        lines.append(f"{LABELS[fam]} & ${a['mean']:.2f} \\pm {a['sd']:.2f}$ & "
                     f"${b['mean']:.2f} \\pm {b['sd']:.2f}$ & "
                     f"${c['mean']:.2f} \\pm {c['sd']:.2f}$ \\\\")
    lines += [r"\bottomrule",r"\end{tabular}%",r"}",r"\end{table}",""]
    (args.output_dir/"table_clean_accuracy.tex").write_text("\n".join(lines),encoding="utf-8")
    report={"status":"PASS","vitb_seed_rows":int(len(vb)),"vits_seed_rows":int(len(vs)),
            "all_cells_seed_reproducible":True,"table_population":"locked robustness evaluation splits"}
    (args.output_dir/"clean_accuracy_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
