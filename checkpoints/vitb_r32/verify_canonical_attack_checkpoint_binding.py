#!/usr/bin/env python3
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
manifest=json.loads((ROOT/'checkpoints/vitb_r32/VITB_R32_CHECKPOINT_MANIFEST.json').read_text())
by_path={r['checkpoint_path']:r for r in manifest['checkpoints']}
seen=[]
for p in sorted((ROOT/'data/canonical_inputs').rglob('attacks_*.json')):
    d=json.loads(p.read_text())
    dataset='cifar100' if 'cifar_base' in p.parts else 'imagenet100'
    for family,seeds in d['results'].items():
        for seed_s,block in seeds.items():
            cp=block['checkpoint']; seed=int(seed_s)
            assert block.get('checkpoint_sha256') is None
            assert cp in by_path
            row=by_path[cp]
            assert row['dataset']==dataset and row['pe_family']==family and int(row['seed'])==seed
            assert len(row['sha256'])==64
            seen.append((dataset,family,seed,cp))
assert len(seen)==48 and len(set(seen))==48
print('PASS: 48/48 canonical ViT-B attack checkpoint paths bind to the post-hoc SHA-256 manifest')
print('NOTE: canonical attack JSONs record paths, not contemporaneous checkpoint digests')
