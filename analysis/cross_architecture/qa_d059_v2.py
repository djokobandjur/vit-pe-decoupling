#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def rows(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def near(a: float, b: float, tol: float = 5e-7) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    checks = []

    def check(name, condition, detail=''):
        checks.append({'name': name, 'pass': bool(condition), 'detail': detail})

    main_table = rows(out/'D059_ATTACK_MAIN_TABLE_v2.csv')
    by_f = {r['family']: r for r in main_table}
    check('three attack families', set(by_f)=={'learned','sinusoidal','rope'})
    check('learned sign count 6', int(float(by_f['learned']['negative_deltas']))==6)
    check('sinusoidal sign count 6', int(float(by_f['sinusoidal']['negative_deltas']))==6)
    check('rope sign count 5', int(float(by_f['rope']['negative_deltas']))==5)
    check('learned p floor', near(float(by_f['learned']['exact_sign_flip_p_two_sided']),0.03125,1e-12))
    check('sinusoidal p floor', near(float(by_f['sinusoidal']['exact_sign_flip_p_two_sided']),0.03125,1e-12))
    check('rope adjacent rung', near(float(by_f['rope']['exact_sign_flip_p_two_sided']),0.0625,1e-12))

    ordinal = rows(out/'D059_INTERACTION_ORDINAL_ROBUSTNESS_v2.csv')
    check('ordinal has six seeds', len(ordinal)==6)
    check('modal order 5 of 6', sum(r['matches_modal_order']=='True' for r in ordinal)==5)
    check('seed42 exception', [int(r['seed']) for r in ordinal if r['matches_modal_order']!='True']==[42])
    rank_sums = {
        f: sum(float(r[f'{f}_rank']) for r in ordinal)
        for f in ['learned','sinusoidal','rope']
    }
    check('rank sums 8 11 17', rank_sums=={'learned':8.0,'sinusoidal':11.0,'rope':17.0}, str(rank_sums))

    did = {r['contrast']: r for r in rows(out/'D059_INTERACTION_PAIRWISE_DID_v2.csv')}
    check('DID learned-rope 5/6', int(float(did['learned_minus_rope']['same_sign_seeds']))==5)
    check('DID sinusoidal-rope 6/6', int(float(did['sinusoidal_minus_rope']['same_sign_seeds']))==6)
    check('DID learned-sinusoidal 5/6', int(float(did['learned_minus_sinusoidal']['same_sign_seeds']))==5)
    check('DID sinusoidal-rope CI below zero', float(did['sinusoidal_minus_rope']['ci95_t_high'])<0)

    welch = rows(out/'D059_WELCH_SENSITIVITY_v2.csv')
    attacks = [r for r in welch if r['metric']=='attack_nauc']
    check('Welch three attack families', len(attacks)==3)
    check('Welch attack intervals below zero', all(float(r['welch_ci95_high'])<0 for r in attacks))
    check('paired and Welch same direction', all(r['paired_welch_same_direction']=='True' for r in attacks))

    disp = {r['family']: r for r in rows(out/'D059_ARCHITECTURE_DISPERSION_AUDIT_v2.csv')}
    check('Learned SD ratio ~29.41', abs(float(disp['learned']['sd_ratio_vits_over_vitb'])-29.4139274584)<1e-6)
    check('RoPE SD ratio ~4.665', abs(float(disp['rope']['sd_ratio_vits_over_vitb'])-4.66476174395)<1e-6)
    check('Sinusoidal SD ratio ~1.097', abs(float(disp['sinusoidal']['sd_ratio_vits_over_vitb'])-1.09708557964)<1e-6)
    check('Learned LOO ratio stays > 19', float(disp['learned']['loo_sd_ratio_min'])>19)

    signs = {(r['family'],r['metric']):r for r in rows(out/'D059_ATTACK_NOISE_SIGN_BALANCE_v2.csv')}
    check('noise Learned 3 negative', int(float(signs[('learned','noise_nauc')]['negative_seeds']))==3)
    check('noise Sinusoidal 2 negative', int(float(signs[('sinusoidal','noise_nauc')]['negative_seeds']))==2)
    check('noise RoPE 3 negative', int(float(signs[('rope','noise_nauc')]['negative_seeds']))==3)

    above = rows(out/'D059_NORMALIZED_NAUC_ABOVE_ONE_AUDIT_v2.csv')
    check('one nAUC above one record', len(above)==1)
    if above:
        r=above[0]
        check('above-one record is ViT-S sinusoidal seed123', r['architecture']=='vits_amp' and r['family']=='sinusoidal' and int(r['seed'])==123)
        check('above-one value ~1.0009374', abs(float(r['noise_nauc'])-1.000937403793558)<1e-12)

    summary=json.loads((out/'D059_ANALYSIS_SUMMARY_v2.json').read_text())
    check('summary no new GPU', summary['new_gpu_experiment_required'] is False)
    check('summary revision v2', summary['revision']=='D059_REANALYSIS_v2')
    check('sign-flip step 2/64', near(summary['sign_flip_quantisation']['step'],2/64,1e-15))

    status = 'PASS' if all(c['pass'] for c in checks) else 'FAIL'
    report={'status':status,'n_checks':len(checks),'passed':sum(c['pass'] for c in checks),'checks':checks}
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2))
    if status!='PASS':
        raise SystemExit(1)

if __name__=='__main__':
    main()
