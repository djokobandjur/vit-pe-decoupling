#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    out = ROOT / 'artifacts/reproduced'
    if not out.exists():
        raise SystemExit('Run scripts/reproduce_all.py first.')

    checks: list[dict] = []

    def check(name: str, condition: bool, detail: str = '') -> None:
        checks.append({'name': name, 'pass': bool(condition), 'detail': detail})
        print(('PASS' if condition else 'FAIL') + ': ' + name + (f' — {detail}' if detail else ''))

    exact_pairs = [
        ('canonical primary aggregate',
         ROOT / 'analysis/canonical_cohorts/reference_outputs/primary_aggregate_nauc_v18.csv',
         out / 'canonical/primary_aggregate_nauc_v18.csv'),
        ('canonical wider-range aggregate',
         ROOT / 'analysis/canonical_cohorts/reference_outputs/cifar_wide_aggregate_nauc_v18.csv',
         out / 'canonical/cifar_wide_aggregate_nauc_v18.csv'),
        ('canonical seed-level nAUC',
         ROOT / 'analysis/canonical_cohorts/reference_outputs/seed_level_nauc_v18.csv',
         out / 'canonical/seed_level_nauc_v18.csv'),
        ('canonical statistics',
         ROOT / 'analysis/canonical_cohorts/reference_outputs/canonical_statistics_v18.json',
         out / 'canonical/canonical_statistics_v18.json'),
        ('precision bridge summary',
         ROOT / 'analysis/precision_bridge/reference_outputs/D029_ANALYSIS_SUMMARY_v1.json',
         out / 'precision_bridge/D029_ANALYSIS_SUMMARY_v1.json'),
        ('precision bridge family endpoints',
         ROOT / 'analysis/precision_bridge/reference_outputs/D029_FAMILY_ENDPOINT_STATISTICS_v1.csv',
         out / 'precision_bridge/D029_FAMILY_ENDPOINT_STATISTICS_v1.csv'),
        ('cross-architecture main table',
         ROOT / 'analysis/cross_architecture/reference_outputs/D059_ATTACK_MAIN_TABLE_v2.csv',
         out / 'cross_architecture/D059_ATTACK_MAIN_TABLE_v2.csv'),
        ('cross-architecture summary',
         ROOT / 'analysis/cross_architecture/reference_outputs/D059_ANALYSIS_SUMMARY_v2.json',
         out / 'cross_architecture/D059_ANALYSIS_SUMMARY_v2.json'),
        ('cross-architecture Welch sensitivity',
         ROOT / 'analysis/cross_architecture/reference_outputs/D059_WELCH_SENSITIVITY_v2.csv',
         out / 'cross_architecture/D059_WELCH_SENSITIVITY_v2.csv'),
        ('cross-architecture ordinal reconstruction',
         ROOT / 'analysis/cross_architecture/reference_outputs/D059_INTERACTION_ORDINAL_ROBUSTNESS_v2.csv',
         out / 'cross_architecture/D059_INTERACTION_ORDINAL_ROBUSTNESS_v2.csv'),
        ('primary manuscript table',
         ROOT / 'manuscript/tables/table_primary_nauc.tex',
         out / 'manuscript_assets/table_primary_nauc.tex'),
        ('wider-range manuscript table',
         ROOT / 'manuscript/tables/table_cifar_wide.tex',
         out / 'manuscript_assets/table_cifar_wide.tex'),
    ]
    for name, reference, generated in exact_pairs:
        ok = reference.exists() and generated.exists() and sha256(reference) == sha256(generated)
        check(name, ok, f'{sha256(generated) if generated.exists() else "missing"}')

    qa = json.loads((out / 'cross_architecture/cross_architecture_QA.json').read_text())
    check('cross-architecture 31/31 QA', qa['status'] == 'PASS' and qa['passed'] == qa['n_checks'] == 31)

    report = {
        'status': 'PASS' if all(c['pass'] for c in checks) else 'FAIL',
        'n_checks': len(checks),
        'passed': sum(c['pass'] for c in checks),
        'checks': checks,
        'note': 'PDF visual identity is checked in the repository release audit; PDF byte hashes can differ because of metadata.'
    }
    (out / 'REPRODUCTION_VERIFICATION_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
