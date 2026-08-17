# R1-R3 final audit closeout — v7

**Status:** PASS — 3/3 CLOSED

## R1 — annotated versus locked numerical-mode record

`configs/attack_eval_numerical_mode.json` is an annotated reviewer-facing copy. It now points explicitly to the byte-exact lock:

- `audit/historical/sinusoidal_audit/ATTACK_EVAL_NUMERICAL_MODE_v1.json`
- SHA-256 `df0341e1953c9457bdf9dec409bbdd7f122246f85204f0b999ebc666a70748f7`

The verifier requires that this digest match both the referenced file bytes and the `ATTACK_EVAL_NUMERICAL_MODE_v1.json` lock in `FINAL_AUDIT_REPORT_v1.json`.

## R2 — claim-matrix evidence resolvability

V20-G29 now names `manuscript/pe_robustness_nn_main.tex` and `manuscript/pe_robustness_nn_supplement.tex` explicitly. All evidence tokens in all 29 PASS gates are checked for existence; wildcard entries must resolve to at least one repository file.

## R3 — MATH-SDPA scope

The manuscript now states that MATH SDPA is explicitly selected where `torch.nn.attention` is available, as in the locked PyTorch 2.8 audit environment. The older-PyTorch fallback is explicitly not claimed to force MATH.

## External pre-submission item

Public GitHub visibility is an external hosting state and cannot be certified by the offline repository archive. It is listed as a manual pre-submission check in `audit/PRE_SUBMISSION_CHECKLIST.md`.

## Regression status

- repository verifier: PASS (168 checks)
- repository manifest: 370/370 PASS
- pytest: 25/25 PASS
- all-restart lock artifacts: 3/3 byte-exact PASS
- full GPU-free reproduction: PASS
- exact-output checks: 35/35 PASS
- manuscript: 46 pages
- supplement: 5 pages
