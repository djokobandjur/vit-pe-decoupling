# S1 audit closeout — v8

**Status: PASS**

The distributed abstract is byte-identical to v7. Direct `texcount -brief` on the extracted `abstract` environment returns **237** words. `audit/manuscript_internal_label_removal.md` already reported 237; the stale value was `242` in `audit/PUBLIC_REPOSITORY_RELEASE_AUDIT_v1.0.0.json`, which is corrected to 237.

A regression guard now requires the two audit records to agree. When `texcount` is available, `scripts/verify_repository.py` additionally extracts the abstract from `manuscript/pe_robustness_nn_main.tex`, runs `texcount`, and requires the live result to equal the recorded value.

No manuscript text, numerical result, table, figure, or scientific claim changed.
