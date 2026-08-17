from pathlib import Path
from pypdf import PdfReader
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_manuscript_has_no_internal_d_codes():
    main = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(
        encoding="utf-8"
    )
    supplement = (
        ROOT / "manuscript/pe_robustness_nn_supplement.tex"
    ).read_text(encoding="utf-8")
    assert re.search(r"\bD\d{3}[a-z]?\b", main + supplement) is None


def test_cross_architecture_reference_qa_passes():
    report = read_json(
        "analysis/cross_architecture/reference_outputs/D059_V2_QA_REPORT.json"
    )
    assert report["status"] == "PASS"
    assert report["passed"] == report["n_checks"] == 31


def test_precision_bridge_reference_status():
    report = read_json(
        "analysis/precision_bridge/reference_outputs/D029_ANALYSIS_SUMMARY_v1.json"
    )
    assert report["status"] == "PASS_ANALYSIS_COMPLETE"
    assert report["bridge_decision"]["overall_status"] == "INCONCLUSIVE"


def test_public_precision_wording_authors_and_trainability_boundary():
    text = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(
        encoding="utf-8"
    )
    assert "AMP/" + "BF" + "16" not in text
    assert "AMP--FP16" in text
    assert r"Milo\v{s} Ban{\dj}ur" in text
    assert "djoko.bandjur@pr.ac.rs" in text
    assert "milos.bandjur@pr.ac.rs" in text
    assert "All " + "six ViT-S/16 AMP" not in text
    assert "did not yield valid checkpoints" in text


def test_no_envelope_sensitivity_passes():
    report = read_json(
        "analysis/canonical_cohorts/reference_outputs/"
        "no_envelope_sensitivity_report.json"
    )
    assert report["status"] == "PASS"
    assert report["seed_level_rows"] == 48
    assert report["all_48_raw_gaps_positive_at_0p09"]
    assert report["all_48_raw_gaps_positive_at_0p095"]
    assert report["all_within_seed_family_ranks_preserved"]


def test_postlock_grid_sensitivity_passes_and_strengthens_result():
    report = read_json(
        "analysis/cross_cohort/reference_outputs/"
        "postlock_grid_sensitivity_report.json"
    )
    assert report["status"] == "PASS"
    assert report["same_task_loss_objective"]
    assert report["same_schedule"] == "200x5"
    assert report["same_split_and_checkpoint_cohort"]
    assert report["all_seed_deltas_negative"]
    assert report["qualitative_conclusion_changed"] is False
    assert abs(report["primary_mean"] - 0.7663660668996223) < 1e-12
    assert abs(report["augmented_mean"] - 0.7217772442386754) < 1e-12


def test_local_geometry_public_evidence_passes():
    report = read_json(
        "analysis/local_geometry/reference_outputs/"
        "geometry_public_verification.json"
    )
    assert report["status"] == "PASS"
    assert report["direction_rows"] == 1584
    assert report["checkpoint_groups"] == 48
    assert report["task_directions"] == 48
    assert report["gaussian_directions"] == 1536
    assert report["probe_radius_verified"]


def test_clean_accuracy_table_is_seed_reproducible():
    report = read_json(
        "analysis/clean_accuracy/reference_outputs/clean_accuracy_report.json"
    )
    assert report["status"] == "PASS"
    assert report["vitb_seed_rows"] == 48
    assert report["vits_seed_rows"] == 24
    assert report["all_cells_seed_reproducible"]
    assert report["table_population"] == "locked robustness evaluation splits"


def test_all_central_tables_are_external_generated_inputs():
    main = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(
        encoding="utf-8"
    )
    supplement = (
        ROOT / "manuscript/pe_robustness_nn_supplement.tex"
    ).read_text(encoding="utf-8")
    table_names = [
        "table_primary_nauc",
        "table_cifar_wide",
        "table_clean_accuracy",
        "table_factorial_comparison_design",
        "table_cross_architecture_nauc",
        "table_amp_matched_nauc",
        "table_objective_comparison",
        "table_s_welch",
        "table_s_orders",
        "table_s_did",
        "table_s_dispersion",
        "table_s_signs",
        "table_no_envelope_sensitivity",
        "table_postlock_grid_sensitivity",
    ]
    combined = main + supplement
    for name in table_names:
        assert (
            f"\\input{{tables/{name}}}" in combined
            or f"\\input{{tables/{name}.tex}}" in combined
        )
        assert (ROOT / "manuscript/tables" / f"{name}.tex").exists()


def test_claim_evidence_matrix_is_well_formed_and_complete():
    with (
        ROOT / "audit/final_claim_to_evidence_matrix.csv"
    ).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert all(None not in row for row in rows)
    assert all(row["status"] == "PASS" for row in rows)


def test_pytest_is_in_all_documented_environments():
    assert "pytest" in (ROOT / "requirements.txt").read_text()
    assert "pytest" in (ROOT / "requirements-lock.txt").read_text()
    assert "pytest" in (ROOT / "environment.yml").read_text()


def test_training_source_scope_is_truthful():
    training = ROOT / "experiments/training"
    assert (training / "vitb_amp_training_source.py").exists()
    assert not (training / "vits_full_scale_experiment.py").exists()
    scope = (training / "README.md").read_text(encoding="utf-8")
    assert "ViT-S/16 full-FP32 and AMP production training launchers are **not**" in scope
    source = (training / "vitb_amp_training_source.py").read_text(encoding="utf-8")
    assert "'embed_dim': 768" in source
    assert "'num_heads': 12" in source
    assert "torch.amp.GradScaler('cuda')" in source
    assert "[42, 123, 456]" in source


def test_random_noise_middle_order_is_not_overclaimed():
    text = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
    assert "preserves the random-noise ranking" not in text
    assert "random-noise ranking can remain stable" not in text
    assert "random-noise ranking is stable" not in text
    assert text.count("random-noise extremes") >= 3


def test_geometry_provenance_and_radius_boundary():
    provenance = read_json(
        "analysis/local_geometry/reference_outputs/geometry_public_provenance.json"
    )
    assert provenance["status"] == "PASS_FULL_RAW_CHECKPOINT_PROCESSED_HASH_CHAIN"
    assert provenance["checkpoint_digests"]["available"] is True
    assert provenance["checkpoint_digests"]["checkpoint_records"] == 48
    assert provenance["checkpoint_digests"]["historical_imagenet_sinusoidal_matches"] == 6
    assert all(item["sha256"] for item in provenance["raw_source_files"])
    report = read_json(
        "analysis/local_geometry/reference_outputs/geometry_public_verification.json"
    )
    assert report["public_processed_artifact_hash_chain"] == "PASS"
    assert report["full_raw_checkpoint_processed_hash_chain"] == "PASS"
    assert report["raw_to_direction_level_reconstruction"] == "PASS_1584_OF_1584"
    assert report["checkpoint_digest_status"] == "PASS_48_OF_48"
    assert report["historical_checkpoint_crosscheck"] == "PASS_6_OF_6"
    assert report["radius_sensitivity_family_order_preserved"]


def test_orcid_dropout_clipping_and_postlock_closure():
    main = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
    supplement = (
        ROOT / "manuscript/pe_robustness_nn_supplement.tex"
    ).read_text(encoding="utf-8")
    assert "0000-0001-9034-6854" in main
    assert "global gradient-norm clipping at 1.0" in main
    assert "magnitudes are not scale invariant" in main
    assert "[0.096597,0.106636]" in supplement
    manifest = read_json("checkpoints/vits_fp32_checkpoint_manifest.json")
    assert manifest["architecture"]["dropout"] == 0.0
    assert manifest["architecture"]["training_recipe_dropout"] == 0.1



def test_v1_v2_historical_evidence_boundaries_are_public_and_truthful():
    main = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
    harm = read_json("audit/historical/vitb_harmonization/VITB_HARMONIZATION_PUBLIC_EVIDENCE.json")
    assert harm["status"] == "PUBLIC_SCOPE_DOCUMENTED"
    assert len(harm["public_evidence"]["registered_harmonization_runs"]) == 5
    assert "complete matched comparisons" not in main.lower()
    assert "0.005$ in selected attack loss" not in main
    flat = re.sub(r"\s+", " ", main)
    assert "do not claim public row-level reproduction" in flat
    hw = (ROOT / "audit/historical/cross_hardware/POINT_3_5_AUDIT_REPORT_CANONICAL_v10.md").read_text()
    assert "0.000491" in hw and "2.55718e-05" in hw
    assert "audit-recorded cross-hardware" in main
    assert "Blackwell log is not redistributed" in main
    assert (ROOT / "audit/historical/cross_hardware/D033A_PROVENANCE_REPORT.json").read_bytes() == (ROOT / "data/provenance_report.json").read_bytes()


def test_v3_v4_v5_v6_public_scope_and_manifests():
    main = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
    assert "48 canonical ViT-B/16 checkpoints" in main
    assert "The ViT-S/16 cohorts are not" in main
    assert "restricted to the 48 ViT-B/16 checkpoints" in main
    md = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in ROOT.rglob("*.md"))
    low = md.lower()
    assert "every attempted vit-s/16 amp training run diverged" not in low
    assert "all six attempted runs diverged" not in low
    amp = read_json("checkpoints/vits_amp/VITS_FP16_CHECKPOINT_MANIFEST.json")
    assert len(amp["rows"]) == 18
    assert {r["family"] for r in amp["rows"]} == {"learned", "sinusoidal", "rope"}
    protocol = read_json("experiments/configs/vits_in100_locked_protocol.json")
    assert "pending" not in protocol
    assert "Complete" in protocol["release_status"]["vits_full_n6"]


def test_v7_checkpoint_binding_direct_restart_and_supplement_callouts():
    main = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "manuscript/pe_robustness_nn_supplement.tex").read_text(encoding="utf-8")
    binding = read_json("checkpoints/vitb_r32/VITB_CANONICAL_ATTACK_CHECKPOINT_BINDING.json")
    assert binding["status"] == "PASS_48_OF_48_PATHS_BOUND_TO_DIGEST_MANIFEST"
    assert len(binding["records"]) == 48
    assert all(r["attack_json_embedded_checkpoint_sha256"] is None for r in binding["records"])
    assert "reconstructed post hoc" in main
    formal = read_json("data/objective_specific/direct_displacement/FORMAL_CONVERGENCE_RESULT.json")
    b = next(x for x in formal["budgets"] if abs(float(x["budget"])-0.02)<1e-12)
    r2 = next(x for x in b["paired_restart_comparison"] if x["restart"]==2)
    assert abs(r2["normalized_accuracy_200"]-0.668980635732554)<1e-12
    assert abs(r2["normalized_accuracy_400"]-0.6021191085129703)<1e-12
    assert abs(b["normalized_accuracy_absolute_delta"]-0.00036536353671901)<1e-12
    for label in ("tab:s-orders","tab:s-did","tab:s-dispersion","tab:s-signs"):
        assert f"\\ref{{{label}}}" in supplement


def test_v6_coverage_extension_numerical_mode_and_split_provenance():
    main = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
    cov = read_json("audit/historical/sinusoidal_audit/VITS_SINUSOIDAL_COVERAGE_EXTENSION_PUBLIC_EVIDENCE.json")
    assert cov["status"] == "PASS_EVIDENCE_BOUNDED"
    assert len(cov["records"]) == 3
    assert [r["seed"] for r in cov["records"]] == [123,456,1213]
    assert all(r["target_reached"] is False for r in cov["records"])
    assert all(r["absolute_difference_ledger_vs_canonical_max"] < 5e-10 for r in cov["records"])
    assert "COMPLETE\\_TARGET\\_NOT\\_REACHED" in main
    flat = re.sub(r"\s+", " ", main.lower())
    assert "original row-level coverage-extension jsons are not redistributed" in flat
    num = read_json("configs/attack_eval_numerical_mode.json")
    assert num["attack_forward_autocast"] is False
    assert num["attack_backward_autocast"] is False
    assert num["grad_scaler_during_attack"] is False
    assert num["sdpa_backend"] == "MATH"
    assert num["matmul_allow_tf32"] is True
    assert "not strict exclusion of TF32" in main
    lock = read_json("audit/historical/sinusoidal_audit/SINUSOIDAL_PER_RESTART_AUDIT_INPUT_LOCK_v1.json")
    assert lock["dataset_reports"]["vitb"]["sample_order_sha256"] == lock["dataset_reports"]["vits"]["sample_order_sha256"]
    assert lock["dataset_reports"]["vitb"]["split_sha256"] == lock["dataset_reports"]["vits"]["split_sha256"]


def test_v6_schedule_threshold_semantics_and_supplement_refs():
    main = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
    harm = read_json("audit/historical/vitb_harmonization/VITB_HARMONIZATION_PUBLIC_EVIDENCE.json")
    assert harm["historical_development_threshold_semantics"] == "EMPIRICAL_SEPARATION_NOT_A_PRIORI_THRESHOLD"
    flat = re.sub(r"\s+", " ", main)
    assert "did not use an a priori numerical convergence threshold" in flat
    assert "Table~S6" not in main and "Table~S7" not in main
    assert "\\ref{tab:s-no-envelope}" in main
    assert "\\ref{tab:s-postlock-grid}" in main
    assert "\\externaldocument{pe_robustness_nn_supplement}" in main


def test_v6_release_build_artifact_and_page_count_guards():
    assert "*.csv -text" in (ROOT/".gitattributes").read_text(encoding="utf-8")
    forbidden=(".aux",".log",".out",".spl",".synctex.gz")
    assert not [p for p in (ROOT/"manuscript").iterdir() if p.is_file() and p.name.endswith(forbidden)]
    sums=(ROOT/"SHA256SUMS_REPOSITORY.txt").read_text(encoding="utf-8")
    assert not any(line.strip().endswith(forbidden) for line in sums.splitlines())
    doc=(ROOT/"audit/manuscript_internal_label_removal.md").read_text(encoding="utf-8")
    for label,pdf in (("Main manuscript","pe_robustness_nn_main.pdf"),("Supplement","pe_robustness_nn_supplement.pdf")):
        m=re.search(rf"{label}:\s*(\d+)\s*pages",doc)
        assert m
        assert len(PdfReader(str(ROOT/"manuscript"/pdf)).pages)==int(m.group(1))


def test_v7_annotated_numerical_mode_resolves_locked_artifact():
    annotated = read_json("configs/attack_eval_numerical_mode.json")
    ref = annotated["locked_artifact_reference"]
    locked_path = ROOT / ref["path"]
    assert locked_path.exists()
    import hashlib
    actual = hashlib.sha256(locked_path.read_bytes()).hexdigest()
    final = read_json("data/objective_specific/all_restart_audit/FINAL_AUDIT_REPORT_v1.json")
    assert actual == ref["sha256"] == final["locks"]["ATTACK_EVAL_NUMERICAL_MODE_v1.json"]
    assert "annotated reviewer-facing copy" in ref["note"]


def test_v7_all_claim_matrix_evidence_paths_resolve():
    import csv
    with (ROOT / "audit/final_claim_to_evidence_matrix.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    unresolved = []
    for row in rows:
        for token in (part.strip() for part in row["evidence"].split(";")):
            if any(ch in token for ch in "*?["):
                if not list(ROOT.glob(token)):
                    unresolved.append((row["gate_id"], token))
            elif not (ROOT / token).exists():
                unresolved.append((row["gate_id"], token))
    assert unresolved == []


def test_v7_math_sdpa_claim_is_locked_environment_conditioned():
    main = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", main)
    assert "where the installed PyTorch exposes" in flat
    assert "as in the locked PyTorch 2.8 audit environment" in flat
    assert "older-PyTorch fallback paths are not asserted to force MATH" in flat
    annotated = read_json("configs/attack_eval_numerical_mode.json")
    assert "older-PyTorch fallback is not claimed to force MATH" in annotated["claim_semantics"]["sdpa_backend_scope"]


def test_v7_all_restart_lock_artifacts_resolve_byte_exactly():
    import hashlib
    final = read_json("data/objective_specific/all_restart_audit/FINAL_AUDIT_REPORT_v1.json")
    paths = {
        "ATTACK_EVAL_NUMERICAL_MODE_v1.json": ROOT / "audit/historical/sinusoidal_audit/ATTACK_EVAL_NUMERICAL_MODE_v1.json",
        "SINUSOIDAL_PER_RESTART_AUDIT_INPUT_LOCK_v1.json": ROOT / "audit/historical/sinusoidal_audit/SINUSOIDAL_PER_RESTART_AUDIT_INPUT_LOCK_v1.json",
        "D033A_PROVENANCE_REPORT.json": ROOT / "audit/historical/sinusoidal_audit/D033A_PROVENANCE_REPORT.json",
    }
    for name, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == final["locks"][name]


def test_v8_abstract_texcount_metadata_is_consistent():
    import json, re, shutil, subprocess, tempfile
    label_doc = (ROOT / "audit/manuscript_internal_label_removal.md").read_text(encoding="utf-8")
    m = re.search(r"Abstract:\s*(\d+)\s*words by texcount", label_doc)
    assert m
    label_count = int(m.group(1))
    release = json.loads((ROOT / "audit/PUBLIC_REPOSITORY_RELEASE_AUDIT_v1.0.0.json").read_text(encoding="utf-8"))
    release_count = int(release["verification"]["manuscript_build"]["abstract_words_texcount"])
    assert label_count == release_count == 236
    texcount = shutil.which("texcount")
    if texcount:
        main = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
        abstract = re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", main, flags=re.S)
        assert abstract
        name = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".tex", encoding="utf-8", delete=False) as h:
                h.write(abstract.group(0))
                name = h.name
            out = subprocess.run([texcount, "-brief", name], check=True, capture_output=True, text=True).stdout
            live = re.search(r"(^|\n)\s*(\d+)\+\d+\+\d+", out)
            assert live
            assert int(live.group(2)) == 236
        finally:
            if name:
                Path(name).unlink(missing_ok=True)
