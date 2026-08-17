#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "environment.yml",
        "requirements.txt",
        "requirements-lock.txt",
        "manuscript/pe_robustness_nn_main.tex",
        "manuscript/pe_robustness_nn_supplement.tex",
        "analysis/canonical_cohorts/canonical_analysis.py",
        "analysis/canonical_cohorts/build_no_envelope_sensitivity.py",
        "analysis/canonical_cohorts/reference_outputs/no_envelope_sensitivity_report.json",
        "analysis/cross_cohort/build_postlock_grid_sensitivity.py",
        "analysis/cross_cohort/reference_outputs/postlock_grid_sensitivity_report.json",
        "analysis/clean_accuracy/build_clean_accuracy_table.py",
        "analysis/clean_accuracy/reference_outputs/clean_accuracy_report.json",
        "analysis/local_geometry/verify_geometry_analysis.py",
        "analysis/local_geometry/reference_outputs/geometry_direction_level.csv",
        "analysis/local_geometry/reference_outputs/geometry_seed_level.csv",
        "analysis/local_geometry/reference_outputs/geometry_aggregate.csv",
        "analysis/local_geometry/reference_outputs/probe_radius_evidence.json",
        "analysis/local_geometry/reference_outputs/geometry_public_verification.json",
        "analysis/local_geometry/reference_outputs/geometry_public_provenance.json",
        "analysis/local_geometry/reference_outputs/geometry_radius_sensitivity.csv",
        "analysis/local_geometry/raw_sources/SHA256_ALL_ORIGINAL_FILES.txt",
        "analysis/local_geometry/raw_sources/imagenet_seeds_42_123.json",
        "audit/historical/vitb_harmonization/run_vitb_harmonization.py",
        "audit/historical/vitb_harmonization/EXPERIMENT_RUN_LEDGER_v1.csv",
        "audit/historical/vitb_harmonization/VITB_HARMONIZATION_PUBLIC_EVIDENCE.json",
        "audit/historical/cross_hardware/POINT_3_5_AUDIT_REPORT_CANONICAL_v10.md",
        "audit/historical/cross_hardware/D033A_PROVENANCE_REPORT.json",
        "audit/historical/sinusoidal_audit/ATTACK_EVAL_NUMERICAL_MODE_v1.json",
        "audit/historical/sinusoidal_audit/SINUSOIDAL_PER_RESTART_AUDIT_INPUT_LOCK_v1.json",
        "audit/historical/sinusoidal_audit/ATTACK_EVAL_NUMERICAL_MODE_POLICY.md",
        "audit/historical/sinusoidal_audit/VITS_SINUSOIDAL_COVERAGE_EXTENSION_PUBLIC_EVIDENCE.json",
        "manuscript/pe_robustness_nn_main.pdf",
        "manuscript/pe_robustness_nn_supplement.pdf",
        "checkpoints/vitb_r32/VITB_CANONICAL_ATTACK_CHECKPOINT_BINDING.json",
        "checkpoints/vits_amp/VITS_FP16_CHECKPOINT_MANIFEST.json",
        "analysis/local_geometry/raw_sources/cifar_seeds_1011_1213.json",
        "analysis/local_geometry/execution_provenance/code/cross_family_rho_decoupling.py",
        "checkpoints/vitb_r32/VITB_R32_CHECKPOINT_MANIFEST.json",
        "checkpoints/vitb_r32/VITB_R32_CHECKPOINT_VERIFICATION.json",
        "audit/N1_N5_AUDIT_CLOSEOUT.json",
        "audit/V1_V7_AUDIT_CLOSEOUT.json",
        "audit/N1_N6_AUDIT_CLOSEOUT_v6.json",
        "audit/N1_N6_AUDIT_CLOSEOUT_v6.md",
        "audit/PRE_SUBMISSION_CHECKLIST.md",
        "audit/R1_R3_AUDIT_CLOSEOUT_v7.json",
        "audit/R1_R3_AUDIT_CLOSEOUT_v7.md",
        "audit/S1_AUDIT_CLOSEOUT_v8.json",
        "audit/S1_AUDIT_CLOSEOUT_v8.md",
        "experiments/training/README.md",
        "experiments/training/vitb_amp_training_source.py",
        "analysis/precision_bridge/run_precision_bridge.py",
        "analysis/cross_architecture/build_amp_matched_cross_architecture.py",
        "audit/trainability_boundary_governance.json",
        "audit/final_claim_to_evidence_matrix.csv",
        "scripts/generate_manuscript_tables.py",
        "scripts/generate_repository_manifest.py",
        "data/vits_fp32_results.zip",
        "data/vits_amp_results.zip",
    ]
    table_names = [
        "table_primary_nauc.tex",
        "table_cifar_wide.tex",
        "table_clean_accuracy.tex",
        "table_factorial_comparison_design.tex",
        "table_cross_architecture_nauc.tex",
        "table_amp_matched_nauc.tex",
        "table_objective_comparison.tex",
        "table_s_welch.tex",
        "table_s_orders.tex",
        "table_s_did.tex",
        "table_s_dispersion.tex",
        "table_s_signs.tex",
        "table_no_envelope_sensitivity.tex",
        "table_postlock_grid_sensitivity.tex",
    ]
    required.extend(f"manuscript/tables/{name}" for name in table_names)
    for relative in required:
        check((ROOT / relative).exists(), f"required file exists: {relative}")

    manuscript = (
        ROOT / "manuscript/pe_robustness_nn_main.tex"
    ).read_text(encoding="utf-8")
    supplement = (
        ROOT / "manuscript/pe_robustness_nn_supplement.tex"
    ).read_text(encoding="utf-8")
    reviewer_text = manuscript + "\n" + supplement

    check(
        re.search(r"\bD\d{3}[a-z]?\b", reviewer_text) is None,
        "reviewer-facing sources contain no internal D-codes",
    )
    check(
        "canonical v19 reproducibility package" not in reviewer_text.lower(),
        "reviewer-facing sources contain no obsolete package-version wording",
    )
    check(
        "AMP/" + "BF" + "16" not in reviewer_text,
        "reviewer-facing sources use the locked FP16 wording",
    )
    check(
        "All " + "six ViT-S/16 AMP" not in manuscript
        and "all " + "six attempted runs diverged" not in manuscript.lower(),
        "abstract avoids an unsupported exact ALiBi failure count",
    )
    check(
        "did not yield valid checkpoints" in manuscript,
        "abstract uses the cohort-level ALiBi trainability wording",
    )
    markdown_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in ROOT.rglob("*.md")
        if ".git" not in path.parts
    ).lower()
    unsupported_trainability_phrases = (
        "every attempted vit-s/16 amp training run diverged",
        "all six attempted runs diverged",
        "all six vit-s/16 amp",
    )
    check(
        all(phrase not in markdown_text for phrase in unsupported_trainability_phrases),
        "repository-wide Markdown avoids unsupported exact ViT-S AMP ALiBi failure wording",
    )

    gitattributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    check("*.csv -text" in gitattributes, "Git preserves CSV evidence bytes without line-ending normalization")

    forbidden_release_suffixes = (".aux", ".log", ".out", ".spl", ".synctex.gz")
    leaked_build = [
        str(p.relative_to(ROOT)) for p in (ROOT / "manuscript").iterdir()
        if p.is_file() and any(p.name.endswith(s) for s in forbidden_release_suffixes)
    ]
    check(not leaked_build, "release tree excludes LaTeX build artifacts ignored by Git")
    sums_path = ROOT / "SHA256SUMS_REPOSITORY.txt"
    if sums_path.exists():
        sums_text = sums_path.read_text(encoding="utf-8")
        check(
            not any(any(line.strip().endswith(s) for s in forbidden_release_suffixes) for line in sums_text.splitlines()),
            "repository SHA-256 manifest excludes ignored LaTeX build artifacts",
        )
        manifest_records = []
        for line in sums_text.splitlines():
            if not line.strip():
                continue
            digest, relative = line.split("  ", 1)
            manifest_records.append((digest, relative))
        actual_files = sorted(
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file()
            and path.name != "SHA256SUMS_REPOSITORY.txt"
            and ".git" not in path.parts
            and "dist" not in path.parts
            and "artifacts" not in path.parts
            and "build" not in path.parts
            and ".pytest_cache" not in path.parts
            and "__pycache__" not in path.parts
            and not any(path.name.endswith(s) for s in forbidden_release_suffixes)
        )
        listed_files = sorted(relative for _, relative in manifest_records)
        check(listed_files == actual_files, "repository SHA-256 manifest enumerates the complete release tree")
        bad_hashes = []
        for expected, relative in manifest_records:
            h = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if h != expected:
                bad_hashes.append(relative)
        check(not bad_hashes, "repository SHA-256 manifest hashes all release files correctly")

    label_doc = (ROOT / "audit/manuscript_internal_label_removal.md").read_text(encoding="utf-8")
    for label, pdf_name in (("Main manuscript", "pe_robustness_nn_main.pdf"), ("Supplement", "pe_robustness_nn_supplement.pdf")):
        m = re.search(rf"{re.escape(label)}:\s*(\d+)\s*pages", label_doc)
        check(m is not None, f"{label} page-count claim exists in audit record")
        actual = len(PdfReader(str(ROOT / "manuscript" / pdf_name)).pages)
        check(actual == int(m.group(1)), f"{label} page count matches audit record")

    abstract_claim = re.search(r"Abstract:\s*(\d+)\s*words by texcount", label_doc)
    check(abstract_claim is not None, "abstract texcount claim exists in label-removal audit")
    release_audit = load_json("audit/PUBLIC_REPOSITORY_RELEASE_AUDIT_v1.0.0.json")
    release_abstract_words = int(release_audit["verification"]["manuscript_build"]["abstract_words_texcount"])
    label_abstract_words = int(abstract_claim.group(1))
    check(
        label_abstract_words == release_abstract_words,
        "abstract texcount claims agree across audit records",
    )

    texcount_bin = shutil.which("texcount")
    if texcount_bin:
        main_source = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
        abstract_match = re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", main_source, flags=re.S)
        check(abstract_match is not None, "abstract environment is extractable for texcount verification")
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".tex", encoding="utf-8", delete=False) as handle:
                handle.write(abstract_match.group(0))
                temp_name = handle.name
            proc = subprocess.run(
                [texcount_bin, "-brief", temp_name],
                check=True,
                capture_output=True,
                text=True,
            )
            live_match = re.search(r"(^|\n)\s*(\d+)\+\d+\+\d+", proc.stdout)
            check(live_match is not None, "texcount output for abstract is parseable")
            live_words = int(live_match.group(2))
            check(
                live_words == release_abstract_words,
                f"live texcount abstract count matches audit records ({live_words})",
            )
        finally:
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)
    else:
        print("INFO: texcount unavailable; cross-record abstract-count consistency still verified")
    coverage = load_json("audit/historical/sinusoidal_audit/VITS_SINUSOIDAL_COVERAGE_EXTENSION_PUBLIC_EVIDENCE.json")
    check(coverage["status"] == "PASS_EVIDENCE_BOUNDED" and len(coverage["records"]) == 3,
          "ViT-S Sinusoidal coverage-extension evidence is bounded and complete at ledger level")
    check(all(not r["target_reached"] for r in coverage["records"]),
          "all three registered ViT-S Sinusoidal coverage-extension targets were not reached")
    numerical = load_json("configs/attack_eval_numerical_mode.json")
    check(numerical["matmul_allow_tf32"] is True and numerical["attack_forward_autocast"] is False
          and numerical["attack_backward_autocast"] is False and numerical["grad_scaler_during_attack"] is False,
          "numerical-mode wording matches float32/no-autocast with TF32 permitted")
    split_cfg = load_json("configs/data_splits.json")["ImageNet-100"]
    check(split_cfg["validation_copy_provenance"]["verified_same_sample_order"]
          and split_cfg["validation_copy_provenance"]["verified_same_split"],
          "ViT-B and ViT-S ImageNet validation copies are hash-verified equivalent for ordering and split")
    check("Table~S6" not in manuscript and "Table~S7" not in manuscript and "Supplementary Table~S7" not in manuscript,
          "main manuscript uses label-based Supplement table references")
    check("\\ref{tab:s-no-envelope}" in manuscript and "\\ref{tab:s-postlock-grid}" in manuscript,
          "main manuscript references Supplement S6/S7 by labels")

    check(
        r"Milo\v{s} Ban{\dj}ur" in manuscript
        and "djoko.bandjur@pr.ac.rs" in manuscript
        and "milos.bandjur@pr.ac.rs" in manuscript
        and "0000-0001-9034-6854" in manuscript
        and "0009-0007-0124-3943" in manuscript,
        "reviewer-facing manuscript contains both authors, emails, and ORCIDs",
    )

    prohibited_ranking_phrases = (
        "preserves the random-noise ranking",
        "random-noise ranking can remain stable",
        "random-noise ranking is stable",
    )
    check(
        all(phrase not in manuscript for phrase in prohibited_ranking_phrases),
        "qualified random-noise-extremes wording is propagated globally",
    )
    check(
        manuscript.count("random-noise extremes") >= 3,
        "random-noise extremes are stated in introduction, contributions, and conclusion",
    )

    training_dir = ROOT / "experiments/training"
    check(
        not (training_dir / "vits_full_scale_experiment.py").exists(),
        "misleading byte-identical ViT-S training-source duplicate is absent",
    )
    vitb_source = (training_dir / "vitb_amp_training_source.py").read_text(
        encoding="utf-8"
    )
    check(
        "torch.amp.GradScaler('cuda')" in vitb_source
        and "torch.amp.autocast('cuda')" in vitb_source
        and "'embed_dim': 768" in vitb_source
        and "'num_heads': 12" in vitb_source
        and "default=256" in vitb_source
        and "[42, 123, 456]" in vitb_source,
        "public training driver is explicitly identified as historical ViT-B AMP source",
    )
    training_scope = (training_dir / "README.md").read_text(encoding="utf-8")
    check(
        "ViT-S/16 full-FP32 and AMP production training launchers are **not**"
        in training_scope
        and "not the locked six-seed" in training_scope,
        "training-source claim boundary is documented",
    )

    harmonization = load_json(
        "audit/historical/vitb_harmonization/VITB_HARMONIZATION_PUBLIC_EVIDENCE.json"
    )
    check(
        harmonization["status"] == "PUBLIC_SCOPE_DOCUMENTED"
        and len(harmonization["public_evidence"]["registered_harmonization_runs"]) == 5
        and "row-level" in harmonization["public_scope_limit"],
        "ViT-B historical harmonization evidence boundary is explicit",
    )
    check(
        "complete matched comparisons" not in manuscript.lower()
        and "0.005$ in selected attack loss" not in manuscript
        and "do not claim public row-level" in manuscript and "reproduction" in manuscript,
        "manuscript does not overclaim public ViT-B harmonization evidence",
    )
    hardware_summary = (
        ROOT / "audit/historical/cross_hardware/POINT_3_5_AUDIT_REPORT_CANONICAL_v10.md"
    ).read_text(encoding="utf-8")
    check(
        "0.000491" in hardware_summary
        and "2.55718e-05" in hardware_summary
        and "audit-recorded cross-hardware" in manuscript
        and "Blackwell log is not redistributed" in manuscript,
        "cross-hardware claim is numerically supported and availability-bounded",
    )
    check(
        (ROOT / "audit/historical/cross_hardware/D033A_PROVENANCE_REPORT.json").read_bytes()
        == (ROOT / "data/provenance_report.json").read_bytes(),
        "named D033A provenance report is distributed byte-identically",
    )

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        manuscript,
        re.S,
    )
    check(abstract_match is not None, "abstract block is present")
    abstract_text = abstract_match.group(1)
    abstract_text = re.sub(
        r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}", r"\1", abstract_text
    )
    abstract_text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", abstract_text)
    abstract_text = re.sub(r"[$\\{}_^~]", " ", abstract_text)
    abstract_words = re.findall(
        r"[A-Za-zÀ-ž0-9]+(?:[-–][A-Za-zÀ-ž0-9]+)*", abstract_text
    )
    check(
        len(abstract_words) <= 250,
        f"abstract remains within the 250-word Article limit ({len(abstract_words)} conservative tokens)",
    )

    for label in (
        "tab:cross-architecture-nauc",
        "tab:amp-matched-nauc",
        "fig:cross-architecture-curves",
        "fig:amp-matched-nauc",
        "fig:layerwise-displacement-profiles",
    ):
        check(
            f"\\ref{{{label}}}" in manuscript,
            f"reviewer-facing manuscript calls out {label}",
        )

    for label in ("tab:s-orders", "tab:s-did", "tab:s-dispersion", "tab:s-signs"):
        check(
            f"\\ref{{{label}}}" in supplement,
            f"supplement calls out {label}",
        )

    for name in table_names[:7]:
        check(
            f"\\input{{tables/{name.removesuffix('.tex')}}}" in manuscript
            or f"\\input{{tables/{name}}}" in manuscript,
            f"main manuscript inputs generated table {name}",
        )
    for name in table_names[7:]:
        check(
            f"\\input{{tables/{name.removesuffix('.tex')}}}" in supplement
            or f"\\input{{tables/{name}}}" in supplement,
            f"supplement inputs generated table {name}",
        )

    check(
        "ViT-S/16 on ImageNet-100" in manuscript
        and "$75{,}648$" in manuscript
        and "$6{,}304$" in manuscript,
        "parameter-space degrees of freedom are stated for ViT-S/16",
    )
    check(
        "prespecified canonical native-budget" in manuscript
        and "post-lock" in manuscript.lower()
        and "0.721777" in manuscript,
        "post-lock task-loss-grid sensitivity is disclosed in the main text",
    )
    check(
        r"\path{analysis/local_geometry/reference_outputs/probe_radius_evidence.json}"
        in manuscript
        and "1,584 direction records" in manuscript,
        "local-geometry numerical evidence is explicitly identified",
    )
    check(
        "48 canonical ViT-B/16 checkpoints" in manuscript
        and "The ViT-S/16 cohorts are not" in manuscript
        and "restricted to the 48 ViT-B/16 checkpoints" in manuscript,
        "local geometry manuscript scope is explicitly ViT-B/16 only",
    )
    check(
        "finite-radius local gain" in manuscript
        and "magnitudes are not scale invariant" in manuscript,
        "local gain ratio is explicitly qualified as radius dependent",
    )
    check(
        "global gradient-norm clipping at 1.0" in manuscript,
        "gradient clipping is documented in the training recipe",
    )
    check(
        "[0.096597,0.106636]" in supplement
        and "cannot alter" in supplement,
        "ViT-B post-lock points are closed above the reported endpoints",
    )
    check(
        "3,464 images" in (ROOT / "manuscript/tables/table_clean_accuracy.tex").read_text()
        and "8,464 images" in (ROOT / "manuscript/tables/table_clean_accuracy.tex").read_text(),
        "clean-accuracy table identifies the public evaluation populations",
    )

    canonical = list(
        csv.DictReader(
            (
                ROOT
                / "analysis/canonical_cohorts/reference_outputs/primary_aggregate_nauc_v18.csv"
            ).open(encoding="utf-8")
        )
    )
    row = next(
        item
        for item in canonical
        if item["dataset"] == "cifar" and item["pe_family"] == "alibi"
    )
    check(
        abs(float(row["noise_nauc_mean"]) - 0.993525541) < 1e-8,
        "canonical CIFAR-100 ALiBi nAUC sentinel",
    )

    precision = load_json(
        "analysis/precision_bridge/reference_outputs/D029_ANALYSIS_SUMMARY_v1.json"
    )
    check(
        precision["status"] == "PASS_ANALYSIS_COMPLETE",
        "precision bridge status",
    )
    statuses = {
        item["family"]: item["equivalence_status"]
        for item in precision["family_rho50_statistics"]
    }
    check(
        statuses
        == {
            "learned": "INCONCLUSIVE",
            "sinusoidal": "EQUIVALENT",
            "rope": "INCONCLUSIVE",
        },
        "family-specific precision-bridge decisions",
    )

    qa = load_json(
        "analysis/cross_architecture/reference_outputs/D059_V2_QA_REPORT.json"
    )
    check(
        qa["status"] == "PASS" and qa["passed"] == qa["n_checks"] == 31,
        "AMP-matched cross-architecture QA",
    )

    restart = load_json(
        "data/objective_specific/all_restart_audit/FINAL_AUDIT_REPORT_v1.json"
    )
    check(
        restart.get("status")
        in {"PASS", "ok", "PASS_AUDIT_COMPLETE", "PASS_COMPLETE"},
        "all-restart audit status",
    )

    direct = load_json(
        "data/objective_specific/direct_displacement/D020_VITS_FORMAL_CENTRAL_ENDPOINT_v1.json"
    )
    direct_text = json.dumps(direct)
    check(
        "0.0095" in direct_text and "0.02" in direct_text,
        "direct-displacement central endpoints present",
    )
    formal = load_json(
        "data/objective_specific/direct_displacement/FORMAL_CONVERGENCE_RESULT.json"
    )
    budget_002 = next(item for item in formal["budgets"] if abs(float(item["budget"]) - 0.02) < 1e-12)
    restart_2 = next(item for item in budget_002["paired_restart_comparison"] if item["restart"] == 2)
    check(
        abs(float(restart_2["normalized_accuracy_200"]) - 0.668980635732554) < 1e-12
        and abs(float(restart_2["normalized_accuracy_400"]) - 0.6021191085129703) < 1e-12
        and abs(float(budget_002["normalized_accuracy_absolute_delta"]) - 0.00036536353671901) < 1e-12
        and "0.66898" in manuscript and "0.60212" in manuscript and "0.000365" in manuscript,
        "direct-displacement restart-level non-convergence is explicitly quantified",
    )

    no_envelope = load_json(
        "analysis/canonical_cohorts/reference_outputs/no_envelope_sensitivity_report.json"
    )
    check(no_envelope["status"] == "PASS", "no-envelope sensitivity status")
    check(
        no_envelope["all_48_raw_gaps_positive_at_0p09"]
        and no_envelope["all_48_raw_gaps_positive_at_0p095"],
        "no-envelope gaps remain positive 48/48 at both endpoints",
    )
    check(
        no_envelope["all_within_seed_family_ranks_preserved"],
        "no-envelope family ranks are preserved",
    )

    postlock = load_json(
        "analysis/cross_cohort/reference_outputs/postlock_grid_sensitivity_report.json"
    )
    check(
        postlock["status"] == "PASS"
        and abs(postlock["primary_mean"] - 0.7663660668996223) < 1e-12
        and abs(postlock["augmented_mean"] - 0.7217772442386754) < 1e-12,
        "post-lock grid sensitivity numerical sentinels",
    )
    check(
        postlock["same_task_loss_objective"]
        and postlock["same_schedule"] == "200x5"
        and postlock["same_split_and_checkpoint_cohort"]
        and postlock["all_seed_deltas_negative"],
        "post-lock audit comparability and direction",
    )

    geometry = load_json(
        "analysis/local_geometry/reference_outputs/geometry_public_verification.json"
    )
    check(
        geometry["status"] == "PASS"
        and geometry["direction_rows"] == 1584
        and geometry["checkpoint_groups"] == 48
        and geometry["probe_radius_verified"]
        and geometry["public_processed_artifact_hash_chain"] == "PASS"
        and geometry["full_raw_checkpoint_processed_hash_chain"] == "PASS"
        and geometry["raw_to_direction_level_reconstruction"] == "PASS_1584_OF_1584"
        and geometry["checkpoint_digest_status"] == "PASS_48_OF_48"
        and geometry["historical_checkpoint_crosscheck"] == "PASS_6_OF_6"
        and geometry["radius_sensitivity_family_order_preserved"],
        "local geometry public verification",
    )
    provenance = load_json(
        "analysis/local_geometry/reference_outputs/geometry_public_provenance.json"
    )
    check(
        provenance["status"] == "PASS_FULL_RAW_CHECKPOINT_PROCESSED_HASH_CHAIN"
        and provenance["checkpoint_digests"]["available"] is True
        and provenance["checkpoint_digests"]["checkpoint_records"] == 48
        and provenance["checkpoint_digests"]["historical_imagenet_sinusoidal_matches"] == 6
        and all(item["sha256"] for item in provenance["raw_source_files"]),
        "geometry full raw/checkpoint/processed provenance chain",
    )
    radius_rows = list(
        csv.DictReader(
            (
                ROOT
                / "analysis/local_geometry/reference_outputs/geometry_radius_sensitivity.csv"
            ).open(encoding="utf-8")
        )
    )
    check(
        len(radius_rows) == 8
        and abs(
            float(
                next(
                    row["calibrated_secant_ratio_mean"]
                    for row in radius_rows
                    if row["dataset"] == "imagenet" and row["pe_type"] == "learned"
                )
            )
            - 21.320224388067093
        )
        < 1e-12,
        "geometry calibrated-radius sensitivity sentinel",
    )

    vits_manifest = load_json("checkpoints/vits_fp32_checkpoint_manifest.json")
    check(
        vits_manifest["architecture"]["dropout"] == 0.0
        and vits_manifest["architecture"]["training_recipe_dropout"] == 0.1
        and "inactive in eval mode"
        in vits_manifest["architecture"]["dropout_context"],
        "ViT-S checkpoint manifest distinguishes evaluation and training dropout",
    )
    check(
        "not redistributed" in vits_manifest["training_source_publication_status"]
        and "not a valid ViT-S FP32 training launcher"
        in vits_manifest["architecture_source_note"],
        "ViT-S manifest does not misrepresent the public ViT-B driver",
    )
    vits_amp_manifest = load_json("checkpoints/vits_amp/VITS_FP16_CHECKPOINT_MANIFEST.json")
    check(
        len(vits_amp_manifest["rows"]) == 18
        and {row["family"] for row in vits_amp_manifest["rows"]} == {"learned", "sinusoidal", "rope"}
        and not vits_amp_manifest["failures"]
        and vits_amp_manifest["complete"] is True,
        "ViT-S AMP 18-checkpoint digest manifest is public and excludes ALiBi",
    )
    binding = load_json("checkpoints/vitb_r32/VITB_CANONICAL_ATTACK_CHECKPOINT_BINDING.json")
    check(
        binding["status"] == "PASS_48_OF_48_PATHS_BOUND_TO_DIGEST_MANIFEST"
        and len(binding["records"]) == 48
        and all(row["path_match"] for row in binding["records"])
        and all(row["attack_json_embedded_checkpoint_sha256"] is None for row in binding["records"]),
        "48 canonical ViT-B attack paths bind to the post-hoc digest manifest",
    )
    check(
        "reconstructed post hoc" in manuscript
        and "historical attack JSONs record" in manuscript
        and "all 48 attack paths bind exactly" in manuscript,
        "manuscript discloses post-hoc ViT-B checkpoint digest binding",
    )

    locked_protocol = load_json("experiments/configs/vits_in100_locked_protocol.json")
    check(
        "pending" not in locked_protocol
        and "Complete" in locked_protocol["release_status"]["vits_full_n6"]
        and "Complete" in locked_protocol["release_status"]["manuscript_numerical_update"]
        and locked_protocol["vitb_reference_harmonization"]["historical_closure_record_redistributed"] is False,
        "locked ViT-S protocol has current release-completion state and truthful harmonization pointer",
    )

    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = load_json(".zenodo.json")
    check(
        cff.count("0000-0001-9034-6854") == 2
        and zenodo["creators"][0]["orcid"] == "0000-0001-9034-6854",
        "corresponding-author ORCID is present in citation and Zenodo metadata",
    )

    clean = load_json(
        "analysis/clean_accuracy/reference_outputs/clean_accuracy_report.json"
    )
    check(
        clean["status"] == "PASS"
        and clean["all_cells_seed_reproducible"]
        and clean["table_population"] == "locked robustness evaluation splits",
        "clean-accuracy table is seed-level reproducible",
    )

    cohorts = load_json("configs/cohorts.json")
    cohort_text = json.dumps(cohorts).lower()
    check(
        "all " + "six attempted runs diverged" not in cohort_text,
        "public cohort metadata avoids an unsupported per-seed ALiBi count",
    )
    governance = load_json("audit/trainability_boundary_governance.json")
    check(
        governance["status"] == "PASS"
        and governance["exact_per_seed_failure_count_publicly_supported"] is False,
        "ALiBi trainability-boundary governance",
    )

    with (
        ROOT / "audit/final_claim_to_evidence_matrix.csv"
    ).open(encoding="utf-8", newline="") as handle:
        gates = list(csv.DictReader(handle))
    check(
        len(gates) == 30
        and all(item["status"] == "PASS" for item in gates)
        and all(None not in item for item in gates),
        "claim-to-evidence matrix has 30 well-formed PASS gates",
    )

    unresolved_evidence = []
    for gate in gates:
        for token in (part.strip() for part in gate["evidence"].split(";")):
            if not token:
                unresolved_evidence.append((gate["gate_id"], token))
                continue
            if any(ch in token for ch in "*?["):
                matches = list(ROOT.glob(token))
                if not matches:
                    unresolved_evidence.append((gate["gate_id"], token))
            elif not (ROOT / token).exists():
                unresolved_evidence.append((gate["gate_id"], token))
    check(
        not unresolved_evidence,
        "all claim-to-evidence matrix evidence paths resolve (glob-aware)",
    )

    annotated_mode = load_json("configs/attack_eval_numerical_mode.json")
    locked_ref = annotated_mode["locked_artifact_reference"]
    locked_path = ROOT / locked_ref["path"]
    locked_hash = hashlib.sha256(locked_path.read_bytes()).hexdigest()
    final_audit = load_json("data/objective_specific/all_restart_audit/FINAL_AUDIT_REPORT_v1.json")
    check(
        locked_hash == locked_ref["sha256"]
        == final_audit["locks"]["ATTACK_EVAL_NUMERICAL_MODE_v1.json"],
        "annotated numerical-mode config resolves to the byte-exact locked artifact",
    )
    lock_paths = {
        "ATTACK_EVAL_NUMERICAL_MODE_v1.json": ROOT / "audit/historical/sinusoidal_audit/ATTACK_EVAL_NUMERICAL_MODE_v1.json",
        "SINUSOIDAL_PER_RESTART_AUDIT_INPUT_LOCK_v1.json": ROOT / "audit/historical/sinusoidal_audit/SINUSOIDAL_PER_RESTART_AUDIT_INPUT_LOCK_v1.json",
        "D033A_PROVENANCE_REPORT.json": ROOT / "audit/historical/sinusoidal_audit/D033A_PROVENANCE_REPORT.json",
    }
    check(
        all(
            hashlib.sha256(path.read_bytes()).hexdigest() == final_audit["locks"][name]
            for name, path in lock_paths.items()
        ),
        "all three all-restart audit lock artifacts resolve byte-exactly",
    )
    check(
        "annotated reviewer-facing copy" in locked_ref["note"]
        and "older-PyTorch fallback is not claimed to force MATH"
        in annotated_mode["claim_semantics"]["sdpa_backend_scope"],
        "numerical-mode annotation and conditional MATH-SDPA scope are explicit",
    )

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    check(
        "pytest" in requirements and "pytest" in lock and "pytest" in environment,
        "pytest is included in all documented default environments",
    )

    print("\nRepository verification completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
