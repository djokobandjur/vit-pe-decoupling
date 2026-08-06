from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_has_no_internal_d_codes():
    text = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
    assert re.search(r"\bD\d{3}[a-z]?\b", text) is None


def test_cross_architecture_reference_qa_passes():
    report = json.loads((ROOT / "analysis/cross_architecture/reference_outputs/D059_V2_QA_REPORT.json").read_text())
    assert report["status"] == "PASS"
    assert report["passed"] == report["n_checks"] == 31


def test_precision_bridge_reference_status():
    report = json.loads((ROOT / "analysis/precision_bridge/reference_outputs/D029_ANALYSIS_SUMMARY_v1.json").read_text())
    assert report["status"] == "PASS_ANALYSIS_COMPLETE"
    assert report["bridge_decision"]["overall_status"] == "INCONCLUSIVE"
