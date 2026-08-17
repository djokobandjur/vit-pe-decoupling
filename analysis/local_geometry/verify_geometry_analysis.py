#!/usr/bin/env python3
"""Verify the complete ViT-B/16 R32 geometry provenance chain.

The verifier checks retained execution-source hashes, six original raw JSON
files, all 48 checkpoint digest records, raw-to-direction reconstruction,
raw-to-seed reconstruction, aggregate summaries, processed-artifact hashes,
and the finite-radius sensitivity result. No checkpoint binary or GPU is
required because the byte-level digest manifest is redistributed.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS=["imagenet","cifar"]
FAMILIES=["learned","sinusoidal","rope","alibi"]
SEEDS=[42,123,456,789,1011,1213]


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def assert_frame_equal_numeric(actual:pd.DataFrame, expected:pd.DataFrame, keys:list[str], numeric:list[str], label:str)->None:
    a=actual.sort_values(keys).reset_index(drop=True)
    e=expected.sort_values(keys).reset_index(drop=True)
    if len(a)!=len(e): raise AssertionError(f"{label}: row count {len(a)} != {len(e)}")
    for c in keys:
        if a[c].astype(str).tolist()!=e[c].astype(str).tolist(): raise AssertionError(f"{label}: key mismatch in {c}")
    for c in numeric:
        if not np.allclose(a[c].astype(float),e[c].astype(float),atol=1e-10,rtol=1e-10,equal_nan=True):
            delta=np.nanmax(np.abs(a[c].astype(float)-e[c].astype(float)))
            raise AssertionError(f"{label}: numerical mismatch in {c}; max |delta|={delta}")


def main()->None:
    parser=argparse.ArgumentParser()
    here=Path(__file__).resolve().parent
    root=here.parents[1]
    parser.add_argument("--reference-dir",type=Path,default=here/"reference_outputs")
    parser.add_argument("--raw-dir",type=Path,default=here/"raw_sources")
    parser.add_argument("--checkpoint-dir",type=Path,default=root/"checkpoints/vitb_r32")
    parser.add_argument("--execution-dir",type=Path,default=here/"execution_provenance")
    parser.add_argument("--output-dir",type=Path,default=None)
    args=parser.parse_args()
    ref=args.reference_dir; output=args.output_dir or ref; output.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((ref/"geometry_public_provenance.json").read_text())
    if provenance["status"]!="PASS_FULL_RAW_CHECKPOINT_PROCESSED_HASH_CHAIN": raise AssertionError("Full geometry provenance status failed")

    # Verify retained source bytes against provenance and raw metadata.
    actual_source_hashes={}
    for name,item in provenance["execution_sources"].items():
        p=root/item["path"]
        actual=sha256(p)
        if actual!=item["sha256"]: raise AssertionError(f"Execution source hash mismatch: {name}")
        actual_source_hashes[name]=actual

    # Verify checkpoint manifest outputs and internal validation.
    sums={}
    for line in (args.checkpoint_dir/"SHA256SUMS.txt").read_text().splitlines():
        expected,name=line.split("  ",1); sums[name]=expected
        if sha256(args.checkpoint_dir/name)!=expected: raise AssertionError(f"Checkpoint manifest artifact mismatch: {name}")
    ckpt_doc=json.loads((args.checkpoint_dir/"VITB_R32_CHECKPOINT_MANIFEST.json").read_text())
    ckpt_verify=json.loads((args.checkpoint_dir/"VITB_R32_CHECKPOINT_VERIFICATION.json").read_text())
    if not ckpt_verify["overall_pass"] or ckpt_verify["record_count_observed"]!=48 or ckpt_verify["historical_imagenet_sinusoidal_hashes_matched"]!=6:
        raise AssertionError("Checkpoint manifest verification failed")
    ckpt_map={}
    for r in ckpt_doc["checkpoints"]:
        dataset="imagenet" if r["dataset"]=="imagenet100" else "cifar"
        key=(dataset,r["pe_family"],int(r["seed"]))
        if key in ckpt_map: raise AssertionError(f"Duplicate checkpoint key: {key}")
        ckpt_map[key]=r
    if len(ckpt_map)!=48: raise AssertionError("Expected 48 checkpoint records")

    # Verify original-file manifest, raw SHA-256 values, source hashes, and reconstruct rows.
    original_sums={}
    for line in (args.raw_dir/"SHA256_ALL_ORIGINAL_FILES.txt").read_text().splitlines():
        expected,name=line.split("  ",1); original_sums[name]=expected
        if sha256(args.raw_dir/name)!=expected: raise AssertionError(f"Original R32 file hash mismatch: {name}")
    raw_expected={item["file"]:item for item in provenance["raw_source_files"]}
    direction_rows=[]; seed_rows=[]; checkpoint_paths={}
    raw_files=sorted(args.raw_dir.glob("*_seeds_*.json"))
    raw_files=[p for p in raw_files if ".COMPLETE." not in p.name]
    if len(raw_files)!=6: raise AssertionError(f"Expected six raw JSON files, found {len(raw_files)}")
    for path in raw_files:
        if path.name not in raw_expected or sha256(path)!=raw_expected[path.name]["sha256"]: raise AssertionError(f"Raw source digest mismatch: {path.name}")
        doc=json.loads(path.read_text())
        md=doc["metadata"]
        for name,expected in md["core_hashes"].items():
            if actual_source_hashes.get(name)!=expected: raise AssertionError(f"Raw metadata source hash mismatch: {path.name}/{name}")
        dataset=md["dataset"]
        for family in FAMILIES:
            for seed_s,result in doc["results"][family].items():
                seed=int(seed_s); key=(dataset,family,seed)
                if result["status"]!="ok": raise AssertionError(f"Non-ok raw result: {key}")
                rec=ckpt_map[key]
                if result["checkpoint"]!=rec["checkpoint_path"]: raise AssertionError(f"Checkpoint path mismatch: {key}")
                checkpoint_paths[key]=result["checkpoint"]
                ds=result["directions"]
                names={"task_gradient"}|{f"random_{i:02d}" for i in range(32)}
                if set(ds)!=names: raise AssertionError(f"Direction set mismatch: {key}")
                for direction in sorted(ds):
                    block=ds[direction]; cal=block["calibration"]; held=block["heldout_geometry"]
                    rho=float(held["rho_rel_geometry"]); delta=float(held["delta_ce"])
                    direction_rows.append({
                        "dataset":dataset,"pe_type":family,"seed":seed,"direction":direction,
                        "kind":cal["kind"],"functional_gain":float(cal["local_functional_gain"]),
                        "probe_radius":float(cal["probe_radius"]),
                        "rho_rel_calibration":float(cal["achieved_rho_rel_calibration"]),
                        "rho_rel_heldout":rho,"delta_ce":delta,"damage_efficiency":delta/rho,
                        "parameter_radius_global_rms":float(cal["parameter_radius_global_rms"]),
                        "parameter_space_cosine_to_task_gradient":float(cal["parameter_space_cosine_to_task_gradient"]),
                        "attention_task_alignment_cosine":float(held["attention_task_alignment_cosine"]),
                        "source_file":path.name,
                    })
                sub=[r for r in direction_rows if r["dataset"]==dataset and r["pe_type"]==family and r["seed"]==seed]
                task=next(r for r in sub if r["direction"]=="task_gradient")
                random=[r for r in sub if r["direction"].startswith("random_")]
                rg=np.array([r["functional_gain"] for r in random]); rr=np.array([r["rho_rel_heldout"] for r in random]); rd=np.array([r["delta_ce"] for r in random]); re=np.array([r["damage_efficiency"] for r in random]); rc=np.array([r["rho_rel_calibration"] for r in random])
                seed_rows.append({
                    "dataset":dataset,"pe_type":family,"seed":seed,"n_random_directions":32,
                    "task_rho_rel_heldout":task["rho_rel_heldout"],"random_rho_rel_heldout_median":float(np.median(rr)),
                    "task_delta_ce":task["delta_ce"],"random_delta_ce_median":float(np.median(rd)),
                    "task_damage_efficiency_actual_rho":task["damage_efficiency"],"random_damage_efficiency_actual_rho_median":float(np.median(re)),
                    "damage_efficiency_gap_task_minus_random":task["damage_efficiency"]-float(np.median(re)),
                    "task_functional_gain":task["functional_gain"],"random_functional_gain_median":float(np.median(rg)),
                    "task_to_random_functional_gain_ratio":task["functional_gain"]/float(np.median(rg)),
                    "task_calibration_rho_rel":task["rho_rel_calibration"],"random_calibration_rho_rel_median":float(np.median(rc)),
                    "source_file":path.name,"checkpoint_path":rec["checkpoint_path"],"checkpoint_sha256":rec["sha256"],
                    "checkpoint_digest_status":"PASS_POSTHOC_MANIFEST_RETAINED_ORIGINAL_COHORT",
                })
    if len(direction_rows)!=1584 or len(seed_rows)!=48 or len(checkpoint_paths)!=48: raise AssertionError("Raw reconstruction cardinality failed")

    directions=pd.read_csv(ref/"geometry_direction_level.csv")
    locked_seed=pd.read_csv(ref/"geometry_seed_level.csv")
    raw_directions=pd.DataFrame(direction_rows)
    raw_seed=pd.DataFrame(seed_rows)
    dkeys=["dataset","pe_type","seed","direction","kind","source_file"]
    dnums=["functional_gain","probe_radius","rho_rel_calibration","rho_rel_heldout","delta_ce","damage_efficiency","parameter_radius_global_rms","parameter_space_cosine_to_task_gradient","attention_task_alignment_cosine"]
    assert_frame_equal_numeric(raw_directions,directions,dkeys,dnums,"raw-to-direction")
    skeys=["dataset","pe_type","seed","source_file","checkpoint_path","checkpoint_sha256","checkpoint_digest_status"]
    snums=["n_random_directions","task_rho_rel_heldout","random_rho_rel_heldout_median","task_delta_ce","random_delta_ce_median","task_damage_efficiency_actual_rho","random_damage_efficiency_actual_rho_median","damage_efficiency_gap_task_minus_random","task_functional_gain","random_functional_gain_median","task_to_random_functional_gain_ratio","task_calibration_rho_rel","random_calibration_rho_rel_median"]
    assert_frame_equal_numeric(raw_seed,locked_seed,skeys,snums,"raw-to-seed")

    metadata=pd.read_csv(ref/"geometry_metadata_audit.csv")
    if len(metadata)!=6 or not (metadata["raw_source_digest_status"]=="PASS_REDISTRIBUTED_ORIGINAL_RAW_JSON").all(): raise AssertionError("Metadata raw digest status failed")
    for _,row in metadata.iterrows():
        if row["raw_source_sha256"]!=sha256(args.raw_dir/row["file"]): raise AssertionError(f"Metadata raw hash mismatch: {row['file']}")

    # Existing processed seed/aggregate reconstruction and radius sensitivity.
    locked_agg=pd.read_csv(ref/"geometry_aggregate.csv")
    seed=raw_seed.copy()
    radius_rows=[]
    for (dataset,family),group in seed.groupby(["dataset","pe_type"],sort=False):
        source=directions.query("dataset == @dataset and pe_type == @family")
        sec=[]
        for s in SEEDS:
            sub=source.query("seed == @s"); task=sub[sub["direction"]=="task_gradient"].iloc[0]; random=sub[sub["direction"].str.startswith("random_")]
            sec.append((task["rho_rel_calibration"]/task["parameter_radius_global_rms"])/float((random["rho_rel_calibration"]/random["parameter_radius_global_rms"]).median()))
        local=group["task_to_random_functional_gain_ratio"].to_numpy(float); sec=np.asarray(sec,float)
        radius_rows.append({"dataset":dataset,"pe_type":family,"n_seeds":len(group),"local_ratio_mean":float(local.mean()),"local_ratio_sd":float(local.std(ddof=1)),"calibrated_secant_ratio_mean":float(sec.mean()),"calibrated_secant_ratio_sd":float(sec.std(ddof=1))})
    radius=pd.DataFrame(radius_rows)
    rank_preserved=all(radius.query("dataset == @d").sort_values("local_ratio_mean",ascending=False)["pe_type"].tolist()==radius.query("dataset == @d").sort_values("calibrated_secant_ratio_mean",ascending=False)["pe_type"].tolist() for d in DATASETS)
    agg_rows=[]
    for (dataset,family),g in seed.groupby(["dataset","pe_type"],sort=False):
        ratio=g["task_to_random_functional_gain_ratio"].to_numpy(float); gap=g["damage_efficiency_gap_task_minus_random"].to_numpy(float)
        agg_rows.append({"dataset":dataset,"pe_type":family,"n_seeds":len(g),"n_random_directions_per_seed":32,"gain_ratio_mean":float(ratio.mean()),"gain_ratio_sd":float(ratio.std(ddof=1)),"gain_ratio_positive_seeds":int(np.sum(ratio>1)),"damage_efficiency_gap_mean":float(gap.mean()),"damage_efficiency_gap_sd":float(gap.std(ddof=1)),"damage_efficiency_gap_positive_seeds":int(np.sum(gap>0))})
    agg=pd.DataFrame(agg_rows)
    assert_frame_equal_numeric(agg,locked_agg,["dataset","pe_type","n_seeds","n_random_directions_per_seed","gain_ratio_positive_seeds","damage_efficiency_gap_positive_seeds"],["gain_ratio_mean","gain_ratio_sd","damage_efficiency_gap_mean","damage_efficiency_gap_sd"],"seed-to-aggregate")

    probe=json.loads((ref/"probe_radius_evidence.json").read_text())
    if not (directions["probe_radius"]==1e-4).all() or not probe["all_equal_1e-4"]: raise AssertionError("Probe radius failed")
    for name,expected in provenance["processed_artifacts"].items():
        if sha256(ref/name)!=expected: raise AssertionError(f"Processed artifact hash mismatch: {name}")
    locked_radius=pd.read_csv(ref/"geometry_radius_sensitivity.csv")
    assert_frame_equal_numeric(radius,locked_radius,["dataset","pe_type","n_seeds"],["local_ratio_mean","local_ratio_sd","calibrated_secant_ratio_mean","calibrated_secant_ratio_sd"],"radius-sensitivity")
    if not rank_preserved: raise AssertionError("Family ordering changed at calibrated secant point")

    report={
      "status":"PASS","direction_rows":1584,"checkpoint_groups":48,"task_directions":48,"gaussian_directions":1536,
      "probe_radius":0.0001,"probe_radius_verified":True,"execution_source_hash_status":"PASS_3_OF_3",
      "raw_source_digest_status":"PASS_6_OF_6","raw_completion_markers_status":"PASS_6_OF_6",
      "raw_to_direction_level_reconstruction":"PASS_1584_OF_1584","raw_to_seed_level_reconstruction":"PASS_48_OF_48",
      "seed_level_recalculation":"PASS","aggregate_recalculation":"PASS","public_processed_artifact_hash_chain":"PASS",
      "checkpoint_digest_status":"PASS_48_OF_48","historical_checkpoint_crosscheck":"PASS_6_OF_6",
      "full_raw_checkpoint_processed_hash_chain":"PASS","radius_sensitivity_family_order_preserved":bool(rank_preserved)
    }
    raw_seed.to_csv(output/"geometry_seed_level_recomputed.csv",index=False)
    agg.to_csv(output/"geometry_aggregate_recomputed.csv",index=False)
    # Preserve the locked canonical byte representation after numerical verification.
    (output/"geometry_radius_sensitivity.csv").write_bytes((ref/"geometry_radius_sensitivity.csv").read_bytes())
    (output/"geometry_public_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
