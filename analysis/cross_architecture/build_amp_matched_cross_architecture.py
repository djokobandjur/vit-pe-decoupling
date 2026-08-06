#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, t as student_t

SEEDS = [42, 123, 456, 789, 1011, 1213]
FAMILIES = ["learned", "sinusoidal", "rope"]
LABELS = {"learned": "Learned", "sinusoidal": "Sinusoidal", "rope": "RoPE"}
ARCHS = ["vitb_amp", "vits_amp"]
TAU_ABS = 5e-7
TAU_REL = 5e-6
SUPPORT_MULTIPLIERS = [1.0, 0.9, 0.75]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(safe_json(obj), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def safe_extract(zip_path: Path, out: Path) -> None:
    out_res = out.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            dest = (out / info.filename).resolve()
            if out_res != dest and out_res not in dest.parents:
                raise ValueError(f"unsafe archive member: {info.filename}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"symlink archive member: {info.filename}")
        zf.extractall(out)


def near(a: float, b: float) -> bool:
    return abs(a - b) <= max(TAU_ABS, TAU_REL * max(abs(a), abs(b), 1e-12))


def exact_sign_flip(vals: np.ndarray) -> float:
    vals = np.asarray(vals, float)
    obs = abs(float(vals.mean()))
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(vals))))
    stats = np.abs((signs * vals[None, :]).mean(axis=1))
    return float(np.mean(stats >= obs - 1e-15))


def contrast_stats(vals: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(vals, float)
    n = len(vals)
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    if n > 1 and se > 0:
        q = float(student_t.ppf(0.975, n - 1))
        ci = (mean - q * se, mean + q * se)
    else:
        ci = (mean, mean)
    return {
        "n": n,
        "mean_delta_vits_minus_vitb": mean,
        "sd_delta": sd,
        "ci95_t_low": float(ci[0]),
        "ci95_t_high": float(ci[1]),
        "negative_deltas": int((vals < 0).sum()),
        "positive_deltas": int((vals > 0).sum()),
        "zero_deltas": int((vals == 0).sum()),
        "exact_sign_flip_p_two_sided": exact_sign_flip(vals),
        "min_delta": float(vals.min()),
        "max_delta": float(vals.max()),
    }


def endpoint(points: pd.DataFrame, q: float) -> dict[str, Any]:
    pts = points.sort_values("budget")
    eps = pts["budget"].to_numpy(float)
    acc = pts["accuracy"].to_numpy(float)
    rho = pts["rho"].to_numpy(float)
    if len(eps) == 0 or eps[0] > 0:
        eps = np.concatenate(([0.0], eps))
        acc = np.concatenate(([1.0], acc))
        rho = np.concatenate(([0.0], rho))
    env = np.minimum.accumulate(acc)
    hit = np.where(env <= q)[0]
    if len(hit) == 0:
        return {"status": "NOT_BRACKETED", "q": q, "max_budget": float(eps[-1]), "min_envelope_accuracy": float(env.min())}
    j = int(hit[0])
    if j == 0:
        return {"status": "AT_OR_BELOW_CLEAN", "q": q, "epsilon": float(eps[0]), "rho": float(rho[0])}
    e0, e1 = float(eps[j-1]), float(eps[j])
    a0, a1 = float(env[j-1]), float(env[j])
    r0, r1 = float(rho[j-1]), float(rho[j])
    w = 1.0 if abs(a1-a0) < 1e-15 else (q-a0)/(a1-a0)
    return {
        "status": "IDENTIFIED", "q": q,
        "epsilon": e0 + w*(e1-e0), "rho": r0 + w*(r1-r0),
        "epsilon_left": e0, "epsilon_right": e1,
        "rho_left": r0, "rho_right": r1,
        "accuracy_left": a0, "accuracy_right": a1,
        "epsilon_bracket_width": e1-e0,
        "rho_bracket_abs_width": abs(r1-r0),
    }


def dedupe(points: pd.DataFrame, mode: str) -> pd.DataFrame:
    pts = points[np.isfinite(points.rho) & np.isfinite(points.accuracy) & (points.rho > 0)].copy()
    pts = pts.sort_values(["rho", "budget", "source_index"]).reset_index(drop=True)
    groups: list[list[pd.Series]] = []
    for _, row in pts.iterrows():
        if not groups or not near(float(groups[-1][0]["rho"]), float(row["rho"])):
            groups.append([row])
        else:
            groups[-1].append(row)
    selected: list[dict[str, Any]] = []
    for g in groups:
        if mode == "attack":
            def key(r: pd.Series):
                loss = float(r.get("attack_loss", np.nan))
                return (float(r["accuracy"]), -loss if np.isfinite(loss) else float("inf"), float(r["budget"]), int(r["source_index"]))
            selected.append(dict(min(g, key=key)))
        else:
            row = dict(g[0])
            row["rho"] = float(np.mean([float(x.rho) for x in g]))
            row["accuracy"] = float(np.mean([float(x.accuracy) for x in g]))
            selected.append(row)
    return pd.DataFrame(selected).sort_values("rho")


def curve(points: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    d = dedupe(points, mode)
    x = np.concatenate(([0.0], d.rho.to_numpy(float)))
    y = np.concatenate(([1.0], d.accuracy.to_numpy(float)))
    if mode == "attack":
        y = np.minimum.accumulate(y)
    if not np.all(np.diff(x) > 0):
        raise ValueError("non-increasing rho after dedupe")
    return x, y


def nauc(x: np.ndarray, y: np.ndarray, end: float) -> float:
    if x[-1] + 1e-12 < end:
        raise ValueError(f"support {x[-1]} below requested {end}")
    keep = x < end
    xx = np.concatenate((x[keep], [end]))
    yy = np.concatenate((y[keep], [np.interp(end, x, y)]))
    return float(np.trapezoid(yy, xx) / end)


def load_vitb(adv_path: Path, noise_path: Path, clean_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    adv = pd.read_csv(adv_path)
    noi = pd.read_csv(noise_path)
    cln = pd.read_csv(clean_path)
    adv = adv[(adv.dataset == "imagenet") & adv.family.isin(FAMILIES) & adv.seed.isin(SEEDS)].copy()
    noi = noi[(noi.dataset == "imagenet") & noi.family.isin(FAMILIES) & noi.seed.isin(SEEDS)].copy()
    cln = cln[(cln.dataset == "imagenet") & cln.family.isin(FAMILIES) & cln.seed.isin(SEEDS)].copy()
    adv = adv.rename(columns={"accuracy":"accuracy", "rho":"rho"})
    adv["architecture"] = "vitb_amp"
    adv["source_index"] = adv.groupby(["family","seed"]).cumcount()
    noi["architecture"] = "vitb_amp"
    noi["source_index"] = noi.groupby(["family","seed"]).cumcount()
    cln = cln.rename(columns={"clean_acc_eval":"clean_accuracy_percent"})
    cln["architecture"] = "vitb_amp"
    report = {
        "attack_rows": len(adv), "noise_rows": len(noi), "clean_rows": len(cln),
        "families": sorted(adv.family.unique()), "seeds": sorted(int(x) for x in adv.seed.unique()),
        "datasets": sorted(set(adv.dataset.unique()) | set(noi.dataset.unique()) | set(cln.dataset.unique())),
        "attack_source": str(adv_path), "noise_source": str(noise_path), "clean_source": str(clean_path),
    }
    return adv, noi, cln, report


def load_vits_amp(zip_path: Path, temp_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    safe_extract(zip_path, temp_root)
    root = temp_root / "vits_in100_fp16_robustness_v1"
    session_root = root / "final_n6"
    sessions = sorted(session_root.glob("session_*_seed_*"))
    if len(sessions) != 6:
        raise ValueError(f"expected 6 AMP sessions, found {len(sessions)}")
    attacks: list[dict[str, Any]] = []
    noises_raw: list[dict[str, Any]] = []
    cleans: list[dict[str, Any]] = []
    split_hashes, order_hashes, split_file_hashes, protocol_ids = set(), set(), set(), set()
    config_by_family = {f:set() for f in FAMILIES}
    completion = []
    for sess in sessions:
        protocol = json.loads((sess / "SESSION_PROTOCOL.json").read_text())
        seed = int(protocol["seed"])
        protocol_ids.add(protocol["protocol_id"])
        completion.append(json.loads((sess / "SESSION_COMPLETE.json").read_text()))
        noise = json.loads((sess / "noise_all_families.json").read_text())
        meta = noise["metadata"]
        split_hashes.add(meta["split"]["split_sha256"])
        order_hashes.add(meta["split"]["sample_order_sha256"])
        split_file_hashes.add(meta["split_file_sha256"])
        for family in FAMILIES:
            sr = noise["results"][family][str(seed)]
            cleans.append({"architecture":"vits_amp", "family":family, "seed":seed,
                           "clean_accuracy_percent":float(sr["clean_acc_eval"]),
                           "checkpoint_sha256":sr["checkpoint_sha256"]})
            for draw in sr["noise"]["draws"]:
                for idx, p in enumerate(sorted(draw["points"].values(), key=lambda z: float(z["budget"]))):
                    noises_raw.append({"architecture":"vits_amp", "family":family, "seed":seed,
                                       "draw_seed":int(draw["seed"]), "budget":float(p["budget"]),
                                       "rho":float(p["rho"]["rho_rel"]), "accuracy":float(p["normalized_accuracy"]),
                                       "source_index":idx})
            a = json.loads((sess / f"attacks_{family}.json").read_text())
            am = a["metadata"]
            split_hashes.add(am["split"]["split_sha256"])
            order_hashes.add(am["split"]["sample_order_sha256"])
            split_file_hashes.add(am["split_file_sha256"])
            cfg = am["config"]
            config_by_family[family].add((int(cfg["pgd_steps"]), int(cfg["pgd_restarts"]), float(cfg["pgd_alpha_ratio"]), cfg["parameter_norm"]))
            sr_a = a["results"][family][str(seed)]
            for idx, p in enumerate(sorted(sr_a["attacks"]["pgd_pe"].values(), key=lambda z: float(z["budget"]))):
                records = p.get("restart_records", [])
                if float(p["budget"]) > 0:
                    if len(records) != 5:
                        raise ValueError(f"{family}/{seed}/{p['budget']}: expected 5 restart records")
                    mx = max(float(r["attack_loss"]) for r in records)
                    if not math.isclose(float(p["selected_attack_loss"]), mx, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("selected attack loss mismatch")
                attacks.append({"architecture":"vits_amp", "family":family, "seed":seed,
                                "budget":float(p["budget"]), "rho":float(p["rho"]["rho_rel"]),
                                "accuracy":float(p["normalized_accuracy"]),
                                "attack_loss":float(p["selected_attack_loss"]) if p["selected_attack_loss"] is not None else np.nan,
                                "source_index":idx})
    raw = pd.DataFrame(noises_raw)
    noi = raw.groupby(["architecture","family","seed","budget"], as_index=False).agg(
        rho=("rho","mean"), accuracy=("accuracy","mean"),
        rho_sd_across_draws=("rho", lambda x: float(np.std(x,ddof=1))),
        accuracy_sd_across_draws=("accuracy", lambda x: float(np.std(x,ddof=1))),
        n_draws=("draw_seed","nunique"))
    noi["source_index"] = noi.groupby(["family","seed"]).cumcount()
    integrity = json.loads((root / "final_audit" / "FINAL_N6_INTEGRITY_AUDIT.json").read_text())
    convergence = json.loads((root / "convergence_audit" / "VITS_FP16_CONVERGENCE_AUDIT.json").read_text())
    locked = json.loads((root / "protocol" / "VITS_FP16_LOCKED_PROTOCOL.json").read_text())
    report = {
        "sessions": len(sessions), "completion_markers": len(completion),
        "split_sha256": sorted(split_hashes), "sample_order_sha256": sorted(order_hashes),
        "split_file_sha256": sorted(split_file_hashes), "protocol_ids": sorted(protocol_ids),
        "attack_configs": {f:[list(v) for v in sorted(config_by_family[f])] for f in FAMILIES},
        "final_integrity_audit": integrity, "convergence_audit": convergence,
        "locked_protocol": locked, "zip_sha256": sha256(zip_path),
    }
    return pd.DataFrame(attacks), noi, pd.DataFrame(cleans), report


def exact_friedman(matrix: np.ndarray) -> dict[str, Any]:
    # rows=seeds/blocks, cols=families/treatments; greater ranks for greater values.
    n, k = matrix.shape
    ranks = np.vstack([rankdata(row, method="average") for row in matrix])
    sums = ranks.sum(axis=0)
    q_obs = float(12/(n*k*(k+1))*np.sum(sums*sums) - 3*n*(k+1))
    perms = list(itertools.permutations(range(1,k+1)))
    count = 0; total = 0
    for choices in itertools.product(range(len(perms)), repeat=n):
        rs = np.zeros(k)
        for c in choices:
            rs += np.asarray(perms[c], float)
        q = float(12/(n*k*(k+1))*np.sum(rs*rs) - 3*n*(k+1))
        count += int(q >= q_obs - 1e-12); total += 1
    asym = friedmanchisquare(*[matrix[:,j] for j in range(k)])
    return {"n_blocks":n, "n_families":k, "Q":q_obs, "exact_p":count/total,
            "asymptotic_p":float(asym.pvalue), "rank_sums":{FAMILIES[j]:float(sums[j]) for j in range(k)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitb-adversarial-csv", type=Path, required=True)
    ap.add_argument("--vitb-noise-csv", type=Path, required=True)
    ap.add_argument("--vitb-clean-csv", type=Path, required=True)
    ap.add_argument("--vitb-pipeline-report", type=Path)
    ap.add_argument("--vits-amp-zip", type=Path, required=True)
    ap.add_argument("--d033-provenance", type=Path, required=True)
    ap.add_argument("--attack-eval-mode", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="d059_amp_") as tmp:
        vitb_a, vitb_n, vitb_c, vitb_report = load_vitb(args.vitb_adversarial_csv, args.vitb_noise_csv, args.vitb_clean_csv)
        vits_a, vits_n, vits_c, vits_report = load_vits_amp(args.vits_amp_zip, Path(tmp))

    attacks = pd.concat([vitb_a, vits_a], ignore_index=True)
    noises = pd.concat([vitb_n, vits_n], ignore_index=True)
    cleans = pd.concat([vitb_c, vits_c], ignore_index=True)
    attacks = attacks.sort_values(["architecture","family","seed","budget"])
    noises = noises.sort_values(["architecture","family","seed","budget"])
    cleans = cleans.sort_values(["architecture","family","seed"])

    # Hard coverage checks.
    expected = {(a,f,s) for a in ARCHS for f in FAMILIES for s in SEEDS}
    actual_a = set(map(tuple, attacks[["architecture","family","seed"]].drop_duplicates().to_records(index=False)))
    actual_n = set(map(tuple, noises[["architecture","family","seed"]].drop_duplicates().to_records(index=False)))
    actual_c = set(map(tuple, cleans[["architecture","family","seed"]].drop_duplicates().to_records(index=False)))
    checks = {
        "attack_36_curves": actual_a == expected,
        "noise_36_curves": actual_n == expected,
        "clean_36_rows": actual_c == expected and len(cleans)==36,
        "vits_six_sessions": vits_report["sessions"] == 6,
        "vits_final_audit_present": bool(vits_report["final_integrity_audit"]),
        "d033_present": args.d033_provenance.exists(),
        "numerical_mode_present": args.attack_eval_mode.exists(),
    }
    if not all(checks.values()):
        raise ValueError(f"coverage/integrity checks failed: {checks}")

    # Threshold endpoints.
    end_rows=[]
    for (arch,fam,seed), g in attacks.groupby(["architecture","family","seed"]):
        row={"architecture":arch,"family":fam,"family_label":LABELS[fam],"seed":int(seed),
             "clean_accuracy_percent":float(cleans.query("architecture==@arch and family==@fam and seed==@seed").iloc[0].clean_accuracy_percent),
             "max_rho":float(g.rho.max()),"min_accuracy":float(g.accuracy.min())}
        for q, tag in [(0.8,"80"),(0.5,"50"),(0.2,"20")]:
            ep=endpoint(g,q); row[f"q{tag}_status"]=ep["status"]
            for k in ["epsilon","rho","epsilon_left","epsilon_right","rho_left","rho_right","epsilon_bracket_width","rho_bracket_abs_width"]:
                row[f"q{tag}_{k}"]=ep.get(k)
        row["cliff_width_rho20_minus_rho80"]=(row.get("q20_rho")-row.get("q80_rho")) if row.get("q20_rho") is not None and row.get("q80_rho") is not None else None
        end_rows.append(row)
    endpoints=pd.DataFrame(end_rows).sort_values(["family","seed","architecture"])
    status=endpoints.groupby(["architecture","family"])[["q80_status","q50_status","q20_status"]].agg(lambda x: ";".join(f"{k}:{v}" for k,v in x.value_counts().sort_index().items())).reset_index()

    # Curves/support.
    curves={}; support_rows=[]
    for arch in ARCHS:
        for fam in FAMILIES:
            for seed in SEEDS:
                for mode,df in [("attack",attacks),("noise",noises)]:
                    g=df.query("architecture==@arch and family==@fam and seed==@seed")
                    x,y=curve(g,mode); curves[(arch,fam,seed,mode)]=(x,y)
                    support_rows.append({"architecture":arch,"family":fam,"seed":seed,"curve":mode,"max_rho":float(x[-1]),"n_points":len(x)})
    support=pd.DataFrame(support_rows)
    cross_support=float(support.max_rho.min())
    limiter=support.loc[support.max_rho.idxmin()].to_dict()
    arch_support={a:float(support.query("architecture==@a").max_rho.min()) for a in ARCHS}
    family_support={f:float(support.query("family==@f").max_rho.min()) for f in FAMILIES}
    support_scope_rows=[{"scope":"cross","key":"all","rho_endpoint":cross_support,**{f"limiter_{k}":v for k,v in limiter.items()}}]
    support_scope_rows += [{"scope":"architecture","key":a,"rho_endpoint":v} for a,v in arch_support.items()]
    support_scope_rows += [{"scope":"family","key":f,"rho_endpoint":v} for f,v in family_support.items()]
    support_scopes=pd.DataFrame(support_scope_rows)

    # nAUC on cross, architecture, family scopes and sensitivities.
    nauc_rows=[]
    scope_defs=[]
    for mult in SUPPORT_MULTIPLIERS:
        scope_defs.append(("cross","all",mult,cross_support*mult))
    for a,v in arch_support.items(): scope_defs.append(("architecture",a,1.0,v))
    for f,v in family_support.items(): scope_defs.append(("family",f,1.0,v))
    for scope,key,mult,rho_end in scope_defs:
        for arch in ARCHS:
            if scope=="architecture" and arch!=key: continue
            for fam in FAMILIES:
                if scope=="family" and fam!=key: continue
                for seed in SEEDS:
                    xa,ya=curves[(arch,fam,seed,"attack")]; xn,yn=curves[(arch,fam,seed,"noise")]
                    an=nauc(xa,ya,rho_end); nn=nauc(xn,yn,rho_end)
                    nauc_rows.append({"scope":scope,"scope_key":key,"support_multiplier":mult,"rho_endpoint":rho_end,
                                      "architecture":arch,"family":fam,"family_label":LABELS[fam],"seed":seed,
                                      "attack_nauc":an,"noise_nauc":nn,"noise_minus_attack_gap":nn-an})
    naucs=pd.DataFrame(nauc_rows)

    # Cross-support aggregate and seed-aligned contrasts.
    primary=naucs.query("scope=='cross' and support_multiplier==1.0")
    agg=primary.groupby(["architecture","family"],as_index=False).agg(
        attack_nauc_mean=("attack_nauc","mean"),attack_nauc_sd=("attack_nauc","std"),
        noise_nauc_mean=("noise_nauc","mean"),noise_nauc_sd=("noise_nauc","std"),
        gap_mean=("noise_minus_attack_gap","mean"),gap_sd=("noise_minus_attack_gap","std"))
    contrast_seed=[]; contrast_agg=[]
    for fam in FAMILIES:
        b=primary.query("architecture=='vitb_amp' and family==@fam").sort_values("seed")
        s=primary.query("architecture=='vits_amp' and family==@fam").sort_values("seed")
        for metric in ["attack_nauc","noise_nauc","noise_minus_attack_gap"]:
            vals=s[metric].to_numpy()-b[metric].to_numpy()
            for seed,v,bv,sv in zip(SEEDS,vals,b[metric],s[metric]):
                contrast_seed.append({"family":fam,"seed":seed,"metric":metric,"vitb_value":float(bv),"vits_value":float(sv),"delta_vits_minus_vitb":float(v)})
            contrast_agg.append({"family":fam,"family_label":LABELS[fam],"metric":metric,
                                 "vitb_mean":float(b[metric].mean()),"vitb_sd":float(b[metric].std(ddof=1)),
                                 "vits_mean":float(s[metric].mean()),"vits_sd":float(s[metric].std(ddof=1)),
                                 **contrast_stats(vals)})
    contrast_seed_df=pd.DataFrame(contrast_seed); contrast_agg_df=pd.DataFrame(contrast_agg)

    # Rank agreement on primary and sensitivities.
    ranks=[]
    for mult in SUPPORT_MULTIPLIERS:
        d=naucs.query("scope=='cross' and support_multiplier==@mult")
        for metric in ["attack_nauc","noise_nauc","noise_minus_attack_gap"]:
            orders={}
            for arch in ARCHS:
                means=d.query("architecture==@arch").groupby("family")[metric].mean()
                asc=(metric=="noise_minus_attack_gap")
                means=means.sort_values(ascending=asc)
                orders[arch]=" > ".join(means.index)
            ranks.append({"support_multiplier":mult,"rho_endpoint":cross_support*mult,"metric":metric,
                          "vitb_order":orders["vitb_amp"],"vits_order":orders["vits_amp"],
                          "exact_order_preserved":orders["vitb_amp"]==orders["vits_amp"]})
    ranks_df=pd.DataFrame(ranks)

    # Architecture x PE interaction using seed-level architecture effects.
    attack_delta=contrast_seed_df.query("metric=='attack_nauc'").pivot(index="seed",columns="family",values="delta_vits_minus_vitb").reindex(index=SEEDS,columns=FAMILIES)
    interaction=exact_friedman(attack_delta.to_numpy())
    did=[]
    for f1,f2 in itertools.combinations(FAMILIES,2):
        vals=attack_delta[f1].to_numpy()-attack_delta[f2].to_numpy()
        did.append({"contrast":f"{f1}_minus_{f2}",**contrast_stats(vals)})
    did_df=pd.DataFrame(did)

    # Threshold contrasts, only common identified seeds; explicitly censor Learned q50.
    threshold_seed=[]; threshold_stats=[]
    for fam in FAMILIES:
        for tag in ["80","50","20"]:
            for metric in ["epsilon","rho"]:
                col=f"q{tag}_{metric}"
                b=endpoints.query("architecture=='vitb_amp' and family==@fam").set_index("seed")
                s=endpoints.query("architecture=='vits_amp' and family==@fam").set_index("seed")
                common=[seed for seed in SEEDS if b.loc[seed,f"q{tag}_status"]=="IDENTIFIED" and s.loc[seed,f"q{tag}_status"]=="IDENTIFIED"]
                vals=[]
                for seed in common:
                    delta=float(s.loc[seed,col]-b.loc[seed,col]); vals.append(delta)
                    threshold_seed.append({"family":fam,"threshold":f"q{tag}","metric":metric,"seed":seed,
                                           "vitb_value":float(b.loc[seed,col]),"vits_value":float(s.loc[seed,col]),"delta_vits_minus_vitb":delta})
                row={"family":fam,"family_label":LABELS[fam],"threshold":f"q{tag}","metric":metric,
                     "identified_common_n":len(common),"identified_common_seeds":";".join(map(str,common)),
                     "full_n6_estimable":len(common)==6}
                if vals: row.update(contrast_stats(np.asarray(vals)))
                threshold_stats.append(row)
    threshold_seed_df=pd.DataFrame(threshold_seed); threshold_stats_df=pd.DataFrame(threshold_stats)

    # Clean accuracy contrasts.
    clean_seed=[]; clean_stats=[]
    for fam in FAMILIES:
        b=cleans.query("architecture=='vitb_amp' and family==@fam").sort_values("seed")
        s=cleans.query("architecture=='vits_amp' and family==@fam").sort_values("seed")
        vals=s.clean_accuracy_percent.to_numpy()-b.clean_accuracy_percent.to_numpy()
        for seed,bv,sv,d in zip(SEEDS,b.clean_accuracy_percent,s.clean_accuracy_percent,vals):
            clean_seed.append({"family":fam,"seed":seed,"vitb_clean_accuracy_percent":float(bv),"vits_clean_accuracy_percent":float(sv),"delta_pp_vits_minus_vitb":float(d)})
        clean_stats.append({"family":fam,"family_label":LABELS[fam],
                            "vitb_mean":float(b.clean_accuracy_percent.mean()),"vitb_sd":float(b.clean_accuracy_percent.std(ddof=1)),
                            "vits_mean":float(s.clean_accuracy_percent.mean()),"vits_sd":float(s.clean_accuracy_percent.std(ddof=1)),
                            **contrast_stats(vals)})
    clean_seed_df=pd.DataFrame(clean_seed); clean_stats_df=pd.DataFrame(clean_stats)

    # Learned censoring table.
    learned=endpoints.query("family=='learned'")[["architecture","seed","q80_status","q50_status","q20_status","q80_epsilon","q80_rho","q50_epsilon","q50_rho","q20_epsilon","q20_rho","max_rho","min_accuracy"]].sort_values(["seed","architecture"])

    # Protocol/provenance report.
    d033=json.loads(args.d033_provenance.read_text()); numerical=json.loads(args.attack_eval_mode.read_text())
    pipeline=json.loads(args.vitb_pipeline_report.read_text()) if args.vitb_pipeline_report and args.vitb_pipeline_report.exists() else None
    input_report={
        "status":"PASS", "checks":checks,
        "comparison":"ViT-B/16 AMP-trained vs ViT-S/16 AMP-trained; FP32 attack/eval",
        "analysis_label":"SEED_ALIGNED_CROSS_ARCHITECTURE",
        "strict_pairing":"NOT_CLAIMED",
        "shared_families":FAMILIES,"shared_seeds":SEEDS,
        "vitb":vitb_report,"vits":vits_report,
        "d033_provenance":d033,"attack_eval_numerical_mode":numerical,
        "vitb_pipeline_report":pipeline,
        "scope_limit":"Architecture is the matched factor of interest, but bit-identical initialization and training-order pairing are not demonstrated; results are seed-aligned, not strict paired causal estimates.",
    }

    summary={
        "status":"PASS_ANALYSIS_COMPLETE",
        "new_gpu_experiment_required":False,
        "comparison":"AMP-matched cross-architecture ViT-B/16 vs ViT-S/16",
        "training_regime":"AMP/mixed precision for both cohorts",
        "attack_eval_regime":"FP32",
        "families":FAMILIES,"seeds":SEEDS,
        "nauc_cross_rho_endpoint":cross_support,"support_limiter":limiter,
        "nauc_arch_endpoints":arch_support,"nauc_family_endpoints":family_support,
        "primary_attack_ranking_vitb":ranks_df.query("support_multiplier==1 and metric=='attack_nauc'").iloc[0].vitb_order,
        "primary_attack_ranking_vits":ranks_df.query("support_multiplier==1 and metric=='attack_nauc'").iloc[0].vits_order,
        "primary_noise_ranking_vitb":ranks_df.query("support_multiplier==1 and metric=='noise_nauc'").iloc[0].vitb_order,
        "primary_noise_ranking_vits":ranks_df.query("support_multiplier==1 and metric=='noise_nauc'").iloc[0].vits_order,
        "architecture_by_pe_interaction":interaction,
        "learned_q50_full_n6":"NOT_ESTIMABLE: ViT-B q50 not bracketed for seeds 123, 456, 1011",
        "main_interpretation":"AMP-matched backbone scaling changes adversarial robustness in a PE-dependent way while random-noise nAUC remains nearly unchanged. The ViT-B Learned>RoPE ordering reverses to RoPE>Learned in ViT-S. This is not a universal scaling law and does not include matched ALiBi robustness.",
        "alibi_scope":"ViT-B-only robustness plus separate ViT-S AMP training-collapse evidence; excluded from symmetric cross-architecture robustness metrics.",
    }

    # Write deterministic outputs.
    attacks.to_csv(out/"D059_ATTACK_POINT_LEVEL_v1.csv",index=False,float_format="%.17g")
    noises.to_csv(out/"D059_NOISE_POINT_LEVEL_v1.csv",index=False,float_format="%.17g")
    cleans.to_csv(out/"D059_CLEAN_ACCURACY_SEED_LEVEL_v1.csv",index=False,float_format="%.17g")
    endpoints.to_csv(out/"D059_ENDPOINTS_BY_ARCH_FAMILY_SEED_v1.csv",index=False,float_format="%.17g")
    status.to_csv(out/"D059_ENDPOINT_STATUS_SUMMARY_v1.csv",index=False)
    support.to_csv(out/"D059_CURVE_SUPPORT_AUDIT_v1.csv",index=False,float_format="%.17g")
    support_scopes.to_csv(out/"D059_SUPPORT_SCOPES_v1.csv",index=False,float_format="%.17g")
    naucs.to_csv(out/"D059_SEED_LEVEL_NAUC_v1.csv",index=False,float_format="%.17g")
    agg.to_csv(out/"D059_PRIMARY_NAUC_AGGREGATE_v1.csv",index=False,float_format="%.17g")
    contrast_seed_df.to_csv(out/"D059_SEED_LEVEL_ARCHITECTURE_CONTRASTS_v1.csv",index=False,float_format="%.17g")
    contrast_agg_df.to_csv(out/"D059_FAMILY_ARCHITECTURE_CONTRASTS_v1.csv",index=False,float_format="%.17g")
    ranks_df.to_csv(out/"D059_RANKING_AGREEMENT_v1.csv",index=False,float_format="%.17g")
    did_df.to_csv(out/"D059_INTERACTION_PAIRWISE_DID_v1.csv",index=False,float_format="%.17g")
    threshold_seed_df.to_csv(out/"D059_THRESHOLD_SEED_CONTRASTS_v1.csv",index=False,float_format="%.17g")
    threshold_stats_df.to_csv(out/"D059_THRESHOLD_CONTRAST_STATISTICS_v1.csv",index=False,float_format="%.17g")
    clean_seed_df.to_csv(out/"D059_CLEAN_ACCURACY_SEED_CONTRASTS_v1.csv",index=False,float_format="%.17g")
    clean_stats_df.to_csv(out/"D059_CLEAN_ACCURACY_CONTRAST_STATISTICS_v1.csv",index=False,float_format="%.17g")
    learned.to_csv(out/"D059_LEARNED_THRESHOLD_CENSORING_AUDIT_v1.csv",index=False,float_format="%.17g")
    dump_json(out/"D059_ARCHITECTURE_PE_INTERACTION_TEST_v1.json",interaction)
    dump_json(out/"D059_INPUT_AND_PROTOCOL_INTEGRITY_REPORT_v1.json",input_report)
    dump_json(out/"D059_ANALYSIS_SUMMARY_v1.json",summary)

    print(json.dumps(safe_json(summary), indent=2))

if __name__ == "__main__":
    main()
