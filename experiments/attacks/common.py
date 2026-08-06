#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXPECTED_SEEDS = (42, 123, 456, 789, 1011, 1213)
FAMILIES = ("learned", "sinusoidal", "rope", "alibi")
EXPECTED_TRAIN_IMAGES = 126_689
EXPECTED_VAL_IMAGES = 5_000
EXPECTED_CLASSES = 100
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CANONICAL_CHECKPOINT_FILENAME = "best_model.pth"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_object(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def image_paths_by_class(root: Path) -> List[Path]:
    paths: List[Path] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        paths.extend(sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS))
    return paths


def val_sample_records(val_dir: Path) -> List[Dict[str, Any]]:
    classes = sorted(p.name for p in val_dir.iterdir() if p.is_dir())
    records: List[Dict[str, Any]] = []
    for class_index, class_name in enumerate(classes):
        class_dir = val_dir / class_name
        for p in sorted(q for q in class_dir.iterdir() if q.is_file() and q.suffix.lower() in IMAGE_EXTS):
            records.append({
                "relative_path": str(p.relative_to(val_dir)),
                "class_index": class_index,
                "class_name": class_name,
            })
    return records


def valid_dataset_root(root: Path) -> Tuple[bool, Dict[str, Any]]:
    train, val = root / "train", root / "val"
    info: Dict[str, Any] = {"root": str(root), "exists": root.is_dir()}
    if not train.is_dir() or not val.is_dir():
        info["reason"] = "missing train/val"
        return False, info
    train_classes = sorted(p.name for p in train.iterdir() if p.is_dir())
    val_classes = sorted(p.name for p in val.iterdir() if p.is_dir())
    info.update(train_classes=len(train_classes), val_classes=len(val_classes), classes_match=train_classes == val_classes)
    if len(train_classes) != EXPECTED_CLASSES or len(val_classes) != EXPECTED_CLASSES or train_classes != val_classes:
        info["reason"] = "class mismatch"
        return False, info
    train_n, val_n = count_images(train), count_images(val)
    info.update(train_images=train_n, val_images=val_n)
    ok = train_n == EXPECTED_TRAIN_IMAGES and val_n == EXPECTED_VAL_IMAGES
    info["reason"] = "ok" if ok else "image-count mismatch"
    return ok, info


def resolve_dataset_root(parent: Path, override: Optional[str] = None) -> Tuple[Path, List[Dict[str, Any]]]:
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates += [parent, parent / "reextract_test" / "imagenet100_resized", parent / "imagenet100_resized"]
    if parent.is_dir():
        for p in parent.glob("*/*"):
            if p.is_dir() and (p / "train").is_dir() and (p / "val").is_dir():
                candidates.append(p)
        for p in parent.glob("*"):
            if p.is_dir() and (p / "train").is_dir() and (p / "val").is_dir():
                candidates.append(p)
    seen=set(); reports=[]; valid=[]
    for c in candidates:
        c=c.expanduser().resolve()
        if c in seen: continue
        seen.add(c)
        ok,info=valid_dataset_root(c); reports.append(info)
        if ok: valid.append(c)
    if not valid:
        raise RuntimeError("No valid ImageNet-100 root found. Candidates:\n" + json.dumps(reports, indent=2))
    valid.sort(key=lambda p: ("reextract_test" not in str(p), len(str(p))))
    return valid[0], reports


def checkpoint_path(models_parent: Path, family: str, seed: int) -> Path:
    """Return the only checkpoint filename accepted by both validator and engine."""
    return models_parent / f"{family}_seed{seed}" / CANONICAL_CHECKPOINT_FILENAME


def load_and_validate_split_file(path: Path, val_dir: Optional[Path] = None) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    split = json.loads(path.read_text(encoding="utf-8"))
    stored = split.get("split_sha256")
    if not stored:
        raise RuntimeError(f"Split file has no split_sha256: {path}")
    computed = sha256_json_object({k:v for k,v in split.items() if k != "split_sha256"})
    if computed != stored:
        raise RuntimeError(f"Split self-hash mismatch: stored={stored}, computed={computed}")
    required = ("calibration_indices", "attack_indices", "evaluation_indices", "counts", "sample_order_sha256")
    missing = [k for k in required if k not in split]
    if missing:
        raise RuntimeError(f"Split file missing keys: {missing}")
    cal=list(map(int,split["calibration_indices"])); attack=list(map(int,split["attack_indices"])); ev=list(map(int,split["evaluation_indices"]))
    all_idx=cal+attack+ev
    counts=split["counts"]
    if len(cal)!=int(counts["calibration"]) or len(attack)!=int(counts["attack"]) or len(ev)!=int(counts["evaluation"]):
        raise RuntimeError("Split counts do not match index-list lengths")
    if len(all_idx)!=int(counts["total"]) or len(set(all_idx))!=len(all_idx):
        raise RuntimeError("Split indices do not form a unique full partition")
    if min(all_idx, default=0)<0 or max(all_idx, default=-1)>=int(counts["total"]):
        raise RuntimeError("Split index out of range")
    if set(cal)&set(attack) or set(cal)&set(ev) or set(attack)&set(ev):
        raise RuntimeError("Split partitions overlap")
    if val_dir is not None:
        val_dir=val_dir.expanduser().resolve()
        records=val_sample_records(val_dir)
        if len(records)!=int(counts["total"]):
            raise RuntimeError(f"Current val sample count {len(records)} != frozen split total {counts['total']}")
        sample_hash=sha256_json_object(records)
        if sample_hash != split["sample_order_sha256"]:
            raise RuntimeError(f"Validation sample order hash mismatch: current={sample_hash}, frozen={split['sample_order_sha256']}")
    split["split_file"] = str(path)
    split["split_file_sha256"] = sha256_file(path)
    return split


def load_env_file(path: Path) -> Dict[str, str]:
    env = os.environ.copy()
    if not path.is_file(): raise FileNotFoundError(path)
    cmd=['bash','-lc','set -a; source "$1"; env -0','bash',str(path)]
    r=subprocess.run(cmd,capture_output=True)
    if r.returncode: raise RuntimeError(r.stderr.decode(errors='replace'))
    for item in r.stdout.split(b'\0'):
        if b'=' in item:
            k,v=item.split(b'=',1); env[k.decode()]=v.decode(errors='replace')
    return env


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)
