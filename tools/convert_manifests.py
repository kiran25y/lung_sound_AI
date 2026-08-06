#!/usr/bin/env python3
"""Convert the Hanil hospital manifests (train_manifest.csv + test_manifest.csv,
which use *different* schemas) into ONE unified manifest.csv the pipeline expects:

    filepath, patient_id, label, hospital, device, split

Handles:
  * train schema:  path,label,patient_id,device,hospital,split
  * test  schema:  filepath,basename,label_4cls,class_name_4cls,label_2cls,...
  * the 4th 'both' class -> keep (4-class) or drop (3-class) via --classes
  * patient id extraction from the "NNN)..." filename pattern when no column exists
  * hospital/device normalisation (seoul_combined->seoul, HospitalStethoscope->stethoscope)
  * optional path remap for the archive location (--path-replace OLD=NEW)

USAGE
  python tools/convert_manifests.py \
      --train /workspace/dataset/lungsound/train_manifest.csv \
      --test  /workspace/dataset/lungsound/test_manifest.csv \
      --classes 4 \
      --out data/manifest.csv
"""
from __future__ import annotations
import argparse
import re
import sys
from collections import Counter
from pathlib import Path
import pandas as pd

LABEL_NORM = {
    "normal": "normal", "n": "normal",
    "crackle": "crackle", "crackles": "crackle", "c": "crackle",
    "wheeze": "wheeze", "wheezes": "wheeze", "w": "wheeze",
    "both": "both", "b": "both",
}
FILE_COLS = ["filepath", "path", "file", "filename", "wav", "audio", "audio_path"]
LABEL_COLS = ["label", "class_name_4cls", "class_name", "class_name_2cls", "diagnosis"]


def norm_label(x) -> str | None:
    return LABEL_NORM.get(str(x).strip().lower())


def norm_hospital(x: str) -> str | None:
    x = x.lower()
    if "sungbook" in x or "sungbuk" in x or "아카이브" in x or "songbuk" in x:
        return "sungbook"
    if "seoul" in x:
        return "seoul"
    return None


def norm_device(x: str) -> str | None:
    x = x.lower()
    if "steth" in x or "scope" in x:
        return "stethoscope"
    if "phone" in x or "iphone" in x or "android" in x or "mobile" in x:
        return "smartphone"
    return None


def pick(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def extract_patient(basename: str) -> str | None:
    m = re.match(r"^(\d+)\)", str(basename))          # 50013)1-1-N.wav -> 50013
    if m:
        return m.group(1)
    m = re.search(r"(p\d+)", str(basename), re.IGNORECASE)
    return m.group(1) if m else None


def convert_one(csv_path: str, default_split: str, default_hospital: str | None,
                default_device: str, path_replace, classes: int,
                rebase_dir: str | None = None) -> pd.DataFrame:
    import os
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    fcol = pick(cols, FILE_COLS)
    lcol = pick(cols, LABEL_COLS)
    if not fcol or not lcol:
        sys.exit(f"{csv_path}: could not find file/label columns among {cols}")

    out = pd.DataFrame()
    out["filepath"] = df[fcol].astype(str)
    if rebase_dir:
        # reconstruct every path as <rebase_dir>/<basename>. Robust to wrong parent
        # folders AND to Unicode (NFC/NFD) differences in the original path.
        rebase_dir = rebase_dir.rstrip("/")
        out["filepath"] = out["filepath"].map(lambda p: f"{rebase_dir}/{os.path.basename(p)}")
    for old, new in (path_replace or []):        # apply every replacement in order
        out["filepath"] = out["filepath"].str.replace(old, new, regex=False)
    out["label"] = df[lcol].map(norm_label)

    # patient id: prefer an existing column, else extract from basename
    pcol = pick(cols, ["patient_id", "patient", "pid"])
    base = df[pick(cols, ["basename"])] if pick(cols, ["basename"]) else out["filepath"].map(lambda p: Path(p).name)
    if pcol:
        out["patient_raw"] = df[pcol].astype(str)
    else:
        out["patient_raw"] = base.map(extract_patient)

    # hospital
    hcol = pick(cols, ["hospital"])
    if hcol:
        out["hospital"] = df[hcol].map(lambda v: norm_hospital(str(v)) or default_hospital)
    else:
        out["hospital"] = out["filepath"].map(lambda p: norm_hospital(p) or default_hospital)

    # device
    dcol = pick(cols, ["device"])
    if dcol:
        out["device"] = df[dcol].map(lambda v: norm_device(str(v)) or default_device)
    else:
        out["device"] = out["filepath"].map(lambda p: norm_device(p) or default_device)

    out["split"] = df[pick(cols, ["split"])] if pick(cols, ["split"]) else default_split

    # namespace patient ids by hospital so seoul 432 != sungbook 432
    out["patient_id"] = [f"{h}_{p}" for h, p in zip(out["hospital"], out["patient_raw"])]

    out = out.dropna(subset=["label"])                 # drop rows with unmapped labels
    if classes == 3:
        out = out[out["label"] != "both"]
    return out[["filepath", "patient_id", "label", "hospital", "device", "split"]]


def main():
    ap = argparse.ArgumentParser(description="Convert Hanil manifests to unified format")
    ap.add_argument("--train", required=True, help="Seoul internal manifest")
    ap.add_argument("--test", required=True, help="Sungbook external manifest")
    ap.add_argument("--classes", type=int, choices=[3, 4], default=4,
                    help="4 = keep 'both' (ICBHI-standard); 3 = drop 'both'")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--train-dir", default=None,
                    help="rebase every TRAIN file to <dir>/<basename> (fixes wrong parent "
                         "folders + Unicode path issues). e.g. .../hospital_dataset/audio_data")
    ap.add_argument("--test-dir", default=None,
                    help="rebase every TEST file to <dir>/<basename>. "
                         "e.g. .../stethoscope_dataset_sungbook")
    ap.add_argument("--path-replace", action="append", default=None,
                    help='remap a path substring (repeatable), '
                         'e.g. --path-replace "/workspace/아카이브=/workspace/dataset/lungsound" '
                         '--path-replace "stethoscope_dataset_seoul=hospital_dataset/audio_data"')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prs = []
    for pr in (args.path_replace or []):
        if "=" in pr:
            prs.append(tuple(pr.split("=", 1)))
        else:
            sys.exit(f"--path-replace must be 'OLD=NEW', got: {pr}")

    tr = convert_one(args.train, "train", "seoul", "stethoscope", prs, args.classes,
                     rebase_dir=args.train_dir)
    te = convert_one(args.test, "external_test", "sungbook", "stethoscope", prs, args.classes,
                     rebase_dir=args.test_dir)
    man = pd.concat([tr, te], ignore_index=True)

    print(f"[convert] classes={args.classes}  (both {'kept' if args.classes==4 else 'dropped'})")
    print(f"  train rows : {len(tr):>6}  ({tr.hospital.iloc[0] if len(tr) else '-'})")
    print(f"  test  rows : {len(te):>6}  ({te.hospital.iloc[0] if len(te) else '-'})")
    print(f"  TOTAL      : {len(man):>6}")
    print(f"  by label   : {dict(Counter(man.label))}")
    print(f"  by hosp×dev: {dict(Counter(zip(man.hospital, man.device)))}")
    print(f"  patients   : seoul={tr.patient_id.nunique()}  sungbook={te.patient_id.nunique()}")
    c = Counter(man.label); top = max(c.values()); tot = sum(c.values())
    print(f"  imbalance  : majority class = {100*top/tot:.0f}% of clips "
          f"-> use focal loss + per-class metrics")

    # verify files actually exist on disk BEFORE writing — catches path issues now
    import os
    exists = man["filepath"].map(os.path.exists)
    n_missing = int((~exists).sum())
    if n_missing:
        print(f"\n  ! {n_missing}/{len(man)} files NOT found on disk. Examples:")
        for p in man.loc[~exists, "filepath"].head(5):
            print(f"      {p}")
        print("  -> add/adjust --path-replace so these resolve, then re-run.")
    else:
        print(f"\n  ✓ all {len(man)} files found on disk.")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(args.out, index=False)
    print(f"\n[done] wrote {len(man)} rows -> {args.out}")
    nc = args.classes
    print(f"Next: python phase1_baseline.py --manifest {args.out} --epochs 60"
          + ("" if nc == 3 else "  --num-classes 4"))


if __name__ == "__main__":
    main()
