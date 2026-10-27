#!/usr/bin/env python3
r"""Build data/manifest.csv for the Hanil pipeline.

The manifest is the single source of truth: one row per clip with columns
    filepath, patient_id, label, hospital, device
The pipeline derives all splits from the hospital/device columns
(seoul+stethoscope = internal, sungbook+stethoscope = external, *+smartphone = device).

Two ways to assign hospital/device, and two ways to get the label. Mix as needed.

────────────────────────── ASSIGNING hospital / device ──────────────────────────
A) EXPLICIT (recommended — no guessing). One --source per folder:
     --source "/path/to/folder|hospital|device"

   Example for the Hanil layout:
     python tools/build_manifest.py \
       --source "/workspace/dataset/lungsound/hospital_dataset/audio_data|seoul|stethoscope" \
       --source "/workspace/dataset/lungsound/stethoscope_dataset_sungbook|sungbook|stethoscope" \
       --source "/workspace/dataset/lungsound/smartphone/iphone_dataset_seoul|seoul|smartphone" \
       --source "/workspace/dataset/lungsound/smartphone/iphone_dataset_sungbook|sungbook|smartphone" \
       --patient-regex "(p\d+)" --out data/manifest.csv --dry-run

B) AUTO (keyword inference from path/filename): use --root instead of --source.

────────────────────────────── GETTING the label ───────────────────────────────
1) FROM FILENAME (default): a label keyword in the filename (--label-map to override).
2) FROM AN EXISTING CSV: --labels-from train_manifest.csv --labels-from test_manifest.csv
   Matched by the audio file's basename. The tool auto-detects the file and label
   columns (or set --labels-file-col / --labels-label-col). If you also pass
   --carry-split, a `split` column is written (train/test) taken from which CSV each
   file came from, so you can reuse your existing train/test division.

Always run with --dry-run first and read the summary before writing.
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}

DEFAULT_LABEL_MAP = {
    "normal": "normal", "norm": "normal", "healthy": "normal", "n": "normal",
    "crackle": "crackle", "crackles": "crackle", "crackling": "crackle", "c": "crackle",
    "wheeze": "wheeze", "wheezes": "wheeze", "wheezing": "wheeze", "w": "wheeze",
}
DEFAULT_HOSPITAL_MAP = {"seoul": "seoul", "sungbook": "sungbook",
                        "sungbuk": "sungbook", "songbuk": "sungbook"}
DEFAULT_DEVICE_MAP = {"stethoscope": "stethoscope", "steth": "stethoscope",
                      "smartphone": "smartphone", "phone": "smartphone",
                      "iphone": "smartphone", "android": "smartphone", "mobile": "smartphone"}

FILE_COL_CANDIDATES = ["filepath", "file", "filename", "path", "wav", "audio", "audio_path"]
LABEL_COL_CANDIDATES = ["label", "class", "target", "y", "diagnosis", "category"]


def parse_map(s, default):
    if not s:
        return default
    out = {}
    for pair in s.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out or default


def match_keyword(haystack, mapping):
    hits = [(k, v) for k, v in mapping.items() if k in haystack]
    if not hits:
        return None
    hits.sort(key=lambda kv: len(kv[0]), reverse=True)   # longest key wins
    return hits[0][1]


def match_label_token(stem, mapping):
    """Match label as a whole token (split on _,-,space,.) so 'n' doesn't fire on 'normal'
    and 'c' doesn't fire inside 'scope'. Falls back to substring for multi-char keys."""
    tokens = set(re.split(r"[_\-\s\.]+", stem.lower()))
    # exact-token match first (handles single-letter codes safely)
    for k, v in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        if k in tokens:
            return v
    # then substring for words >= 4 chars (e.g. 'crackles' inside 'fine_crackles_x')
    for k, v in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        if len(k) >= 4 and k in stem.lower():
            return v
    return None


def infer_patient(stem, regex):
    if regex:
        m = re.search(regex, stem, flags=re.IGNORECASE)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return re.split(r"[_\-\s]", stem, maxsplit=1)[0]


def load_label_lookup(csv_paths, file_col, label_col):
    """basename(no ext) -> (label, source_split). source_split from filename of the CSV."""
    lookup = {}
    for cp in csv_paths:
        split_tag = "train" if "train" in Path(cp).stem.lower() else \
                    ("test" if "test" in Path(cp).stem.lower() else Path(cp).stem)
        with open(cp, newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            fc = file_col or next((c for c in cols if c.lower() in FILE_COL_CANDIDATES), None)
            lc = label_col or next((c for c in cols if c.lower() in LABEL_COL_CANDIDATES), None)
            if not fc or not lc:
                print(f"  ! {cp}: could not find file/label columns in {cols}. "
                      f"Use --labels-file-col / --labels-label-col.", file=sys.stderr)
                continue
            n = 0
            for row in reader:
                key = Path(str(row[fc])).stem.lower()
                lookup[key] = (str(row[lc]).strip().lower(), split_tag)
                n += 1
            print(f"  [labels] {cp}: {n} rows via '{fc}'->'{lc}' (split='{split_tag}')")
    return lookup


def collect_sources(args):
    """Return list of (path, hospital, device) roots."""
    srcs = []
    for s in (args.source or []):
        parts = s.split("|")
        if len(parts) != 3:
            sys.exit(f"--source must be 'path|hospital|device', got: {s}")
        srcs.append((parts[0], parts[1].strip().lower(), parts[2].strip().lower()))
    for r in (args.root or []):
        srcs.append((r, None, None))         # auto-infer
    if not srcs:
        sys.exit("Provide at least one --source 'path|hospital|device' or --root PATH")
    return srcs


def build(args):
    label_map = parse_map(args.label_map, DEFAULT_LABEL_MAP)
    hospital_map = parse_map(args.hospital_map, DEFAULT_HOSPITAL_MAP)
    device_map = parse_map(args.device_map, DEFAULT_DEVICE_MAP)

    label_lookup = {}
    if args.labels_from:
        print("[labels] loading labels from existing CSV(s):")
        label_lookup = load_label_lookup(args.labels_from, args.labels_file_col,
                                         args.labels_label_col)

    rows, skipped, previews = [], [], []
    for root, force_h, force_d in collect_sources(args):
        if not os.path.isdir(root):
            print(f"  ! source not found (skipping): {root}", file=sys.stderr)
            continue
        for dirpath, _d, names in os.walk(root):
            for n in sorted(names):
                if Path(n).suffix.lower() not in AUDIO_EXT:
                    continue
                path = os.path.join(dirpath, n)
                stem = Path(n).stem
                key = stem.lower()
                parent_l = dirpath.lower()

                # label
                split_tag = ""
                if key in label_lookup:
                    label, split_tag = label_lookup[key]
                    label = label_map.get(label, label)   # normalise via map if needed
                else:
                    # filename first, then the immediate parent FOLDER NAME (class-folder
                    # layouts) — both via safe token matching, never substring on the full path
                    label = (match_label_token(stem, label_map)
                             or match_label_token(Path(dirpath).name, label_map))
                if label not in ("normal", "crackle", "wheeze"):
                    skipped.append(path)
                    continue

                hospital = force_h or match_keyword(parent_l, hospital_map) \
                    or match_keyword(key, hospital_map) or args.default_hospital
                device = force_d or match_keyword(parent_l, device_map) \
                    or match_keyword(key, device_map) or args.default_device
                patient = infer_patient(stem, args.patient_regex)
                pid = patient if patient.startswith(hospital) else f"{hospital}_{patient}"

                row = dict(filepath=os.path.abspath(path), patient_id=pid,
                           label=label, hospital=hospital, device=device)
                if args.carry_split and split_tag:
                    row["split"] = split_tag
                rows.append(row)
                if len(previews) < 10:
                    previews.append((n, pid, label, hospital, device, split_tag or "-"))

    # ── report ────────────────────────────────────────────────────────────────
    print("\n[preview] filename -> patient | label | hospital | device | split")
    for n, pid, lab, hos, dev, sp in previews:
        print(f"  {n:<38} {pid:<16} {lab:<8} {hos:<9} {dev:<11} {sp}")
    print("\n[summary]")
    print(f"  matched    : {len(rows)}")
    print(f"  skipped    : {len(skipped)} (no valid normal/crackle/wheeze label)")
    print(f"  patients   : {len(set(r['patient_id'] for r in rows))}")
    print(f"  by label   : {dict(Counter(r['label'] for r in rows))}")
    print(f"  by hospital: {dict(Counter(r['hospital'] for r in rows))}")
    print(f"  by device  : {dict(Counter(r['device'] for r in rows))}")
    print(f"  steth×hosp : {dict(Counter((r['hospital'],r['device']) for r in rows))}")

    n_pat = len(set(r["patient_id"] for r in rows))
    if rows and n_pat > 0.8 * len(rows):
        print("\n  ! WARNING: almost every clip has a unique patient_id. Clips from the same "
              "patient must share an id or the patient-level split is defeated. "
              "Tune --patient-regex to capture your patient token.")
    if not any(r["hospital"] == "sungbook" and r["device"] == "stethoscope" for r in rows):
        print("  ! NOTE: no sungbook+stethoscope clips — external_test will be empty.")
    if skipped:
        print(f"\n  {min(len(skipped),12)} of {len(skipped)} skipped examples:")
        for p in skipped[:12]:
            print(f"    - {Path(p).name}")
        print("  -> either add their label token to --label-map, or supply --labels-from CSV.")

    if args.dry_run:
        print("\n[dry-run] nothing written. Re-run without --dry-run to write.")
        return 0
    if not rows:
        sys.exit("No rows matched — nothing to write.")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["filepath", "patient_id", "label", "hospital", "device"]
    if args.carry_split and any("split" in r for r in rows):
        fields.append("split")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n[done] wrote {len(rows)} rows -> {out}")
    print(f"Next: python phase1_baseline.py --manifest {out} --epochs 60")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Build manifest.csv for the Hanil pipeline")
    ap.add_argument("--source", action="append", help="'path|hospital|device' (repeatable)")
    ap.add_argument("--root", action="append", help="auto-infer hospital/device (repeatable)")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--labels-from", action="append",
                    help="existing CSV to pull labels from, matched by filename (repeatable)")
    ap.add_argument("--labels-file-col", default=None)
    ap.add_argument("--labels-label-col", default=None)
    ap.add_argument("--carry-split", action="store_true",
                    help="write a 'split' column from which CSV each file came from")
    ap.add_argument("--label-map", default=None)
    ap.add_argument("--hospital-map", default=None)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--patient-regex", default=None, help='e.g. "(p\\d+)"')
    ap.add_argument("--default-hospital", default="seoul")
    ap.add_argument("--default-device", default="stethoscope")
    ap.add_argument("--dry-run", action="store_true")
    sys.exit(build(ap.parse_args()))


if __name__ == "__main__":
    main()
