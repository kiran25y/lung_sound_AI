#!/usr/bin/env python3
"""check_split_config.py — figure out which adaptation config produced a results.json.

The .pt checkpoint does NOT store training flags, but the SPLIT SIZES in results.json
fingerprint them:
  --smartphone-train-frac  folds phone patients into train  -> shrinks smartphone_* tests
  --sungbook-train-frac     folds Sungbook patients into train -> shrinks external_test

  # just read what the results.json implies + a verdict
  python check_split_config.py --results runs/aug_all/results.json

  # reproduce sizes for candidate fracs on your manifest and find the exact match
  python check_split_config.py --results runs/aug_all/results.json --manifest data/manifest.csv
"""
import argparse, json, itertools
KEYS = ["internal_test", "external_test", "smartphone_seoul", "smartphone_sungbook"]


def counts_from_results(path):
    sp = json.load(open(path))["splits"]
    out = {}
    for k in KEYS:
        if k in sp and "confusion" in sp[k]:
            out[k] = int(sum(sum(r) for r in sp[k]["confusion"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    obs = counts_from_results(args.results)
    print("[observed in results.json]")
    for k in KEYS:
        if k in obs:
            print(f"  {k:<22}{obs[k]}")
    sps = obs.get("smartphone_seoul", 0)
    print("\n[verdict]")
    print("  device adaptation (smartphone): " +
          ("OFF — full phone test, no phone in training" if sps > 1500
           else "ON — phone patients folded into training"))
    print("  -> if OFF, the model over-alarms on phones (high false_alarm). Retrain with "
          "--smartphone-train-frac 0.5 to fix.")

    if not args.manifest:
        print("\n(add --manifest data/manifest.csv to find the EXACT --*-train-frac that produced this)")
        return

    import phase1_baseline as pb
    pb.set_multilabel(True)
    print(f"\n{'sungbook':>9}{'phone':>7}   {'internal':>9}{'external':>9}{'sp_seoul':>9}{'sp_sung':>9}   match")
    for sbf, spf in itertools.product([0.0, 0.3, 0.4, 0.5],
                                      [0.0, 0.3, 0.4, 0.5]):
        s = pb.build_splits(args.manifest, sungbook_train_frac=sbf,
                            smartphone_train_frac=spf, balance_normal_ratio=None)
        s = pb.filter_existing(s)
        row = {k: len(s[k]) for k in KEYS}
        match = all(abs(row[k] - obs.get(k, -1)) <= 3 for k in KEYS if k in obs)
        print(f"{sbf:>9}{spf:>7}   {row['internal_test']:>9}{row['external_test']:>9}"
              f"{row['smartphone_seoul']:>9}{row['smartphone_sungbook']:>9}   {'<== MATCH' if match else ''}")


if __name__ == "__main__":
    main()
