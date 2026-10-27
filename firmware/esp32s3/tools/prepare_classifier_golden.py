#!/usr/bin/env python3
"""Prepare quantized feature payloads for a serial ESP32 golden-set replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from send_wav import fnv1a, load_pcm16, quantize, reference_logmel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--input-scale", type=float, default=0.0510409623)
    parser.add_argument("--input-zero-point", type=int, default=5)
    args = parser.parse_args()

    document = json.loads(args.expected_json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, object]] = []
    for index, row in enumerate(document["rows"], 1):
        wav = Path(row["wav"])
        input_q = quantize(
            reference_logmel(load_pcm16(wav)),
            args.input_scale,
            args.input_zero_point,
        )
        payload = input_q.tobytes(order="C")
        input_hash = fnv1a(payload)
        if input_hash != row["input_qhash"]:
            raise RuntimeError(
                f"feature hash changed for {wav.name}: {input_hash} != "
                f"{row['input_qhash']}"
            )
        payload_path = args.out_dir / f"{index:03d}.bin"
        payload_path.write_bytes(payload)
        prepared.append({
            "wav": str(wav.resolve()),
            "label": row["label"],
            "truth2": row["truth2"],
            "truth3": row["truth3"],
            "input_qhash": input_hash,
            "host_out_q": row["host_out_q"],
            "payload": str(payload_path.resolve()),
        })

    manifest = {
        "input_scale": args.input_scale,
        "input_zero_point": args.input_zero_point,
        "rows": prepared,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"prepared {len(prepared)} payloads -> {manifest_path}")


if __name__ == "__main__":
    main()
