"""Offline inspection of the M13 packed half-history/RFM + linear-RGB captures."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib


WIDTH, HEIGHT = 1920, 1080
ROOT = Path(__file__).resolve().parents[1]


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to((ROOT / "build").resolve()):
        parser.error("generated output must be under build/")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"display_transform": "Reinhard v/(1+v), gamma 1/2.2 (visualization only)",
              "width": WIDTH, "height": HEIGHT, "frames": []}
    hashes = []
    passed = True
    for frame in range(3):
        raw = (args.captures / f"frame{frame}.bin").read_bytes()
        if len(raw) != WIDTH * HEIGHT * 32:
            raise ValueError("incorrect packed frame extent")
        sha = hashlib.sha256(raw).hexdigest()
        hashes.append(sha)
        nonfinite = recurrence_errors = 0
        low, high = math.inf, -math.inf
        scanlines = bytearray()
        for pixel, record in enumerate(struct.iter_unpack("<8e4f", raw)):
            nonfinite += sum(not math.isfinite(v) for v in record)
            recurrence_errors += sum(not 0 <= v <= 1 for v in record[4:8])
            if pixel % WIDTH == 0:
                scanlines.append(0)
            for value in record[8:11]:
                low, high = min(low, value), max(high, value)
                if not math.isfinite(value):
                    value = 0
                mapped = (max(value, 0) / (1 + max(value, 0))) ** (1 / 2.2)
                scanlines.append(min(255, max(0, round(mapped * 255))))
        image = b"\x89PNG\r\n\x1a\n"
        image += chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
        image += chunk(b"IDAT", zlib.compress(scanlines))
        image += chunk(b"IEND", b"")
        (args.output_dir / f"frame{frame}.png").write_bytes(image)
        input_hash = hashlib.sha256((args.captures / f"input-frame{frame}.bin").read_bytes()).hexdigest()
        report["frames"].append({"frame": frame, "packed_sha256": sha,
                                 "input_sha256": input_hash, "nonfinite": nonfinite,
                                 "recurrence_outside_0_1": recurrence_errors,
                                 "linear_rgb_min": low, "linear_rgb_max": high})
        passed &= nonfinite == 0 and recurrence_errors == 0 and low >= 0 and high <= 64000
    report["camera_cut_matches_clean_initial"] = hashes[0] == hashes[2]
    report["continued_output_differs"] = hashes[0] != hashes[1]
    passed &= report["camera_cut_matches_clean_initial"] and report["continued_output_differs"]
    report["inspection_pass"] = bool(passed)
    (args.output_dir / "inspection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"M13 offline inspection {'PASS' if passed else 'FAIL'}: {args.output_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
