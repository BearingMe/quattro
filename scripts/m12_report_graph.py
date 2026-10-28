"""Offline numerical report for fsr4-graph logical captures (no GPU/runtime)."""

import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
BYTES = (8294400, 8294400, 8294400, 4147200, 4147200, 4147200,
         2073600, 2073600, 2073600, 4147200, 4147200, 8294400, 8294400, 33177600)
WIDTHS = (960, 960, 960, 480, 480, 480, 240, 240, 240, 480, 480, 960, 960, 1920)
CHANNELS = (16, 16, 16, 32, 32, 32, 64, 64, 64, 32, 32, 16, 16, 8)
PINNED_OUTPUT = "a57bcd063967affddf29cc24709b9d8a7fd9f8e7244e7010521d7b27f605b6de"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def halfwords(data):
    words = array.array("H")
    words.frombytes(data)
    if sys.byteorder != "little":
        words.byteswap()
    return words


def f16(word):
    return struct.unpack("<e", struct.pack("<H", word))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("--identities", type=Path, required=True,
                        help="capture-time artifact hash ledger, retained before changing builds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = (ROOT / "build").resolve()
    if not args.output.resolve().is_relative_to(build):
        parser.error("generated output must be under build/")
    # The ledger was recorded at capture time. Do not re-hash the shared exe
    # now: another configuration may already have replaced it. Preserve its
    # exact bytes/hash and explicitly separate it from numeric re-comparison.
    identities = args.identities.read_bytes()
    ledger = json.loads(identities.decode("utf-8-sig"))
    report = {"reference": "FSR 4.0.2 c5e34b4 fsr4_model_v07_i8_quality 1080p",
              "configuration": ledger["configuration"],
              "artifact_ledger": str(args.identities),
              "artifact_ledger_sha256": sha256(identities),
              "provenance_note": "Capture-time ledger supplied by collector; this script verifies numerical captures, not historical execution provenance.",
              "fixtures": []}
    passed = True
    for fixture in range(3):
        result = {"fixture": fixture, "passes": []}
        for index, expected_bytes in enumerate(BYTES):
            stem = args.captures / f"fixture{fixture}-pass{index}"
            own = Path(f"{stem}-own.bin").read_bytes()
            control = Path(f"{stem}-control.bin").read_bytes()
            if len(own) != expected_bytes or len(control) != expected_bytes:
                raise ValueError(f"pass {index}: incorrect logical extent")
            different = 0 if own == control else sum(a != b for a, b in zip(own, control))
            passed &= different == 0
            entry = {"pass": index, "bytes": expected_bytes,
                     "control_differing_bytes": different,
                     "own_sha256": sha256(own), "control_sha256": sha256(control)}
            if index == 13:
                words = halfwords(own)
                finite = nonzero = 0
                low, high = math.inf, -math.inf
                for word in words:
                    value = f16(word)
                    finite += math.isfinite(value)
                    nonzero += value != 0.0
                    low, high = min(low, value), max(high, value)
                entry["fp16"] = {"halfwords": len(words), "nonfinite": len(words) - finite,
                                 "nonzero": nonzero, "min": low, "max": high}
                passed &= finite == len(words) and nonzero > 0
                if fixture == 2:
                    entry["matches_prior_control_sha256"] = sha256(own) == PINNED_OUTPUT
                    passed &= entry["matches_prior_control_sha256"]
            if fixture == 2:
                shipped = Path(f"{stem}-shipped.bin").read_bytes()
                if len(shipped) != expected_bytes:
                    raise ValueError(f"pass {index}: shipped extent mismatch")
                band = {11: 64, 12: 66, 13: 132}.get(index, 0)
                stride = CHANNELS[index] * (2 if index == 13 else 1)
                byte_count = outside = unit_count = 0
                max_delta = 0.0
                if own != shipped:
                    for i, (a, b) in enumerate(zip(own, shipped)):
                        if a != b:
                            byte_count += 1
                            outside += (i // stride) % WIDTHS[index] >= band
                    if index == 13:
                        for a, b in zip(halfwords(own), halfwords(shipped)):
                            if a != b:
                                unit_count += 1
                                max_delta = max(max_delta, abs(f16(a) - f16(b)))
                    else:
                        unit_count = byte_count
                        for a, b in zip(own, shipped):
                            max_delta = max(max_delta, abs((a if a < 128 else a - 256) -
                                                         (b if b < 128 else b - 256)))
                entry["shipped"] = {"sha256": sha256(shipped), "differing_bytes": byte_count,
                                    "differing_units": unit_count,
                                    "unit": "halfword" if index == 13 else "INT8 byte",
                                    "max_abs_delta": max_delta, "band_x_less_than": band,
                                    "differing_bytes_outside_band": outside}
                passed &= outside == 0
            result["passes"].append(entry)
        report["fixtures"].append(result)
    report["numeric_pass"] = bool(passed)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{ledger['configuration']}: numeric {'PASS' if passed else 'FAIL'}; {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
