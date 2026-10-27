#!/usr/bin/env python3
"""Bind a completed full CTest run to the current executable/operator artifacts.

Run immediately after testing and BEFORE rebuilding the shared build/bin for
the other preset. This records evidence; it never runs inference or a build.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "build/reports/m12"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("debug", "release"), required=True)
    args = parser.parse_args()
    log = REPORT / f"{args.config}-final-ctest.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    completed = re.search(r"100% tests passed(?:, 0 tests failed)? out of (\d+)", text)
    if not completed or int(completed.group(1)) != 18:
        raise ValueError("expected a completed, passing full 18-test checkpoint suite")
    # Verbose output must contain the numeric assertions and repeat checks,
    # not merely CTest's generic pass summary.
    for model_pass in (1, 2, 12):
        for fixture in (0, 1):
            if f"ConvNext pass{model_pass} fixture{fixture}: 0/8294400 mismatches" not in text:
                raise ValueError("missing exact ConvNext comparison evidence")
    if text.count("repeated native execution: bit-identical") != 3:
        raise ValueError("missing ConvNext repeated-execution evidence")
    paths = [ROOT / "build/bin/rdna3-int8.exe", REPORT / "weights.bin", log]
    if (REPORT / "hardware.json").exists():
        paths.append(REPORT / "hardware.json")
    paths += sorted((ROOT / "build/bin/shaders/m12").glob("*.cso"))
    paths += sorted((ROOT / "build/bin/shaders/m12ref").glob("*.cso"))
    paths += [ROOT / "CMakeLists.txt", ROOT / "src/gpu/graph.cpp",
              ROOT / "src/gpu/shaders/conv_i8.hlsli", ROOT / "src/gpu/shaders/gemm_i8.hlsl",
              ROOT / "src/experiments/fsr4_reproduction.cpp"]
    paths += sorted((ROOT / "src/inference/models").rglob("*.cpp"))
    paths += sorted((ROOT / "src/inference/models").rglob("*.hlsl"))
    manifest = {"preset": args.config, "testsPassed": int(completed.group(1)),
                "scope": "partial M12: weights and passes0/1/2/12, NOT complete model gate",
                "sha256": {str(path.relative_to(ROOT)).replace("\\", "/"):
                           hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    destination = REPORT / f"{args.config}-validation-manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"recorded {args.config} full-suite/operator identities: {destination}")


if __name__ == "__main__":
    main()
