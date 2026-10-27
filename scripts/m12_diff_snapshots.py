#!/usr/bin/env python3
"""Collect/recompare three shipped-reference runs and three predication controls.

Run without flags to compare retained captures; --collect invokes the existing
C++ experiment CLI. No inference is performed in Python. Every artifact stays
under build/reports/m12/ref/{run1,run2,run3,control1,control2,control3}.
"""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "build/reports/m12"
REF = REPORT / "ref"
SCRATCH_END = 20736000


def compare(paths: list[Path], half: bool = False, region: tuple[int, int] | None = None):
    data = [path.read_bytes() for path in paths]
    if len({len(value) for value in data}) != 1:
        raise ValueError("capture sizes disagree")
    expected_bytes = 1920 * 1080 * 8 * 2 if half else SCRATCH_END + 1024
    if len(data[0]) != expected_bytes:
        raise ValueError(f"capture size {len(data[0])} != expected {expected_bytes}")
    hashes = [hashlib.sha256(value).hexdigest() for value in data]
    result = {"files": [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths],
              "sha256": hashes, "bytesPerFile": len(data[0])}
    if half:
        values = [memoryview(value).cast("H") for value in data]
    else:
        values = data
    count = 0
    rows, columns = set(), set()
    max_delta = 0.0
    outside = 0
    # Skip identical chunks; avoid a full Python byte walk for deterministic
    # snapshots, while comparing all bytes (not merely trusting their hashes).
    chunk = 32768
    for start in range(0, len(values[0]), chunk):
        end = min(start + chunk, len(values[0]))
        if all(value[start:end] == values[0][start:end] for value in values[1:]):
            continue
        for index in range(start, end):
            if all(value[index] == values[0][index] for value in values[1:]):
                continue
            count += 1
            if half:
                pixel = index // 8
                columns.add(pixel % 1920)
                rows.add(pixel // 1920)
                decoded = [struct.unpack_from("<e", value, index * 2)[0] for value in data]
                max_delta = max(max_delta, max(decoded) - min(decoded))
            elif region is not None:
                begin, limit = region
                if begin <= index < limit:
                    pixel = (index - begin) // 16
                    columns.add(pixel % 960)
                    rows.add(pixel // 960)
                else:
                    outside += 1
    result.update({"differentHalfwords" if half else "differentBytes": count,
                   "minX": min(columns) if columns else None,
                   "maxX": max(columns) if columns else None,
                   "rowsAffected": len(rows)})
    if half:
        result["maxAbsDelta"] = max_delta
    elif region:
        result["differentBytesOutsideRegion"] = outside
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--prefix", default="", help="fresh capture-set prefix, letters/digits/hyphens only")
    parser.add_argument("--legacy", action="store_true", help="explicitly compare old captures without identity manifests")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]*", args.prefix):
        raise ValueError("invalid prefix")
    if args.collect and args.legacy:
        raise ValueError("new captures must include provenance; --legacy is compare-only")
    names = [f"{args.prefix}run{i}" for i in range(1, 4)] + [f"{args.prefix}control{i}" for i in range(1, 4)]
    def identities():
        paths = [ROOT / "build/bin/rdna3-int8.exe"]
        paths += [ROOT / f"build/bin/shaders/m12ref/pass{i}.cso" for i in range(14)]
        paths += [ROOT / "build/bin/shaders/m12control/pass11.cso", REPORT / "control/manifest.json"]
        return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths}
    if args.collect:
        for ordinal, name in enumerate(names):
            directory = REF / name
            if directory.exists():
                raise FileExistsError(f"refusing to overwrite retained evidence: {directory}")
            REF.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({"M12_SNAPSHOTS": "1", "M12_DUMP": "1", "M12_RUN_ID": name,
                        "M12_PREDICATED": "1" if ordinal >= 3 else "0"})
            before = identities()
            completed = subprocess.run([str(ROOT / "build/bin/rdna3-int8.exe"),
                    "run", "fsr4-oracle"], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", check=False)
            if completed.returncode != 0:
                # A competing writer may own `directory`: never write into it
                # after our subprocess failed exclusive directory creation.
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=REF,
                        prefix=f"{name}-", suffix=".failed.log", delete=False) as failed:
                    failed.write(completed.stdout)
                    log_path = Path(failed.name)
                raise RuntimeError(f"{name}: oracle failed; see {log_path}")
            with (directory / "run.log").open("x", encoding="utf-8") as log:
                log.write(completed.stdout)
            if identities() != before:
                raise RuntimeError("executable or shaders changed during capture")
            manifest = {"mode": "predicated derivative control" if ordinal >= 3 else "unmodified reference",
                        "implementationSha256": before,
                        "inputRecipe": "LCG seed0x5EED4F12; 1664525*s+1013904223 mod2^32; ((s>>16)%4095-2047)/1024; 7 valid FP16 channels; channel7 and padded rows zero",
                        "inputFnv1aLog": re.findall(r"input features:.*", completed.stdout)}
            with (directory / "manifest.json").open("x", encoding="utf-8") as output:
                output.write(json.dumps(manifest, indent=2) + "\n")
            print(f"captured {name}", flush=True)

    manifests = {name: json.loads((REF / name / "manifest.json").read_text(encoding="utf-8"))
                 if (REF / name / "manifest.json").exists() else None for name in names}
    if not args.legacy:
        if any(manifest is None for manifest in manifests.values()):
            raise ValueError("missing capture identity manifest (old evidence requires explicit --legacy)")
        baseline = manifests[names[0]]
        if len(baseline["inputFnv1aLog"]) != 1:
            raise ValueError("expected exactly one input hash record")
        for ordinal, name in enumerate(names):
            manifest = manifests[name]
            if manifest["implementationSha256"] != baseline["implementationSha256"]:
                raise ValueError("implementation identities changed between captures")
            if manifest["inputFnv1aLog"] != baseline["inputFnv1aLog"] or manifest["inputRecipe"] != baseline["inputRecipe"]:
                raise ValueError("input identities changed between captures")
            expected_mode = "predicated derivative control" if ordinal >= 3 else "unmodified reference"
            if manifest["mode"] != expected_mode:
                raise ValueError("capture control/reference mode mismatch")
    summary = {"scope": "gfx1101, v07 INT8 Quality 1080p, synthetic LCG fixture",
               "provenanceVerified": not args.legacy,
               "reference": [], "control": [], "tailNonzeroBytes": {}}
    for kind, run_names in (("reference", names[:3]), ("control", names[3:])):
        for index in range(14):
            result = compare([REF / name / f"scratch-pass{index}.bin" for name in run_names],
                             region=(12441600, SCRATCH_END) if index == 11 else None)
            result["pass"] = index
            summary[kind].append(result)
            print(f"{kind} pass {index:2}: {result['differentBytes']} differing bytes", flush=True)
        summary[kind + "Output"] = compare([REF / name / "output.bin" for name in run_names], half=True)
    summary["referenceVsControlOutput"] = compare(
        [REF / names[0] / "output.bin", REF / names[3] / "output.bin"], half=True)
    for name in names:
        summary["tailNonzeroBytes"][name] = {
            str(index): sum(value != 0 for value in
                (REF / name / f"scratch-pass{index}.bin").read_bytes()[SCRATCH_END:])
            for index in (9, 10, 11, 12, 13)}
    summary["captureManifests"] = manifests
    (REPORT / f"{args.prefix}determinism.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("reference", "control")}, indent=2))
    # These assert the causal-control expectation, not the original graph's
    # correctness. x<132 is checked for THIS captured fixture only.
    if any(item["differentBytes"] for item in summary["reference"][:11]):
        raise RuntimeError("reference variation before pass11")
    if any(item["differentBytes"] for item in summary["control"]) or summary["controlOutput"]["differentHalfwords"]:
        raise RuntimeError("predicated control is not deterministic")
    if summary["referenceVsControlOutput"]["maxX"] is not None and summary["referenceVsControlOutput"]["maxX"] >= 132:
        raise RuntimeError("reference/control difference extends outside predicted band")


if __name__ == "__main__":
    main()
