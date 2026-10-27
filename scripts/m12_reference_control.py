#!/usr/bin/env python3
"""Generate a clearly labelled pass11 predication control under build/ only.

The pinned source tree is never modified. This derivative is a causal control,
NOT the shipped reference. Only the 32/1 DP4A store gains a logical XY guard.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "ml2code_runtime/operators/int8_NHWC/Fused/FNB_CT2D_ADD.hlsli"
SOURCE = ROOT / ".references/sources/FSR-4.0.2-reference/Kits/FidelityFX/upscalers/fsr4/dx12" / RELATIVE
OUT = ROOT / "build/reports/m12/control"


def main():
    original = SOURCE.read_bytes()
    if hashlib.sha256(original).hexdigest() != "72595be6739dda4000f3ebd7a0548474871dfec51c3f70b46e6e51724579ce8d":
        raise ValueError("pinned FNB_CT2D_ADD source hash mismatch")
    text = original.decode("utf-8")
    marker = "const uint2 poBase2D = computeShaderParams.dispatchThreadID.xy * 2 + pk;"
    # There are two DP4A overloads. The 64/2 overload already has this guard;
    # match only the unguarded 32/1 sequence so unexpected upstream changes fail.
    import re
    pattern = re.compile(r"(" + re.escape(marker) + r")\s*\n\s*(uint4 storeDwords)")
    replacement = (r"\1\n        // M12 causal control ONLY: not the shipped reference.\n"
                   r"        if (any(poBase2D >= output.logicalSize.xy)) continue;\n        \2")
    changed, count = pattern.subn(replacement, text)
    if count != 1:
        raise ValueError(f"expected exactly one unguarded 32/1 store, found {count}")
    destination = OUT / "include" / RELATIVE
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(changed, encoding="utf-8")
    manifest = {"purpose": "pass11 logical-size predicate causal control; NOT shipped reference",
                "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
                "sourceSha256": hashlib.sha256(original).hexdigest(),
                "controlSha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "substitutions": count}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"generated pass11 predication control: {destination}")


if __name__ == "__main__":
    main()
