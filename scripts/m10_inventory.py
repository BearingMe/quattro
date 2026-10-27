"""Offline source survey; reads pinned upstream, writes only under build/."""
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parents[1]
model = root / ".references/sources/FSR-4.0.2-reference/Kits/FidelityFX/upscalers/fsr4/internal/shaders/fsr4_model_v07_i8_quality"
source = (model / "passes_1080.hlsl").read_text(encoding="utf-8")
calls = []
for match in re.finditer(r"^\s+(\w+(?:<[^>]+>)?)\(([^\n]*computeShaderParams)\);", source, re.MULTILINE):
    name, arguments = match.groups()
    biases = []
    for argument in arguments.split(", "):
        if "bias" not in argument:
            continue
        declaration = re.search(r"const ([^\n=]+) " + re.escape(argument) + r"\s*=", source)
        if declaration is None or not declaration.group(1).startswith("Tensor1h<"):
            raise RuntimeError(f"Unresolved/non-FP16 bias: {argument}")
        biases.append({"name": argument, "type": declaration.group(1),
                       "line": source.count("\n", 0, declaration.start()) + 1})
    calls.append({"operator": name, "line": source.count("\n", 0, match.start()) + 1,
                  "arguments": arguments.split(", "), "biases": biases})
expected = {"Conv2D_k2s2b", "ConvNextBlock", "FusedConv2D_k2s2b_QuantizedOutput",
            "FasterNetBlock<32, 1>", "FasterNetBlock<64, 2>",
            "FNB_CT2D_ADD<64, 2>", "FNB_CT2D_ADD<32, 1>", "CNB_CT2D<8>"}
if len(calls) != 14 or {call["operator"] for call in calls} != expected:
    raise RuntimeError("Selected source call inventory changed")
report = {"scope": "FSR 4.0.2 c5e34b4 v07 INT8 Quality passes_1080",
          "calls": calls, "padding_calls": source.count("ResetPaddingSeparate("),
          "operator_includes": re.findall(r'#include "(ml2code_runtime/operators/[^\"]+)"', source)}
out = root / "build/reports/m10/source-inventory.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"14 calls resolved; all called biases FP16; {report['padding_calls']} padding passes; {out}")
