#!/usr/bin/env python3
"""Pack pinned M12 constants without translating or executing reference HLSL.

Run m12_extract_graph.py first. All generated data/tables live under build/.
The payload is initializers.bin followed by the embedded uint arrays in pass
order, little-endian and dword aligned. No conversion or requantization occurs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/reports/m12"
MODEL = ROOT / (".references/sources/FSR-4.0.2-reference/Kits/FidelityFX/"
                "upscalers/fsr4/internal/shaders/fsr4_model_v07_i8_quality")
SOURCE_HASH = "ab6762596c4439b759557a048d9b13a907baab99c9bab57cd13c416935e99bac"
INITIALIZER_HASH = "6fce93853235bbf8742210fb2bfd8653b70c90c722af17f5a37d1a3390156356"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def f32(value: float) -> float:
    """Round to binary32, the precision HLSL float arithmetic uses."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fnv1a(data: bytes) -> int:
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def vector(value, rank: int) -> list[int]:
    result = [value] if isinstance(value, int) else list(value)
    require(len(result) == rank and all(isinstance(v, int) and v > 0 for v in result),
            f"invalid rank-{rank} dimension/stride: {value}")
    return result


def main() -> None:
    source = (MODEL / "passes_1080.hlsl").read_bytes()
    initializers = (MODEL / "initializers.bin").read_bytes()
    require(hashlib.sha256(source).hexdigest() == SOURCE_HASH, "model source hash mismatch")
    require(hashlib.sha256(initializers).hexdigest() == INITIALIZER_HASH,
            "initializer hash mismatch")
    graph = json.loads((OUT / "graph.json").read_text(encoding="utf-8"))
    require(graph["model"]["sha256"] == SOURCE_HASH, "stale graph source identity")
    require(graph["model"]["initializers"]["sha256"] == INITIALIZER_HASH,
            "stale graph initializer identity")
    require(len(graph["passes"]) == 14, "expected 14 passes")

    # Each array belongs to exactly one pass. Keep the storage padding verbatim.
    text = source.decode("utf-8")
    payload = bytearray(initializers)
    records, arrays, parameters = [], [], []
    for index, entry in enumerate(graph["passes"]):
        require(entry["index"] == index, "pass order mismatch")
        match = re.search(rf"^#ifdef MLSR_PASS_{index}\n(.*?)"
                          rf"^#endif // #ifdef MLSR_PASS_{index}\s*$", text, re.M | re.S)
        require(match is not None, f"pass {index}: missing source block")
        block = match.group(1)
        embedded = {}
        for array in re.finditer(r"static const uint (\w+)\[(\d+)\]\s*=\s*\{(.*?)\n\};",
                                 block, re.S):
            name, count, body = array.group(1), int(array.group(2)), array.group(3)
            body = re.sub(r"//[^\n]*", "", body)
            tokens = [v.strip() for v in body.split(",") if v.strip()]
            require(all(re.fullmatch(r"0x[0-9a-fA-F]+|[0-9]+", v) for v in tokens),
                    f"pass {index}: invalid word in {name}")
            words = [int(v, 16 if v.startswith("0x") else 10) for v in tokens]
            require(len(words) == count, f"pass {index}: array length mismatch: {name}")
            data = struct.pack(f"<{count}I", *words)
            require(len(payload) % 4 == 0, "unaligned payload")
            require(name not in embedded, f"duplicate array {name}")
            embedded[name] = (len(payload), len(data))
            arrays.append({"pass": index, "name": name, "offset": len(payload),
                           "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            payload.extend(data)
        require(len(embedded) == len(entry["embeddedConstants"]), "stale embedded table")

        # Numeric prefix arguments are intermediate/group quantization metadata,
        # NOT necessarily carried by a QuantizedTensor descriptor.
        call = re.search(rf"^\s+{entry['operator']}(?:<[^>]+>)?\(([^;]+)\);", block, re.M)
        require(call is not None, f"pass {index}: operator invocation missing")
        scalars = []
        for arg in call.group(1).split(","):
            arg = arg.strip()
            if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", arg):
                break
            number = float(arg)
            require(math.isfinite(number) and number > 0, "invalid operator scalar")
            scalars.append(number)
        parameters.append(scalars)

        for tensor in entry["tensors"]:
            storage = tensor["storage"]
            if storage in ("ScratchBuffer", "buffer_NHWC_inputs", "buffer_fused_quantized_NHWC_output"):
                continue
            require(storage is not None, f"missing storage: {tensor['name']}")
            rank = tensor["rank"]
            require(rank in (1, 3, 4), "unsupported weight tensor rank")
            logical = vector(tensor["logicalSize"], rank)
            physical = vector(tensor["storageSize"], rank)
            strides = vector(tensor["storageByteStrides"], rank)
            require(all(a <= b for a, b in zip(logical, physical)), "logical extent exceeds storage")
            require(tensor["dtype"] in ("h", "i8"), "unsupported weight dtype")
            item_bytes = 2 if tensor["dtype"] == "h" else 1
            extent = sum((dim - 1) * stride for dim, stride in zip(physical, strides)) + item_bytes
            offset = tensor["storageByteOffset"]
            require(isinstance(offset, int) and offset >= 0, "invalid weight offset")
            if storage == "InitializerBuffer":
                base, capacity = 0, len(initializers)
            else:
                declaration = re.search(rf"ConstantBufferStorage<\d+>\s+{re.escape(storage)}"
                                        r"\s*=\s*\{\s*(\w+)\s*\}", block)
                require(declaration is not None and declaration.group(1) in embedded,
                        f"unresolved embedded storage: {storage}")
                base, capacity = embedded[declaration.group(1)]
            require(offset + extent <= capacity, f"weight exceeds backing storage: {tensor['name']}")
            scale = tensor.get("quantizationScale", 1.0)
            require(isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0,
                    f"invalid weight scale: {tensor['name']}")
            records.append({"pass": index, "name": tensor["name"], "rank": rank,
                            "itemBytes": item_bytes, "offset": base + offset, "bytes": extent,
                            "logical": logical, "physical": physical, "strides": strides,
                            "scale": scale, "sourceStorage": storage})

    require(len(arrays) == 67 and len(records) == 78, "unexpected weight inventory")
    digest = hashlib.sha256(payload).hexdigest()
    header = ["// Generated by scripts/m12_pack_model.py; do not edit.", "#pragma once",
              '#include "inference/models/fsr4_weights.h"', "namespace inference::fsr4::generated {",
              f"inline constexpr size_t kPayloadBytes = {len(payload)};",
              f"inline constexpr uint64_t kPayloadFnv1a = 0x{fnv1a(payload):016x}ull;",
              f'inline constexpr char kPayloadSha256[] = "{digest}";',
              "inline constexpr WeightTensor kWeights[] = {"]
    for record in records:
        def initializer(key):
            return "{" + ", ".join(str(v) for v in record[key] + [0] * (4 - record["rank"])) + "}"
        header.append(f'    {{{record["pass"]}, "{record["name"]}", {record["rank"]}, '
                      f'{record["itemBytes"]}, {record["offset"]}, {record["bytes"]}, '
                      f'{initializer("logical")}, {initializer("physical")}, '
                      f'{initializer("strides")}, {record["scale"]:.17g}f}},')
    header.extend(["};", "inline constexpr PassScalars kScalars[] = {"])
    for scalars in parameters:
        values = ", ".join(f"{value:.17e}f" for value in scalars) or "0.0f"
        header.append(f"    {{{len(scalars)}, {{{values}}}}},")
    header.extend(["};", "} // namespace inference::fsr4::generated", ""])
    # Ensure integer-looking scale literals remain valid C++ float literals.
    contents = "\n".join(header).replace(", 1f}", ", 1.0f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weights.bin").write_bytes(payload)
    (OUT / "weights.generated.h").write_text(contents, encoding="utf-8")
    shader = ["// Generated fixed weight addresses; no reference operator code."]
    for index in range(14):
        for ordinal, value in enumerate(parameters[index]):
            shader.append(f"#define M12_P{index}_ARG{ordinal} {value:.17e}f")
        for name, tensor in (("INPUT", graph["passes"][index]["tensors"][0]),
                             ("OUTPUT", graph["passes"][index]["tensors"][-1])):
            if "quantizationScale" in tensor:
                shader.append(f"#define M12_P{index}_{name}_SCALE {tensor['quantizationScale']:.17e}f")
        records_of_pass = [r for r in records if r["pass"] == index]
        # The quantized downscale epilogue folds the output-scale reciprocal
        # into the weight*input scale and into the bias; the shipped ISA shows
        # `v_fmaak_f32 sGroup, acc, bias*rcp` with no trailing multiply. Emit
        # the two folded group scales so the reproduction can bake exactly the
        # bits the reference code generation produced. Derivation (all steps
        # rounded to binary32, matching float arithmetic):
        #   fl(fl(weightScale * inputScale) / quantFactor)
        if (graph["passes"][index]["operator"] == "FusedConv2D_k2s2b_QuantizedOutput"
                and len(parameters[index]) >= 2 and records_of_pass):
            input_scale = f32(graph["passes"][index]["tensors"][0]["quantizationScale"])
            weight_scale = f32(records_of_pass[0]["scale"])
            product = f32(weight_scale * input_scale)
            for group in range(2):
                folded = f32(product / f32(parameters[index][group]))
                shader.append(f"#define M12_P{index}_EPI_SCALE{group} {folded:.17e}f")
        for ordinal, record in enumerate(records_of_pass):
            prefix = f"M12_P{index}_W{ordinal}"
            shader.append(f"#define {prefix}_OFFSET {record['offset']}u")
            shader.append(f"#define {prefix}_SCALE {record['scale']:.17e}f")
            for axis, value in enumerate(record["strides"]):
                shader.append(f"#define {prefix}_STRIDE{axis} {value}u")
            # The shipped CNB biases are compile-time FP16 data. Preserve that
            # placement for code-generation/rounding experiments, while the
            # checked runtime payload still contains their exact original bits.
            if record["rank"] == 1 and record["itemBytes"] == 2:
                begin = record["offset"]
                count = record["physical"][0] // 2
                words = struct.unpack_from(f"<{count}I", payload, begin)
                shader.append(f"static const uint {prefix}_WORDS[{count}] = {{" +
                              ", ".join(f"0x{word:08x}u" for word in words) + "};")
    (OUT / "weights.generated.hlsli").write_text("\n".join(shader) + "\n", encoding="utf-8")
    (OUT / "weights.json").write_text(json.dumps({"sourceSha256": SOURCE_HASH,
        "initializerSha256": INITIALIZER_HASH, "payloadSha256": digest,
        "payloadBytes": len(payload), "arrays": arrays, "weights": records,
        "operatorScalars": parameters}, indent=2) + "\n", encoding="utf-8")
    print(f"packed {len(records)} weight tensors, {len(payload)} B, sha256 {digest}")


if __name__ == "__main__":
    main()
