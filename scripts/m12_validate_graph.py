#!/usr/bin/env python3
"""Validate the fixed M12 logical tensor lifetimes and group-scale chain.

This is metadata validation, not proof that a shader respects its descriptors.
Physical overhangs of the shipped reference are deliberately not model tensors.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/reports/m12"
SCRATCH_BYTES = 20736000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_extent(tensor):
    shape = tensor["logicalSize"]
    strides = tensor["storageByteStrides"]
    require(len(shape) == 3 and len(strides) == 3, "expected 3D activation")
    require(all(isinstance(v, int) and v > 0 for v in shape + strides), "invalid activation geometry")
    item_bytes = 2 if tensor["dtype"] == "h" else 1
    require(strides == [shape[2] * item_bytes, shape[0] * shape[2] * item_bytes, item_bytes],
            "unexpected noncontiguous activation")
    offset = tensor["storageByteOffset"]
    require(isinstance(offset, int) and offset >= 0, "unresolved activation base offset")
    return offset, offset + shape[0] * shape[1] * shape[2] * item_bytes


def main():
    graph = json.loads((OUT / "graph.json").read_text(encoding="utf-8"))
    packed = json.loads((OUT / "weights.json").read_text(encoding="utf-8"))
    require(graph["model"]["sha256"] == packed["sourceSha256"], "metadata source identity mismatch")
    scalars = packed["operatorScalars"]
    expected_shapes = [(960, 540, 16)] * 3 + [(480, 270, 32)] * 3 + \
                      [(240, 135, 64)] * 3 + [(480, 270, 32)] * 2 + \
                      [(960, 540, 16)] * 2 + [(1920, 1080, 8)]
    # Only these fixed operators use first-two input group scale arguments.
    grouped = {4, 5, 7, 8, 9, 10, 11}
    grouped_output = {3: (0, 1), 4: (3, 4), 6: (0, 1), 7: (3, 4),
                      8: (3, 4), 9: (4, 5), 10: (3, 4)}
    live = []
    report = []
    require(len(graph["passes"]) == 14 and len(scalars) == 14, "wrong pass count")
    for index, entry in enumerate(graph["passes"]):
        reads = [tensor for tensor in entry["tensors"][:-1] if tensor["storage"] == "ScratchBuffer"]
        output = entry["tensors"][-1]
        require(tuple(output["logicalSize"]) == expected_shapes[index], f"pass{index}: wrong output shape")
        inputs = []
        for ordinal, tensor in enumerate(reads):
            begin, end = tensor_extent(tensor)
            matches = [item for item in live if item["begin"] == begin and item["end"] == end]
            require(len(matches) == 1, f"pass{index}: input lacks one live complete producer")
            producer = matches[0]
            require(tensor["logicalSize"] == producer["shape"], f"pass{index}: input shape mismatch")
            if ordinal == 0 and index in grouped:
                input_scales = scalars[index][:2]
            else:
                require("quantizationScale" in tensor, f"pass{index}: unaccounted input scale")
                input_scales = [tensor["quantizationScale"]] * 2
            require(input_scales == producer["scales"], f"pass{index}: scale chain mismatch")
            inputs.append({"producerPass": producer["pass"], "shape": tensor["logicalSize"],
                           "begin": begin, "end": end, "scales": input_scales})
        begin, end = tensor_extent(output)
        if output["storage"] == "ScratchBuffer":
            require(end <= SCRATCH_BYTES, f"pass{index}: logical output exceeds scratch")
            if index in grouped_output:
                output_scales = [scalars[index][i] for i in grouped_output[index]]
            else:
                require("quantizationScale" in output, f"pass{index}: missing output scale")
                output_scales = [output["quantizationScale"]] * 2
            # An overlapping write invalidates the ENTIRE older tensor; no
            # partial alias is accepted as a valid subsequent producer.
            live = [item for item in live if item["end"] <= begin or item["begin"] >= end]
            live.append({"pass": index, "shape": output["logicalSize"], "begin": begin,
                         "end": end, "scales": output_scales})
        else:
            require(index == 13 and output["storage"] == "buffer_fused_quantized_NHWC_output",
                    "unexpected external output")
            output_scales = []
        report.append({"pass": index, "inputs": inputs, "outputShape": output["logicalSize"],
                       "outputBegin": begin, "outputEnd": end, "outputScales": output_scales})
    (OUT / "tensor_validation.json").write_text(json.dumps({"scope": "fixed logical descriptors only",
        "scratchBytes": SCRATCH_BYTES, "passes": report}, indent=2) + "\n", encoding="utf-8")
    print("M12 metadata: all 14 logical output shapes, scratch lifetimes, main/skip scales PASS")


if __name__ == "__main__":
    main()
