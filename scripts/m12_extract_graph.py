#!/usr/bin/env python3
"""M12: extract the FSR 4 v07 INT8 Quality (1080p) execution graph.

Reads the pinned, read-only reference source and emits a machine-readable
description of every model pass: entry point, thread-group shape, dispatch
grid at 1920x1080, every tensor descriptor (dtype, logical/storage size,
byte strides, storage byte offset, quantization scale, backing buffer),
embedded constants, and the operator invocation.

Nothing is executed and nothing is written outside build/reports/m12/.

Usage:  python scripts/m12_extract_graph.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FSR4 = (ROOT / ".references/sources/FSR-4.0.2-reference/Kits/FidelityFX/upscalers/fsr4")
MODEL = FSR4 / "internal/shaders/fsr4_model_v07_i8_quality"
PASSES = MODEL / "passes_1080.hlsl"
OUT_DIR = ROOT / "build/reports/m12"

# Provider dispatch rules for the non-WMMA (v07-i8) path at 1920x1080.
# Source: dx12/ffx_provider_fsr4_dx12.cpp L1603-1700, L1900-1945.
UPSCALE = (1920, 1080)
ENC = [(UPSCALE[0] // (1 << (i + 1)), UPSCALE[1] // (1 << (i + 1))) for i in range(3)]


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


DISPATCH = {
    0: (ceil_div(UPSCALE[0], 16), ceil_div(UPSCALE[1], 16), 1),
    1: (ceil_div(ENC[0][0], 64), ENC[0][1], 1),
    2: (ceil_div(ENC[0][0], 64), ENC[0][1], 1),
    3: (ceil_div(ENC[1][0], 64), ENC[1][1], 1),
    4: (ceil_div(ENC[1][0], 64), ENC[1][1], 1),
    5: (ceil_div(ENC[1][0], 64), ENC[1][1], 1),
    6: (ceil_div(ENC[2][0], 64), ENC[2][1], 1),
    7: (ceil_div(ENC[2][0], 64), ENC[2][1], 1),
    8: (ceil_div(ENC[2][0], 64), ENC[2][1], 1),
    9: (ceil_div(ENC[2][0], 8), ceil_div(ENC[2][1], 8), 1),
    10: (ceil_div(ENC[1][0], 64), ENC[1][1], 1),
    11: (ceil_div(ENC[1][0], 64), ENC[1][1], 1),
    12: (ceil_div(ENC[0][0], 64), ENC[0][1], 1),
    13: (ceil_div(UPSCALE[0], 16), ceil_div(UPSCALE[1], 16), 1),
}

DTYPE_BYTES = {"i8": 1, "h": 2, "f": 4}


# --- tiny expression evaluator for the generated descriptors -----------------
def ev(expr: str, env: dict):
    """Evaluate a generated scalar expression (uint3(...), vars, +,-,*,/)."""
    expr = expr.strip().rstrip(",")
    m = re.fullmatch(r"u?int([34])\((.*)\)", expr)
    if m:
        parts = [p for p in m.group(2).split(",") if p.strip()]
        return tuple(ev(p, env) for p in parts)
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            value = env[node.id]
            if isinstance(value, (int, float)) or (isinstance(value, tuple) and
                    all(isinstance(item, (int, float)) for item in value)):
                return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError(f"unsupported generated expression: {expr}")
    return evaluate(ast.parse(expr, mode="eval").body)


def parse_consts(block: str) -> dict:
    env: dict = {}
    for m in re.finditer(r"^\s*const\s+(?:uint3|uint4|int3|int4|float|int|uint)\s+"
                         r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);", block, re.M):
        name, expr = m.group(1), m.group(2).strip()
        try:
            env[name] = ev(expr, env)
        except Exception:  # expressions depending on group ids are per-invocation
            env[name] = expr
    return env


def dtype_of(template: str) -> str:
    t = template.replace(" ", "")
    if "QuantizedTensor" in t or "Tensor3i8" in t or "Tensor4i8" in t or "Tensor1i8" in t:
        return "i8"
    if "Tensor3h" in t or "Tensor4h" in t or "Tensor1h" in t:
        return "h"
    if "Tensor3f" in t or "Tensor4f" in t or "Tensor1f" in t:
        return "f"
    return "?"


def rank_of(template: str) -> int:
    m = re.search(r"Tensor([1-4])", template)
    return int(m.group(1)) if m else 0


def buffer_of(template: str) -> str:
    if "ConstantBufferStorage" in template:
        return "embedded-const"
    return "?"


DESCRIPTOR_SHAPE_KEYS = ["logicalSize", "threadGroupSliceStart", "threadGroupSliceSize",
                         "storageSize", "storageByteStrides", "paddingBegin",
                         "paddingEnd"]


def split_fields(body: str) -> list[str]:
    fields, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0 and cur.strip() and not cur.strip().startswith("//"):
            fields.append(cur.strip())
            cur = ""
            continue
        cur += ch
    if cur.strip():
        fields.append(cur.strip())
    return fields


def parse_descriptor(template: str, name: str, body: str, env: dict,
                     storage_map: dict) -> dict:
    """Fields are positional in ML2Code output: 7 shape/group entries, then the
    storage byte offset, then (quantized tensors only) the scale, then storage."""
    body = re.sub(r"//[^\n]*", "", body)
    values = []
    storage = None
    for f in split_fields(body):
        f = f.strip()
        if f.startswith("storage_") or f in ("ScratchBuffer", "InitializerBuffer",
                                             "buffer_NHWC_inputs",
                                             "buffer_fused_quantized_NHWC_output"):
            storage = storage_map.get(f, f)
            values.append(f)
            continue
        try:
            values.append(ev(f, env))
        except Exception:
            values.append(f)
    desc = {"name": name, "template": template, "dtype": dtype_of(template),
            "rank": rank_of(template)}
    for i, key in enumerate(DESCRIPTOR_SHAPE_KEYS):
        if i < len(values) - 2:
            desc[key] = values[i]
    if len(values) >= 8:
        raw_offset = values[7]
        if isinstance(raw_offset, str):
            m = re.search(r"\+\s*(\d+)\s*$", raw_offset)
            desc["storageByteOffset"] = int(m.group(1)) if m else raw_offset
            desc["offsetIsGroupRelative"] = True
        else:
            desc["storageByteOffset"] = raw_offset
    if "Quantized" in template and len(values) >= 10:
        desc["quantizationScale"] = values[8]
    desc["storage"] = storage
    return desc


def parse_embedded(block: str, env: dict) -> list[dict]:
    out = []
    for m in re.finditer(r"static const uint (\w+)\[(\d+)\]\s*=\s*\{(.*?)\n\};", block, re.S):
        name, count, body = m.group(1), int(m.group(2)), m.group(3)
        body_no_comments = re.sub(r"//[^\n]*", "", body)
        words = re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", body_no_comments)
        if len(words) != count:
            raise ValueError(f"embedded array size mismatch: {name}")
        out.append({"name": name, "declaredDwords": count, "encodedDwords": len(words),
                    "bytes": count * 4})
    return out


def main() -> int:
    text = PASSES.read_text(encoding="utf-8", errors="replace")
    passes = []
    for idx in range(14):
        start = text.find(f"#ifdef MLSR_PASS_{idx}\n")
        if start < 0:
            print(f"error: pass {idx} not found", file=sys.stderr)
            return 1
        end = text.find(f"#ifdef MLSR_PASS_{idx}_POST", start)
        block = text[start:end if end > 0 else len(text)]
        env = parse_consts(block)
        storage_map = {m.group(1): m.group(2) for m in re.finditer(
            r"const\s+(?:RW)?BufferStorage\s+(\w+)\s*=\s*\{\s*(\w+)\s*\}", block)}
        entry = re.search(r"void (fsr4_model_v07_i8_pass\d+)\(", block).group(1)
        threads = tuple(int(v) for v in
                        re.search(r"\[numthreads\((\d+),\s*(\d+),\s*(\d+)\)\]", block).groups())
        includes = re.findall(r'#include "ml2code_runtime/operators/([^"]+)"', block)
        op_call = None
        for m in re.finditer(r"^\s{4}([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;]*?>)?\s*\(", block, re.M):
            if m.group(1) in ("ComputeShaderParams", "const", "return"):
                continue
            op_call = m.group(1)
        shapes = [ln.strip()[2:].strip() for ln in block.splitlines()
                  if ln.strip().startswith("// ") and "->" in ln]
        tensors = []
        for m in re.finditer(r"const\s+([A-Za-z0-9_]+)\s*<\s*([^;{=]*?)\s*>\s+"
                             r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{", block):
            template = m.group(1)
            if template in ("RWBufferStorage", "BufferStorage", "ConstantBufferStorage",
                            "ReadOnlyBufferStorage"):
                continue
            depth, i = 0, m.end() - 1
            while i < len(block):
                if block[i] == "{":
                    depth += 1
                elif block[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            body = block[m.end():i]
            tensors.append(parse_descriptor(template + "<" + m.group(2) + ">",
                                            m.group(3), body, env, storage_map))
        p = {
            "index": idx,
            "entry": entry,
            "numthreads": threads,
            "dispatch": DISPATCH[idx],
            "operator": op_call,
            "operatorHeaders": includes,
            "shapeComments": shapes,
            "embeddedConstants": parse_embedded(block, env),
            "tensors": tensors,
        }
        passes.append(p)

    graph = {
        "model": {
            "id": "fsr4_model_v07_i8_quality",
            "generated": "2025-02-26 12:40:27 by ML2Code from fsr4_model_v07_i8.onnx",
            "source": str(PASSES.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(PASSES.read_bytes()).hexdigest(),
            "initializers": {
                "file": str((MODEL / "initializers.bin").relative_to(ROOT)).replace("\\", "/"),
                "bytes": (MODEL / "initializers.bin").stat().st_size,
                "sha256": hashlib.sha256((MODEL / "initializers.bin").read_bytes()).hexdigest(),
            },
            "defines": dict(re.findall(r"^#define (\w+)\s+(\S+)$", text[:2000], re.M)),
        },
        "upscale": {"width": UPSCALE[0], "height": UPSCALE[1],
                    "encoderResolutions": ENC,
                    "dispatchPath": "non-WMMA (v07-i8; supportsWmma is env-gated: MLSR-WMMA)"},
        "passes": passes,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")

    lines = ["# FSR 4 v07 INT8 Quality @1080p — extracted graph", "",
             f"source `{graph['model']['source']}` sha256 `{graph['model']['sha256']}`",
             f"initializers {graph['model']['initializers']['bytes']} B sha256 "
             f"`{graph['model']['initializers']['sha256']}`", "",
             "| # | entry | threads | dispatch | operator | tensors | storage offsets | scales |",
             "|---|---|---|---|---|---|---|---|"]
    for p in passes:
        offs = sorted({str(t.get("storageByteOffset")) for t in p["tensors"]
                       if t.get("storage") == "ScratchBuffer" and "storageByteOffset" in t})
        scales = sorted({str(t.get("quantizationScale")) for t in p["tensors"]
                         if "quantizationScale" in t})
        lines.append(f"| {p['index']} | `{p['entry']}` | {p['numthreads']} | {p['dispatch']} | "
                     f"`{p['operator']}` | {len(p['tensors'])} | {', '.join(offs)} | "
                     f"{len(scales)} distinct |")
    (OUT_DIR / "graph.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'graph.json'} ({len(passes)} passes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
