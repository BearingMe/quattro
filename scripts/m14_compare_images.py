"""Compare identical-input experimental 4.0.2 and official 4.1.2 raw RGB captures.

Cross-version image differences are NOT an isolated INT8 quantization error.
No image clipping, exposure fitting, or post-hoc alignment is used for metrics.
"""

import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib

from m13_analyze_sequence import chunk


W, H = 1920, 1080
ROOT = Path(__file__).resolve().parents[1]


class Metrics:
    def __init__(self):
        self.count = self.different = 0
        self.absolute = self.squared = self.maximum = 0.0

    def add(self, a, b):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("nonfinite RGB in comparison; no silent masking")
        delta = abs(a - b)
        self.count += 1
        self.different += a != b
        self.absolute += delta
        self.squared += delta * delta
        self.maximum = max(self.maximum, delta)

    def report(self):
        if not self.count:
            return {"components": 0}
        mse = self.squared / self.count
        return {"components": self.count, "different_float_values": self.different,
                "mae": self.absolute / self.count, "rmse": math.sqrt(mse),
                "max_abs": self.maximum, "sse": self.squared,
                "psnr_peak_2_db": 10 * math.log10(4 / mse) if mse else None}


def load_rgb(path, official):
    raw = path.read_bytes()
    stride, start = (4, 0) if official else (8, 4)
    if len(raw) != W * H * stride * 4:
        raise ValueError(f"incorrect raw image extent: {path}")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    rgb = array.array("f", (values[p * stride + start + c] for p in range(W * H) for c in range(3)))
    return rgb, hashlib.sha256(raw).hexdigest()


def save_png(path, data):
    image = b"\x89PNG\r\n\x1a\n"
    image += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    image += chunk(b"IDAT", zlib.compress(data)) + chunk(b"IEND", b"")
    path.write_bytes(image)


def display(v):
    v = max(0, v)
    return min(255, max(0, round((v / (1 + v)) ** (1 / 2.2) * 255)))


def temporal(current, previous, shift):
    score = Metrics()
    start = 2 if shift else 0
    for y in range(H):
        for x in range(start, W):
            p = (y * W + x) * 3
            for c in range(3):
                old = previous[p + c] if not shift else (
                    previous[(y * W + x - 2) * 3 + c] + previous[(y * W + x - 1) * 3 + c]) * 0.5
                score.add(current[p + c], old)
    return score.report()


def compare(own, reference, frame, moving):
    scores = {name: Metrics() for name in ("all", "edge", "interior", "invalid_motion_patch")}
    own_png, ref_png, heat = bytearray(), bytearray(), bytearray()
    overshoot = {"experimental_below_0_05": 0, "official_below_0_05": 0,
                 "experimental_above_2": 0, "official_above_2": 0}
    shift = 1 if moving and frame == 1 else 0
    for y in range(H):
        own_png.append(0)
        ref_png.append(0)
        heat.append(0)
        ly = (y + 0.5) / 1.5 - 0.5
        for x in range(W):
            lx = (x + 0.5) / 1.5 - 0.5 - shift
            foreground = 480 <= lx < 800 and 240 <= ly < 480
            rect_edge = ((abs(lx - 480) <= 2 or abs(lx - 800) <= 2) and 238 <= ly <= 482 or
                         (abs(ly - 240) <= 2 or abs(ly - 480) <= 2) and 478 <= lx <= 802)
            checker_edge = min(lx % 16, 16 - lx % 16, ly % 16, 16 - ly % 16) <= 2
            edge = rect_edge or (not foreground and checker_edge)
            patch = moving and frame == 1 and 932 <= x < 988 and 512 <= y < 568
            for c in range(3):
                a, b = own[(y * W + x) * 3 + c], reference[(y * W + x) * 3 + c]
                scores["all"].add(a, b)
                scores["edge" if edge else "interior"].add(a, b)
                if patch:
                    scores["invalid_motion_patch"].add(a, b)
                overshoot["experimental_below_0_05"] += a < 0.05 - 1e-6
                overshoot["official_below_0_05"] += b < 0.05 - 1e-6
                overshoot["experimental_above_2"] += a > 2 + 1e-6
                overshoot["official_above_2"] += b > 2 + 1e-6
                own_png.append(display(a)); ref_png.append(display(b))
                heat.append(min(255, round(abs(a - b) * 16 * 255)))
    result = {name: score.report() for name, score in scores.items()}
    result["edge_sse_fraction"] = scores["edge"].squared / scores["all"].squared if scores["all"].squared else 0
    result["input_domain_excursions"] = overshoot
    return result, (own_png, ref_png, heat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experimental", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("--sequence", choices=("static", "moving-reset"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to((ROOT / "build").resolve()):
        parser.error("generated reports must be under build/")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"experimental_model": "FSR 4.0.2 v07 i8 quality, software temporal adapter",
              "official_provider": "4.1.2, independently verified by capture log",
              "cross_version": True, "sequence": args.sequence, "peak_for_psnr": 2,
              "metrics_space": "unclipped linear RGB; output alpha ignored",
              "edge_mask": "known checker/rectangle boundaries, +/-2 LR pixels (=3 HR pixels)",
              "heatmap": "absolute RGB difference x16, display clipping only",
              "temporal_metric": "static raw frame difference; moving uses known 1.5 HR pixel translation and software bilinear previous sample; reset transitions separate",
              "causality_limit": "Different closed model/temporal code: cannot attribute to INT8, sampler precision, or a single operator from image differences alone.",
              "frames": []}
    previous = None
    first = None
    for frame in range(3):
        own_input = (args.experimental / f"input-frame{frame}.bin").read_bytes()
        ref_input = (args.official / f"input-frame{frame}.bin").read_bytes()
        if own_input != ref_input:
            raise ValueError("input bytes differ; comparison refused")
        own, own_hash = load_rgb(args.experimental / f"frame{frame}.bin", False)
        reference, ref_hash = load_rgb(args.official / f"frame{frame}.rgba32f", True)
        result, images = compare(own, reference, frame, args.sequence == "moving-reset")
        result.update({"frame": frame, "input_sha256": hashlib.sha256(own_input).hexdigest(),
                       "experimental_packed_sha256": own_hash, "official_rgba32f_sha256": ref_hash})
        for label, image in zip(("experimental", "official", "difference-x16"), images):
            save_png(args.output_dir / f"frame{frame}-{label}.png", image)
        if frame == 0:
            first = (own, reference)
        else:
            reset = struct.unpack_from("<I", own_input)[0] != 0
            old = first if reset else previous
            shift = args.sequence == "moving-reset" and not reset
            result["temporal"] = {"transition": "reset-to-first" if reset else "continued",
                                  "experimental": temporal(own, old[0], shift),
                                  "official": temporal(reference, old[1], shift)}
        previous = (own, reference)
        report["frames"].append(result)
    report["comparable_inputs_verified"] = True
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"M14 cross-version comparison complete: {args.output_dir}")
    for frame in report["frames"]:
        print(f"frame{frame['frame']}: MAE {frame['all']['mae']:.9g}, RMSE {frame['all']['rmse']:.9g}, "
              f"max {frame['all']['max_abs']:.9g}, PSNR {frame['all']['psnr_peak_2_db']}, "
              f"edge SSE {100*frame['edge_sse_fraction']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
