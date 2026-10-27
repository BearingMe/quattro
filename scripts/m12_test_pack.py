#!/usr/bin/env python3
"""Offline M12 packer regression checks; temporary artifacts stay in build/."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import m12_pack_model as pack


class ModelPackingTests(unittest.TestCase):
    def setUp(self):
        self.original_out = pack.OUT
        self.original_model = pack.MODEL
        scratch = pack.ROOT / "build/temp/m12-pack-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)
        pack.OUT = self.root / "reports"
        pack.OUT.mkdir()
        self.graph = json.loads((self.original_out / "graph.json").read_text(encoding="utf-8"))
        self.write_graph()

    def tearDown(self):
        pack.OUT = self.original_out
        pack.MODEL = self.original_model
        self.temporary.cleanup()

    def write_graph(self):
        (pack.OUT / "graph.json").write_text(json.dumps(self.graph), encoding="utf-8")

    def run_pack(self):
        with contextlib.redirect_stdout(io.StringIO()):
            pack.main()

    def test_exact_deterministic_payload(self):
        self.run_pack()
        before = {path.name: path.read_bytes() for path in pack.OUT.iterdir()}
        self.run_pack()
        self.assertEqual(before, {path.name: path.read_bytes() for path in pack.OUT.iterdir()})
        data = (pack.OUT / "weights.bin").read_bytes()
        manifest = json.loads((pack.OUT / "weights.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data), 136848)
        self.assertEqual(len(manifest["weights"]), 78)
        self.assertEqual(len(manifest["arrays"]), 67)
        self.assertEqual(data[:89216], (pack.MODEL / "initializers.bin").read_bytes())
        for array in manifest["arrays"]:
            encoded = data[array["offset"]:array["offset"] + array["bytes"]]
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), array["sha256"])
        self.assertEqual([len(v) for v in manifest["operatorScalars"]],
                         [0, 4, 4, 2, 5, 3, 2, 5, 5, 6, 5, 4, 4, 5])
        # Standard FNV-1a vectors: the runtime integrity check must use the
        # correct 64-bit offset basis (the older oracle logger did not).
        self.assertEqual(pack.fnv1a(b""), 0xCBF29CE484222325)
        self.assertEqual(pack.fnv1a(b"a"), 0xAF63DC4C8601EC8C)

    def test_wrong_source_identity(self):
        self.graph["model"]["sha256"] = "0" * 64
        self.write_graph()
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.run_pack()
        self.assertFalse((pack.OUT / "weights.bin").exists())

    def test_wrong_pass_order(self):
        self.graph["passes"][0]["index"] = 1
        self.write_graph()
        with self.assertRaisesRegex(ValueError, "pass order"):
            self.run_pack()

    def test_out_of_range_weight(self):
        self.graph["passes"][0]["tensors"][1]["storageByteOffset"] = 1 << 30
        self.write_graph()
        with self.assertRaisesRegex(ValueError, "exceeds backing storage"):
            self.run_pack()

    def test_invalid_scale(self):
        self.graph["passes"][0]["tensors"][1]["quantizationScale"] = -1
        self.write_graph()
        with self.assertRaisesRegex(ValueError, "invalid weight scale"):
            self.run_pack()

    def test_corrupt_initializer(self):
        pack.MODEL = self.root / "model"
        pack.MODEL.mkdir()
        for name in ("passes_1080.hlsl", "initializers.bin"):
            data = bytearray((self.original_model / name).read_bytes())
            if name == "initializers.bin":
                data[-1] ^= 1
            (pack.MODEL / name).write_bytes(data)
        with self.assertRaisesRegex(ValueError, "initializer hash mismatch"):
            self.run_pack()


if __name__ == "__main__":
    unittest.main()
