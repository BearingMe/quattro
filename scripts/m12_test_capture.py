#!/usr/bin/env python3
"""Regression checks for capture provenance/ownership; never launches the GPU."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import m12_diff_snapshots as capture


class CaptureIntegrityTests(unittest.TestCase):
    def setUp(self):
        scratch = capture.ROOT / "build/temp/m12-capture-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.directory = Path(self.temporary.name)
        self.ref = self.directory / "ref"
        self.ref.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def manifests(self):
        for ordinal, name in enumerate(["run1", "run2", "run3", "control1", "control2", "control3"]):
            directory = self.ref / name
            directory.mkdir()
            manifest = {"implementationSha256": {"exe": "same-executable"},
                        "inputFnv1aLog": ["same-input"], "inputRecipe": "same-recipe",
                        "mode": "predicated derivative control" if ordinal >= 3 else "unmodified reference"}
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def run_main(self, args=()):
        with mock.patch.object(capture, "REF", self.ref), \
             mock.patch("sys.argv", ["m12_diff_snapshots.py", *args]), \
             contextlib.redirect_stdout(io.StringIO()):
            capture.main()

    def test_missing_manifest_fails_before_comparison(self):
        with mock.patch.object(capture, "compare") as compare:
            with self.assertRaisesRegex(ValueError, "missing capture identity"):
                self.run_main()
            compare.assert_not_called()

    def test_different_implementation_fails_before_comparison(self):
        self.manifests()
        path = self.ref / "control3/manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["implementationSha256"]["exe"] = "rebuilt-between-runs"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with mock.patch.object(capture, "compare") as compare:
            with self.assertRaisesRegex(ValueError, "identities changed between captures"):
                self.run_main()
            compare.assert_not_called()

    def test_different_input_fails_before_comparison(self):
        self.manifests()
        path = self.ref / "run2/manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["inputFnv1aLog"] = ["different-input"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with mock.patch.object(capture, "compare") as compare:
            with self.assertRaisesRegex(ValueError, "input identities changed"):
                self.run_main()
            compare.assert_not_called()

    def test_matching_provenance_reaches_comparison(self):
        self.manifests()
        with mock.patch.object(capture, "compare", side_effect=RuntimeError("comparison reached")):
            with self.assertRaisesRegex(RuntimeError, "comparison reached"):
                self.run_main()

    def test_losing_collector_cannot_overwrite_winner_log(self):
        def competing_writer(*args, **kwargs):
            directory = self.ref / "race-run1"
            directory.mkdir()
            (directory / "run.log").write_text("winner evidence", encoding="utf-8")
            return capture.subprocess.CompletedProcess(args[0], 1, stdout="loser failure")

        with mock.patch.object(capture.subprocess, "run", side_effect=competing_writer):
            with self.assertRaisesRegex(RuntimeError, "oracle failed"):
                self.run_main(("--collect", "--prefix", "race-"))
        self.assertEqual((self.ref / "race-run1/run.log").read_text(encoding="utf-8"), "winner evidence")
        failures = list(self.ref.glob("race-run1-*.failed.log"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].read_text(encoding="utf-8"), "loser failure")


if __name__ == "__main__":
    unittest.main()
