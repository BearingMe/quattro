"""Small independent formula regressions for the offline M14 image metrics."""

import math
import unittest

from m14_compare_images import Metrics


class MetricsTests(unittest.TestCase):
    def test_exact(self):
        metric = Metrics()
        for value in (0.0, 0.05, 2.0):
            metric.add(value, value)
        result = metric.report()
        self.assertEqual(result["rmse"], 0)
        self.assertEqual(result["different_float_values"], 0)
        self.assertIsNone(result["psnr_peak_2_db"])

    def test_known_errors(self):
        metric = Metrics()
        metric.add(1, 0)
        metric.add(0, 2)
        result = metric.report()
        self.assertEqual(result["mae"], 1.5)
        self.assertEqual(result["sse"], 5)
        self.assertAlmostEqual(result["rmse"], math.sqrt(2.5))
        self.assertAlmostEqual(result["psnr_peak_2_db"], 10 * math.log10(4 / 2.5))
        self.assertEqual(result["max_abs"], 2)

    def test_no_hdr_clipping(self):
        metric = Metrics()
        metric.add(4, -2)
        self.assertEqual(metric.report()["max_abs"], 6)
        self.assertEqual(metric.report()["rmse"], 6)

    def test_nonfinite_rejected(self):
        for value in (math.inf, -math.inf, math.nan):
            for pair in ((0, value), (value, 0)):
                with self.assertRaises(ValueError):
                    Metrics().add(*pair)

    def test_empty_region(self):
        self.assertEqual(Metrics().report(), {"components": 0})


if __name__ == "__main__":
    unittest.main()
