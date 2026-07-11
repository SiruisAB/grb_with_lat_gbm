from __future__ import annotations

import unittest


class GbmCoreTests(unittest.TestCase):
    def test_normalize_background_interval_accepts_list_input(self) -> None:
        from grb_project.gbm_core import _normalize_background_interval

        self.assertEqual(
            _normalize_background_interval(["-20--10", "100-120"]),
            ("-20--10", "100-120"),
        )

    def test_normalize_background_interval_accepts_string_input(self) -> None:
        from grb_project.gbm_core import _normalize_background_interval

        self.assertEqual(
            _normalize_background_interval("-20--10,100-120"),
            ("-20--10", "100-120"),
        )

    def test_default_time_bins_keep_one_integrated_bin(self) -> None:
        from grb_project.gbm_core import _determine_time_bins

        edges, count, duration = _determine_time_bins(0.0, 100.0, None)

        self.assertEqual(edges.tolist(), [0.0, 100.0])
        self.assertEqual(count, 1)
        self.assertEqual(duration, 100.0)

    def test_explicit_fixed_time_bins_are_honored(self) -> None:
        from grb_project.gbm_core import _determine_time_bins

        edges, count, duration = _determine_time_bins(0.0, 8.0, 4)

        self.assertEqual(edges.tolist(), [0.0, 2.0, 4.0, 6.0, 8.0])
        self.assertEqual(count, 4)
        self.assertEqual(duration, 8.0)


if __name__ == "__main__":
    unittest.main()
