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


if __name__ == "__main__":
    unittest.main()
