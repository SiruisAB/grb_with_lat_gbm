from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from grb_project import lat_processing
from grb_project.lat_processing import _lookup_lat_target_info
from grb_project.session import session


class LookupLatTargetInfoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._old_xls = session.fermilat_grb_xls
        self._old_csv = session.joint_target_csv

        # 精选表：只含一个 2022 暴
        self.xls_path = root / "fermilat-grb.xls"
        pd.DataFrame(
            [
                {
                    "trigname": "bn220101215",
                    "gcn_name": "GRB 220101A",
                    "trigger_met": 662706616.0,
                    "ra,dec": "1.52,31.75",
                }
            ]
        ).to_excel(self.xls_path, sheet_name="GCN", index=False)

        # 交叉表：含 2022 暴（不同值，验证精选表优先）+ 一个 2008 暴
        self.csv_path = root / "lat_download_targets.csv"
        pd.DataFrame(
            [
                {
                    "bnname": "bn220101215",
                    "grb_name": "GRB220101A",
                    "trigger_met": 999999999.0,
                    "ra": 111.0,
                    "dec": 222.0,
                },
                {
                    "bnname": "bn080818945",
                    "grb_name": "GRB080818B",
                    "trigger_met": 240799667.0,
                    "ra": 317.6,
                    "dec": 44.4,
                },
            ]
        ).to_csv(self.csv_path, index=False)

        session.fermilat_grb_xls = str(self.xls_path)
        session.joint_target_csv = str(self.csv_path)

    def tearDown(self):
        session.fermilat_grb_xls = self._old_xls
        session.joint_target_csv = self._old_csv
        self._tmp.cleanup()

    def test_curated_xls_takes_priority(self):
        info = _lookup_lat_target_info("bn220101215")
        self.assertEqual(info["grb_name"], "GRB220101A")
        self.assertEqual(info["trigger_met"], 662706616.0)
        self.assertEqual(info["ra"], 1.52)
        self.assertEqual(info["dec"], 31.75)

    def test_falls_back_to_joint_target_csv(self):
        info = _lookup_lat_target_info("bn080818945")
        self.assertIsNotNone(info)
        self.assertEqual(info["grb_name"], "GRB080818B")
        self.assertEqual(info["trigger_met"], 240799667.0)
        self.assertAlmostEqual(info["ra"], 317.6)
        self.assertAlmostEqual(info["dec"], 44.4)

    def test_missing_everywhere_returns_none(self):
        self.assertIsNone(_lookup_lat_target_info("bn999999999"))

    def test_unreadable_xls_falls_back_to_csv(self):
        session.fermilat_grb_xls = str(Path(self._tmp.name) / "nope.xls")
        info = _lookup_lat_target_info("bn080818945")
        self.assertIsNotNone(info)
        self.assertEqual(info["grb_name"], "GRB080818B")

    def test_csv_row_with_nan_coords_returns_none(self):
        bad_csv = Path(self._tmp.name) / "bad.csv"
        pd.DataFrame(
            [{"bnname": "bn080818945", "grb_name": "GRB080818B",
              "trigger_met": 240799667.0, "ra": float("nan"), "dec": 44.4}]
        ).to_csv(bad_csv, index=False)
        session.joint_target_csv = str(bad_csv)
        session.fermilat_grb_xls = str(Path(self._tmp.name) / "nope.xls")
        self.assertIsNone(_lookup_lat_target_info("bn080818945"))


if __name__ == "__main__":
    unittest.main()
