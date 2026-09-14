from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from grb_project.joint_selection import (
    JointSelectionCriteria,
    load_joint_target_table,
    select_joint_targets,
)


def _row(bnname, grb_name, **overrides):
    row = dict(
        bnname=bnname,
        grb_name=grb_name,
        gbm_held=True,
        lat_downloaded=True,
        lat_grb_valid=True,
        requires_lle_data=False,
        lat_only_lle=False,
        flgc_lat_ts=30.0,
        trigger_met=600000000.0,
        ra=10.0,
        dec=20.0,
        T0=0.0,
        T1=100.0,
        window_source="flgc_tl",
    )
    row.update(overrides)
    return row


class JointSelectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.data_dir = root / "GBM_data"
        self.lat_root = root / "Extended_data_ex"

        # 就绪暴：GBM 有 TTE、LAT 有事件 + SC
        self.ready = [
            ("bn081001234", "GRB081001A", 30.0),
            ("bn130502345", "GRB130502A", 50.0),
            ("bn170809123", "GRB170809A", 9.0),
            ("bn220101215", "GRB220101A", 25.0),
        ]
        # 数据目录齐备、靠标志位/参数被排除的暴（让判据能走到对应分支）
        gbm_and_lat = [
            ("bn150127398", "GRB150127A"),  # LLE-only，供 include_lle_only 测试
            ("bn110518860", "GRB110518A"),  # T1 缺失，供参数完备性测试
        ]
        for bn, grb in gbm_and_lat:
            gbm = self.data_dir / bn
            gbm.mkdir(parents=True, exist_ok=True)
            (gbm / f"glg_tte_n0_{bn}_v00.fit").write_bytes(b"")
            lat = self.lat_root / grb
            lat.mkdir(parents=True, exist_ok=True)
            (lat / "Lxxx_EV00.fits").write_bytes(b"")
            (lat / "Lxxx_SC00.fits").write_bytes(b"")

        for bn, grb, _ts in self.ready:
            gbm = self.data_dir / bn
            gbm.mkdir(parents=True, exist_ok=True)
            (gbm / f"glg_tte_n0_{bn}_v00.fit").write_bytes(b"")
            lat = self.lat_root / grb
            lat.mkdir(parents=True, exist_ok=True)
            (lat / "Lxxx_EV00.fits").write_bytes(b"")
            (lat / "Lxxx_SC00.fits").write_bytes(b"")

        # LAT 目录在、GBM 目录不在（测磁盘复核与 gbm 标志位两个分支）
        for bn, grb in [("bn210822388", "GRB210822A"), ("bn160702516", "GRB160702A")]:
            lat = self.lat_root / grb
            lat.mkdir(parents=True, exist_ok=True)
            (lat / "Lyyy_EV00.fits").write_bytes(b"")
            (lat / "Lyyy_SC00.fits").write_bytes(b"")
        # GBM 目录在、LAT 目录不在（测 LAT 磁盘复核）
        gbm = self.data_dir / "bn120911268"
        gbm.mkdir(parents=True, exist_ok=True)
        (gbm / "glg_tte_n0_bn120911268_v00.fit").write_bytes(b"")

        self.table = pd.DataFrame(
            [_row(bn, grb, flgc_lat_ts=ts) for bn, grb, ts in self.ready]
            + [
                _row("bn140330180", "GRB140330A", lat_grb_valid=False),
                _row("bn150127398", "GRB150127A", requires_lle_data=True, lat_only_lle=True),
                _row("bn160702516", "GRB160702A", gbm_held=False),
                _row("bn210822388", "GRB210822A"),  # 标志位齐但磁盘无 GBM 目录
                _row("bn120911268", "GRB120911B"),  # LAT 目录缺失（磁盘复核）
                _row("bn110518860", "GRB110518A", T1=float("nan")),
                _row("badname", "GRB999999A"),
            ]
        )

    def tearDown(self):
        self._tmp.cleanup()

    def select(self, **kwargs):
        criteria = JointSelectionCriteria(**kwargs)
        return select_joint_targets(
            criteria, table=self.table, data_dir=self.data_dir, lat_root=self.lat_root
        )

    def test_ready_targets_selected_and_sorted(self):
        result = self.select()
        self.assertEqual(
            result.bnnames,
            ["bn081001234", "bn130502345", "bn170809123", "bn220101215"],
        )
        frame = result.to_frame()
        self.assertEqual(
            list(frame.columns),
            [
                "bnname", "grb_name", "year", "trigger_met", "T0", "T1",
                "ra", "dec", "window_source", "lat_ts",
            ],
        )
        self.assertEqual(result.per_year_counts(), {2008: 1, 2013: 1, 2017: 1, 2022: 1})

    def test_exclusion_reasons_grouped(self):
        result = self.select()
        reasons = {reason: set(names) for reason, names in result.excluded.items()}
        self.assertIn("bn140330180", reasons.get("LAT 对应体已撤回", set()))
        self.assertIn("bn150127398", reasons.get("LLE-only 探测", set()))
        self.assertIn("bn160702516", reasons.get("本地无 GBM 数据", set()))
        self.assertIn("bn210822388", reasons.get("GBM 目录缺失或无 TTE", set()))
        self.assertIn("bn120911268", reasons.get("LAT 目录缺失或不完整", set()))
        self.assertIn("bn110518860", reasons.get("参数不完备（MET/RA/Dec/T0/T1）", set()))
        self.assertIn("badname", reasons.get("bnname 非法", set()))

    def test_year_range_filters(self):
        result = self.select(year_from=2013, year_to=2017)
        self.assertEqual(result.bnnames, ["bn130502345", "bn170809123"])
        self.assertIn("bn081001234", result.excluded.get("年份 < 2013", []))
        self.assertIn("bn220101215", result.excluded.get("年份 > 2017", []))

    def test_min_lat_ts_threshold(self):
        result = self.select(min_lat_ts=30.0)
        self.assertIn("bn170809123", result.excluded.get("LAT TS < 30", []))
        self.assertNotIn("bn170809123", result.bnnames)
        self.assertIn("bn081001234", result.bnnames)  # 等于阈值保留
        result5 = self.select(min_lat_ts=5.0)
        self.assertIn("bn170809123", result5.bnnames)  # 阈值降到 5 后放行

    def test_include_lle_only(self):
        result = self.select(include_lle_only=True)
        self.assertIn("bn150127398", result.bnnames)

    def test_no_verify_on_disk_trusts_flags(self):
        result = self.select(verify_on_disk=False)
        self.assertIn("bn210822388", result.bnnames)  # 标志位为真即选入
        self.assertIn("bn120911268", result.bnnames)

    def test_lat_mode_skips_gbm_requirement(self):
        result = self.select(analysis_mode="lat")
        self.assertIn("bn160702516", result.bnnames)  # gbm_held=False 也可选
        self.assertIn("bn210822388", result.bnnames)  # 磁盘无 GBM 目录也可选

    def test_gbm_only_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.select(analysis_mode="gbm")

    def test_load_table_reports_missing_columns(self):
        bad_csv = Path(self._tmp.name) / "bad_table.csv"
        pd.DataFrame([{"bnname": "bn081001234"}]).to_csv(bad_csv, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_joint_target_table(str(bad_csv))
        self.assertIn("grb_name", str(ctx.exception))

    def test_lat_mode_still_verifies_lat_dir(self):
        result = self.select(analysis_mode="lat")
        # bn120911268 标志位齐全，但 LAT 目录在磁盘上不存在，lat 模式仍要排除
        self.assertNotIn("bn120911268", result.bnnames)


if __name__ == "__main__":
    unittest.main()
