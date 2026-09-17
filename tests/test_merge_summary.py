from __future__ import annotations

"""merge_summary 的去重与优先级测试。"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from grb_project.merge_summary import merge


def _write(path: Path, rows: list) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class MergeSummaryTests(unittest.TestCase):
    def _run(self, batch_rows, recon_rows):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            batch = _write(tmp / "batch.csv", batch_rows)
            recon = _write(tmp / "recon.csv", recon_rows)
            out = tmp / "out.csv"
            stats = merge(batch, recon, out)
            return stats, pd.read_csv(out)

    def test_recon_only_rows_are_added(self):
        stats, df = self._run(
            [{"bnname": "bn1", "model": "band", "Flux(erg/cm2/s)": 1.0}],
            [
                {"bnname": "bn1", "model": "band", "Flux(erg/cm2/s)": 9.9},
                {"bnname": "bn2", "model": "band", "Flux(erg/cm2/s)": 2.0},
            ],
        )
        self.assertEqual(stats["only_recon"], 1)
        self.assertEqual(stats["overlap_kept_batch"], 1)
        self.assertEqual(len(df), 2)
        self.assertEqual(set(df.bnname), {"bn1", "bn2"})

    def test_overlap_prefers_batch_row(self):
        """重复目标必须保留批次的值，不能被重建值覆盖。"""
        _, df = self._run(
            [{"bnname": "bn1", "model": "band", "Flux(erg/cm2/s)": 1.0}],
            [{"bnname": "bn1", "model": "band", "Flux(erg/cm2/s)": 9.9}],
        )
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(float(df.iloc[0]["Flux(erg/cm2/s)"]), 1.0)

    def test_schema_follows_batch_columns(self):
        """重建行多出来的列必须被丢掉，保持与批次 CSV 一致。"""
        _, df = self._run(
            [{"bnname": "bn1", "model": "band"}],
            [{"bnname": "bn2", "model": "band", "_reconstructed": True, "多余列": 1}],
        )
        self.assertNotIn("_reconstructed", df.columns)
        self.assertNotIn("多余列", df.columns)

    def test_model_is_part_of_the_key(self):
        """同一个暴的不同模型是两行，不能互相去重。"""
        stats, df = self._run(
            [{"bnname": "bn1", "model": "band", "Flux(erg/cm2/s)": 1.0}],
            [
                {"bnname": "bn1", "model": "band", "Flux(erg/cm2/s)": 9.9},
                {"bnname": "bn1", "model": "band+bb", "Flux(erg/cm2/s)": 3.0},
            ],
        )
        self.assertEqual(stats["only_recon"], 1)
        self.assertEqual(len(df), 2)

    def test_idempotent(self):
        """重复合并结果不变（cron 会反复触发）。"""
        rows_b = [{"bnname": "bn1", "model": "band", "Flux(erg/cm2/s)": 1.0}]
        rows_r = [
            {"bnname": "bn1", "model": "band", "Flux(erg/cm2/s)": 9.9},
            {"bnname": "bn2", "model": "band", "Flux(erg/cm2/s)": 2.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            batch = _write(tmp / "batch.csv", rows_b)
            recon = _write(tmp / "recon.csv", rows_r)
            out = tmp / "out.csv"
            merge(batch, recon, out)
            first = pd.read_csv(out)
            # 第二次以合并结果为输入
            merge(out, recon, out)
            second = pd.read_csv(out)
            pd.testing.assert_frame_equal(first, second)

    def test_no_part_file_left_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            batch = _write(tmp / "batch.csv", [{"bnname": "bn1", "model": "band"}])
            recon = _write(tmp / "recon.csv", [{"bnname": "bn2", "model": "band"}])
            out = tmp / "out.csv"
            merge(batch, recon, out)
            self.assertEqual([p.name for p in tmp.iterdir() if p.name.endswith(".part")], [])


if __name__ == "__main__":
    unittest.main()
