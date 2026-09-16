from __future__ import annotations

"""增量写汇总 CSV 的回归测试。

背景：summary_results.csv 原本只在整批跑完时写一次，中途停止（手动停止 /
SIGTERM / 崩溃）会把已完成目标的汇总全部丢掉。实测 46 个目标跑 8 小时后
SIGTERM 停止，CSV 完全是空的。
"""

import os
import tempfile
import unittest
from pathlib import Path

from grb_project.project import _atomic_write_summary


class AtomicWriteSummaryTests(unittest.TestCase):
    def test_writes_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "summary.csv"
            _atomic_write_summary([{"bnname": "bn1", "model": "band"}], path)
            self.assertTrue(path.exists())
            self.assertIn("bn1", path.read_text(encoding="utf-8"))

    def test_creates_missing_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a" / "b" / "c" / "summary.csv"
            _atomic_write_summary([{"bnname": "bn1"}], path)
            self.assertTrue(path.exists())

    def test_empty_rows_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _atomic_write_summary([], path)
            self.assertFalse(path.exists())

    def test_repeated_writes_grow_the_file(self):
        """第二个目标完成后，CSV 必须同时含两条，而不是只剩最后一条。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _atomic_write_summary([{"bnname": "bn1"}], path)
            _atomic_write_summary([{"bnname": "bn1"}, {"bnname": "bn2"}], path)
            content = path.read_text(encoding="utf-8")
            self.assertIn("bn1", content)
            self.assertIn("bn2", content)

    def test_leaves_no_part_file_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _atomic_write_summary([{"bnname": "bn1"}], path)
            leftovers = [p.name for p in Path(tmp).iterdir() if p.name.endswith(".part")]
            self.assertEqual(leftovers, [])

    def test_overwrite_does_not_truncate_to_garbage(self):
        """旧文件被完整替换，不会残留上一次的长内容。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _atomic_write_summary([{"bnname": f"bn{i}"} for i in range(50)], path)
            _atomic_write_summary([{"bnname": "only"}], path)
            content = path.read_text(encoding="utf-8")
            self.assertIn("only", content)
            self.assertNotIn("bn49", content)

    def test_uses_replace_so_readers_never_see_a_half_file(self):
        """断言实现走的是 os.replace（原子替换），而不是原地写。"""
        import inspect
        from grb_project import project as project_module

        source = inspect.getsource(project_module._atomic_write_summary)
        self.assertIn("os.replace", source)
        self.assertNotIn("path.write_text", source)
        self.assertEqual(os.path.basename(inspect.getsourcefile(project_module)), "project.py")


if __name__ == "__main__":
    unittest.main()
