from __future__ import annotations

import ast
import inspect
import unittest

from grb_project.separate_spectr import _plot_emax_kev, discrete_spectr


class PlotEmaxTests(unittest.TestCase):
    def test_mode_string_alone_decides_plot_emax(self) -> None:
        self.assertEqual(_plot_emax_kev("gbm+lat"), 1e8)
        self.assertEqual(_plot_emax_kev("lat"), 1e8)
        self.assertEqual(_plot_emax_kev("GBM+LAT"), 1e8)
        self.assertEqual(_plot_emax_kev("gbm"), 1e5)
        self.assertEqual(_plot_emax_kev(""), 1e5)
        self.assertEqual(_plot_emax_kev(None), 1e5)

    def test_emax_not_gated_on_lat_plugin(self) -> None:
        """请求 gbm+lat 时，即使 LAT 插件缺失（降级 bin），图也必须画到 1e8 keV。

        旧实现 `if "lat" in mode and lat is not None: emax = 1e8` 会把降级 bin
        的坐标压回 1e5；现在 emax 只由模式字符串决定。
        """
        tree = ast.parse(inspect.getsource(discrete_spectr))
        has_emax_assignment = any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "emax" for t in node.targets)
            for node in ast.walk(tree)
        )
        self.assertTrue(has_emax_assignment, "discrete_spectr 中未找到 emax 赋值")

        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            emax_inside = any(
                isinstance(inner, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "emax" for t in inner.targets)
                for inner in ast.walk(node)
            )
            if not emax_inside:
                continue
            for sub in ast.walk(node.test):
                if (
                    isinstance(sub, ast.Compare)
                    and isinstance(sub.left, ast.Name)
                    and sub.left.id == "lat"
                    and any(isinstance(c, ast.Constant) and c.value is None for c in sub.comparators)
                ):
                    self.fail("emax 的判定仍依赖 lat is not None，降级 bin 会被压回 1e5")


if __name__ == "__main__":
    unittest.main()
