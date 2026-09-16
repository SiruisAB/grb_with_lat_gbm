from __future__ import annotations

"""单模型拟合超时的回归测试。

背景：dynesty 嵌套采样没有迭代上限（跑到 dlogz 才停），后验一旦退化就会
满核空转；而 _run_parallel_model_jobs 原本只判 process.is_alive()，
一个病态 bin 就能把整批目标无限期堵死（实测堵 3.5 小时、日志零输出）。
"""

import time
import unittest

from grb_project.config import GRBProjectConfig, GRBRunOverrides
from grb_project.project import (
    _DEFAULT_MODEL_FIT_TIMEOUT_S,
    _model_fit_timeout_s,
    _run_parallel_model_jobs,
)


def _hanging_worker(result_queue, fit_kwargs):
    """模拟陷在退化后验里的采样器：永不返回。"""
    while True:
        time.sleep(0.5)


def _instant_worker(result_queue, fit_kwargs):
    result_queue.put({"ok": True, "row": {"model": fit_kwargs.get("model_str")}})


class ModelFitTimeoutSettingTests(unittest.TestCase):
    def test_default_when_nothing_configured(self):
        self.assertEqual(
            _model_fit_timeout_s(None, None), _DEFAULT_MODEL_FIT_TIMEOUT_S
        )

    def test_override_wins_over_config(self):
        cfg = GRBProjectConfig(model_fit_timeout_s=120.0)
        ov = GRBRunOverrides(model_fit_timeout_s=60.0)
        self.assertEqual(_model_fit_timeout_s(cfg, ov), 60.0)

    def test_config_used_when_no_override(self):
        self.assertEqual(
            _model_fit_timeout_s(GRBProjectConfig(model_fit_timeout_s=90.0), None),
            90.0,
        )

    def test_non_positive_disables_timeout(self):
        for value in (0.0, -1.0):
            with self.subTest(value=value):
                self.assertIsNone(
                    _model_fit_timeout_s(
                        GRBProjectConfig(model_fit_timeout_s=value), None
                    )
                )

    def test_garbage_falls_back_to_default(self):
        self.assertEqual(
            _model_fit_timeout_s(
                GRBProjectConfig(model_fit_timeout_s="not-a-number"), None
            ),
            _DEFAULT_MODEL_FIT_TIMEOUT_S,
        )


class ParallelModelJobTimeoutTests(unittest.TestCase):
    def test_hanging_worker_is_killed_and_marked_failed(self):
        outcomes = _run_parallel_model_jobs(
            [{"model_str": "band"}],
            max_workers=1,
            timeout_s=2.0,
            worker_entry=_hanging_worker,
        )

        self.assertEqual(len(outcomes), 1)
        self.assertFalse(outcomes[0]["ok"])
        self.assertIn("2", outcomes[0]["error"])
        self.assertIn("超", outcomes[0]["error"])

    def test_timeout_only_affects_the_hanging_model(self):
        """并行批次里，健康模型必须原样返回结果，不被邻居拖累。"""
        outcomes = _run_parallel_model_jobs(
            [{"model_str": "band"}, {"model_str": "band+bb"}],
            max_workers=2,
            timeout_s=3.0,
            worker_entry=lambda q, kw: (
                _hanging_worker(q, kw)
                if kw["model_str"] == "band+bb"
                else _instant_worker(q, kw)
            ),
        )

        by_model = {row.get("model"): row for row in (o.get("row") for o in outcomes) if row}
        self.assertIn("band", by_model)
        failed = [o for o in outcomes if not o.get("ok")]
        self.assertEqual(len(failed), 1)
        self.assertIn("超", failed[0]["error"])

    def test_no_timeout_lets_worker_finish_normally(self):
        outcomes = _run_parallel_model_jobs(
            [{"model_str": "band"}],
            max_workers=1,
            timeout_s=None,
            worker_entry=_instant_worker,
        )
        self.assertTrue(outcomes[0]["ok"])
        self.assertEqual(outcomes[0]["row"]["model"], "band")


if __name__ == "__main__":
    unittest.main()
