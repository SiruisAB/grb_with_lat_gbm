from __future__ import annotations

import queue
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pandas as pd


class ProjectRegressionTests(unittest.TestCase):
    def test_plot_style_is_preserved_in_run_overrides(self) -> None:
        from grb_project.config import GRBProjectConfig, run_overrides_from_config

        style = {
            "diagnostic_width_in": 6.0,
            "legend_position": "upper left",
            "lat_plot_bins": 6,
        }
        overrides = run_overrides_from_config(GRBProjectConfig(plot_style=style))

        self.assertIsNotNone(overrides)
        self.assertEqual(overrides.plot_style, style)

    def test_legacy_legend_position_updates_both_figure_types(self) -> None:
        from grb_project.publication_style import normalize_plot_style

        style = normalize_plot_style({"legend_position": "lower right"})

        self.assertEqual(style["diagnostic_legend_position"], "lower right")
        self.assertEqual(style["spectrum_legend_position"], "lower right")

    def test_counts_plot_uses_about_five_lat_bins(self) -> None:
        from grb_project.bayesian_fit import _counts_plot_min_rates

        class Plugin:
            def __init__(self, rates):
                self.expected_model_rate = rates

        lat = Plugin([1.0, 2.0, 3.0, 4.0])
        gbm = Plugin([10.0])
        analysis = mock.Mock(data_list={"LAT": lat, "n7": gbm})

        rates = _counts_plot_min_rates(analysis, lat, "bn123", lat_target_bins=5)

        self.assertAlmostEqual(rates[0], 2.0)
        self.assertEqual(rates[1], 1.0)

    def test_counts_plot_handles_extremely_small_lat_model_rate(self) -> None:
        from grb_project.bayesian_fit import _rate_for_target_plot_bins

        plugin = mock.Mock(expected_model_rate=[1.6232825494985804e-164])
        min_rate = _rate_for_target_plot_bins(plugin, target_bins=5)

        self.assertGreaterEqual(min_rate, 0.0)
        self.assertLess(min_rate, sum(plugin.expected_model_rate))

    def test_runtime_intervals_prefer_run_overrides(self) -> None:
        from grb_project.config import GRBRunOverrides
        from grb_project.project import _resolve_runtime_intervals

        active, background = _resolve_runtime_intervals(
            catalog_row=pd.Series(dtype=object),
            special_cfg={"active_interval": "1-2", "background_interval": "-5--1,5-10"},
            project_config=None,
            run_overrides=GRBRunOverrides(
                lightcurve_active_interval="3-4",
                lightcurve_background_intervals=("-20--10", "100-120"),
            ),
        )

        self.assertEqual(active, "3-4")
        self.assertEqual(background, "-20--10,100-120")

    def test_effective_mode_downgrades_when_lat_plugin_is_missing(self) -> None:
        from grb_project.project import _effective_fit_mode

        self.assertEqual(_effective_fit_mode("gbm+lat", None), "gbm")
        self.assertEqual(_effective_fit_mode("gbm+lat", object()), "gbm+lat")
        self.assertEqual(_effective_fit_mode("gbm", None), "gbm")

    def test_run_single_lat_mode_does_not_call_gbm_analysis(self) -> None:
        from grb_project import project

        catalog = pd.DataFrame(
            [{"bnname": "bn123", "gcn_name": "GRB123", "t90_start": 0.0, "t90": 2.0, "ra": 1.0, "dec": 2.0}]
        ).set_index("bnname", drop=False)
        lat_result = {
            "lat_plugin": object(),
            "analysis_segments": [{"tstart": 0.0, "tstop": 2.0, "tag": "analysis_full"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir, self._project_mocks(catalog), mock.patch.object(
            project, "_run_lat_analysis", return_value=lat_result
        ) as run_lat, mock.patch.object(project, "_run_gbm_analysis") as run_gbm:
            summary = project.run_single_analysis(
                target_grbs=["bn123"], analysis_mode="lat", result_root=tmpdir
            )

        run_lat.assert_called_once()
        run_gbm.assert_not_called()
        self.assertEqual(summary.iloc[0]["analysis_status"], "completed")
        self.assertEqual(summary.iloc[0]["analysis_mode"], "lat")

    def test_gbm_lat_mode_does_not_pre_run_unused_lat_analysis(self) -> None:
        from grb_project import project

        catalog = pd.DataFrame(
            [{"bnname": "bn123", "gcn_name": "GRB123", "t90_start": 0.0, "t90": 2.0, "ra": 1.0, "dec": 2.0}]
        ).set_index("bnname", drop=False)

        with tempfile.TemporaryDirectory() as tmpdir, self._project_mocks(catalog), mock.patch.object(
            project, "_run_lat_analysis"
        ) as run_lat, mock.patch.object(project, "_run_gbm_analysis", return_value=[]) as run_gbm:
            project.run_single_analysis(
                target_grbs=["bn123"], analysis_mode="gbm+lat", result_root=tmpdir
            )

        run_lat.assert_not_called()
        run_gbm.assert_called_once()

    def test_string_target_is_treated_as_one_target(self) -> None:
        from grb_project.project import _normalize_targets

        self.assertEqual(_normalize_targets("bn123"), ["bn123"])
        self.assertEqual(_normalize_targets(["bn123", "bn456"]), ["bn123", "bn456"])

    def test_run_override_false_has_priority_over_project_config(self) -> None:
        from grb_project.config import GRBProjectConfig, GRBRunOverrides
        from grb_project.project import _configured_value

        self.assertFalse(
            _configured_value(
                GRBRunOverrides(plot_joint_lightcurve=False),
                GRBProjectConfig(plot_joint_lightcurve=True),
                "plot_joint_lightcurve",
            )
        )

    def test_parallel_model_settings_default_to_two_workers(self) -> None:
        from grb_project.config import GRBProjectConfig
        from grb_project.project import _parallel_model_settings

        self.assertEqual(
            _parallel_model_settings(GRBProjectConfig(), None),
            (True, 2),
        )

    def test_parallel_model_settings_can_be_disabled_per_run(self) -> None:
        from grb_project.config import GRBProjectConfig, GRBRunOverrides
        from grb_project.project import _parallel_model_settings

        self.assertEqual(
            _parallel_model_settings(
                GRBProjectConfig(parallel_models=True, model_workers=4),
                GRBRunOverrides(parallel_models=False, model_workers=3),
            ),
            (False, 3),
        )

    @staticmethod
    def _project_mocks(catalog: pd.DataFrame):
        stack = ExitStack()
        stack.enter_context(mock.patch("grb_project.io_utils.read_catalog", return_value=catalog))
        stack.enter_context(
            mock.patch(
                "grb_project.gbm_core._determine_time_interval_and_position",
                return_value=(0.0, 2.0, 1.0, 2.0),
            )
        )
        stack.enter_context(
            mock.patch(
                "grb_project.gbm_core.resolve_active_interval_from_special_and_catalog",
                return_value="0-2",
            )
        )
        stack.enter_context(
            mock.patch(
                "grb_project.gbm_core._build_lat_analysis_segments",
                return_value=[{"tstart": 0.0, "tstop": 2.0, "tag": "analysis_full"}],
            )
        )
        stack.enter_context(mock.patch("grb_project.special_bursts._load_special_burst_config", return_value={}))
        for name in (
            "_append_time_bin_info",
            "_plot_model_comparison",
            "_save_all_models_per_grb",
            "_save_time_bin_analysis",
        ):
            stack.enter_context(mock.patch(f"grb_project.summary_export.{name}"))
        stack.enter_context(mock.patch("grb_project.summary_export._compute_time_bin_analysis", return_value=pd.DataFrame()))
        return stack


class WebRegressionTests(unittest.TestCase):
    def test_target_defaults_include_widget_time_state(self) -> None:
        from grb_project.web_app import _target_defaults

        defaults = _target_defaults(
            pd.Series({"gcn_name": "GRB123", "t90_start": -1.0, "t90": 4.0, "ra": 1.0, "dec": 2.0}),
            None,
            "bn123",
        )

        self.assertEqual(defaults["t0"], 0.0)
        self.assertEqual(defaults["t1"], 3.0)

    def test_failed_summary_is_not_reported_as_success(self) -> None:
        from grb_project.web_app import _analysis_succeeded

        failed = pd.DataFrame([{"analysis_status": "failed"}])
        completed = pd.DataFrame([{"analysis_status": "completed"}])
        self.assertFalse(_analysis_succeeded(failed))
        self.assertTrue(_analysis_succeeded(completed))

    def test_single_result_dir_matches_backend_layout(self) -> None:
        from grb_project.web_app import _single_result_dir

        self.assertEqual(
            _single_result_dir("/tmp/results", "GRB123"),
            Path("/tmp/results/GRB123"),
        )


class PipelineRegressionTests(unittest.TestCase):
    def test_session_log_is_created_and_pipeline_returns_summary(self) -> None:
        from grb_project import pipeline
        from grb_project.logging_utils import log
        from grb_project.session import session

        expected = pd.DataFrame([{"analysis_status": "completed"}])

        def fake_run(**_kwargs):
            log("pipeline-test")
            return expected

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "grb_project.project.run_single_analysis", side_effect=fake_run
        ):
            actual = pipeline.main(result_root=tmpdir, session_log=True)
            log_path = Path(tmpdir) / "analysis_session.log"
            self.assertTrue(log_path.exists())
            self.assertIn("pipeline-test", log_path.read_text(encoding="utf-8"))

        self.assertIs(actual, expected)
        self.assertIsNone(session.log_file_handle)


class LatWorkerRegressionTests(unittest.TestCase):
    def test_worker_exit_without_result_fails_immediately(self) -> None:
        from grb_project.process_utils import wait_for_worker_result

        proc = mock.MagicMock()
        proc.exitcode = 9
        result_queue = mock.MagicMock()
        result_queue.get.side_effect = queue.Empty

        with self.assertRaisesRegex(RuntimeError, "exited with code 9"):
            wait_for_worker_result(proc, result_queue, timeout_s=3600.0)

        proc.terminate.assert_not_called()

    def test_worker_timeout_terminates_process(self) -> None:
        from grb_project.process_utils import wait_for_worker_result

        proc = mock.MagicMock()
        proc.exitcode = None
        proc.is_alive.return_value = True
        result_queue = mock.MagicMock()
        result_queue.get.side_effect = queue.Empty

        with self.assertRaises(TimeoutError):
            wait_for_worker_result(proc, result_queue, timeout_s=0.01)

        proc.terminate.assert_called_once()
        proc.join.assert_called()


class ReviewFixRegressionTests(unittest.TestCase):
    """2026 年代码审查中修复的问题所对应的回归测试。"""

    def test_fit_and_plot_models_use_the_same_pivot_energy(self) -> None:
        """拟合与出图必须共用同一 pivot 能量。

        modelbuild.build_model() 拟合出的归一化 K 是相对某个 piv 定义的；
        separate_spectr.discrete_spectr() 若用不同的 piv 重建模型，谱线和
        反卷积出的数据点会整体偏离 (piv_plot / piv_fit) ** (-index) 倍，
        而图上模型与数据仍然吻合，肉眼无法察觉。
        """
        import inspect
        import re

        from grb_project import modelbuild
        from grb_project.separate_spectr import discrete_spectr

        pattern = re.compile(r"piv\s*=\s*([0-9]+(?:\.[0-9]*)?[Ee][+-]?[0-9]+)")
        fit_pivots = {float(v) for v in pattern.findall(inspect.getsource(modelbuild))}
        plot_pivots = {
            float(v) for v in pattern.findall(inspect.getsource(discrete_spectr))
        }

        self.assertTrue(fit_pivots, "modelbuild 中未找到 piv= 字面量")
        self.assertTrue(plot_pivots, "discrete_spectr 中未找到 piv= 字面量")
        self.assertEqual(
            plot_pivots,
            fit_pivots,
            "出图使用的 pivot 能量集合与拟合不一致",
        )

    def test_multi_color_blackbody_is_finite_at_m_equals_minus_one(self) -> None:
        """多色黑体在 m=-1 处必须给出有限值。

        prefactor 的分子 (m+1) 与分母 (kT_max/kT_min)^(m+1)-1 在 m=-1 处同时
        为 0，旧实现直接相除得到 nan。m 的先验是 Uniform_prior(-2.5, 1)，
        采样点必然覆盖 m=-1，一旦命中该点整条似然就被污染。
        """
        import numpy as np

        from grb_project.modelbuild import MultiColorBlackBody

        mbb = MultiColorBlackBody()
        energies = np.array([10.0, 100.0, 1000.0])
        at_pole = mbb.evaluate(energies, 1e-6, 8.0, 100.0, -1.0)

        self.assertTrue(
            np.all(np.isfinite(at_pole)),
            f"m=-1 处出现非有限值: {at_pole}",
        )

        # 解析极限必须与两侧邻域连续衔接，否则说明极限式写错了。
        for delta in (1e-4, -1e-4):
            nearby = mbb.evaluate(energies, 1e-6, 8.0, 100.0, -1.0 + delta)
            np.testing.assert_allclose(at_pole, nearby, rtol=1e-3)

    def test_display_window_honours_pad_before_seconds(self) -> None:
        """光变展示窗口的前置留白必须真正使用 pad_before_s。

        原实现把前置留白写死成 1.0，形参 pad_before_s 从未参与计算，
        web 界面上的 display_window_pad_before_s 输入框调了也没有任何效果。
        """
        from grb_project.lightcurves import resolve_lightcurve_display_window

        # 默认值 1.0 必须与改动前的硬编码行为逐字一致。
        self.assertEqual(
            resolve_lightcurve_display_window("0.5-10.2", pad_after_s=0.0),
            (-1.0, 11.0),
        )

        # 非默认值必须生效：前置留白 5 秒，起点应为 floor(0.5) - 5 = -5.0。
        self.assertEqual(
            resolve_lightcurve_display_window("0.5-10.2", 5.0, 0.0),
            (-5.0, 11.0),
        )

        # 后置留白同样按传入值生效，且两者互不影响。
        self.assertEqual(
            resolve_lightcurve_display_window("0.5-10.2", 5.0, 3.0),
            (-5.0, 14.0),
        )

    def test_special_burst_segment_names_are_unique_per_burst(self) -> None:
        """同一个暴内的分段名必须唯一。

        分段名会直接当作中间产物的文件名前缀
        （lat_extended_three_ml.py:310 的 outfile=f"{grb_name}_{seg['tag']}"），
        重名会让后一段静默覆盖前一段的 LAT 事件与响应文件。
        GRB240118A 曾把 63.0-69.2 和 69.2-79.0 两段都命名为 seg4。
        """
        import collections

        import yaml

        from grb_project.lightcurves import SPECIAL_BURSTS_YAML

        with open(SPECIAL_BURSTS_YAML, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        bursts = payload["special_bursts"]
        self.assertTrue(bursts, "special_bursts.yaml 中没有任何暴")

        for burst in bursts:
            names = [seg.get("name") for seg in (burst.get("time_segments") or [])]
            duplicated = sorted(
                name for name, count in collections.Counter(names).items() if count > 1
            )
            self.assertEqual(
                duplicated,
                [],
                f"{burst.get('name')} 存在重名分段 {duplicated}: {names}",
            )

    def test_parallel_jobs_reclaim_a_worker_that_never_exits(self) -> None:
        """子进程交回结果后拒绝退出时，整批拟合不能挂死。

        原来的 _run_parallel_model_jobs 用不带 timeout 的 process.join()，
        子进程若卡在解释器退出阶段，父进程会带着已经取到的结果无限期等待。
        """
        from grb_project import project

        class _StuckProcess:
            """把结果放进队列后再也不退出的子进程。"""

            def __init__(self):
                self.exitcode = None
                self._alive = True
                self.join_timeouts = []
                self.terminated = False

            def start(self):
                pass

            def is_alive(self):
                return self._alive

            def join(self, timeout=None):
                if timeout is None:
                    raise AssertionError("join() 没有超时，卡死的子进程会挂住整批拟合")
                self.join_timeouts.append(timeout)

            def terminate(self):
                self.terminated = True
                self._alive = False

        class _FakeQueue:
            def __init__(self):
                self._items = [{"ok": True, "row": {"model": "band"}}]

            def get(self, timeout=None):
                if self._items:
                    return self._items.pop(0)
                raise queue.Empty

            def close(self):
                pass

            def join_thread(self):
                pass

        created = []

        class _FakeContext:
            def Queue(self, maxsize=0):
                return _FakeQueue()

            def Process(self, target=None, args=(), name=None):
                process = _StuckProcess()
                created.append(process)
                return process

        with mock.patch.object(
            project.mp, "get_all_start_methods", return_value=["fork"]
        ), mock.patch.object(project.mp, "get_context", return_value=_FakeContext()):
            outcomes = project._run_parallel_model_jobs([{"model_str": "band"}], 1)

        self.assertEqual(outcomes, [{"ok": True, "row": {"model": "band"}}])
        self.assertTrue(created[0].terminated, "卡死的子进程没有被回收")
        self.assertTrue(
            all(t is not None and t > 0 for t in created[0].join_timeouts),
            f"join 的超时值不合法: {created[0].join_timeouts}",
        )

    def test_web_keeps_every_background_window(self) -> None:
        """网页光变分支不能丢掉第三段及之后的本底窗。

        special_bursts.yaml 里 bn250313607 与 bn220921462 都用三段本底；
        谱拟合那条路径原样透传整串，网页却只取前两段，导致图上的本底与
        实际拟合所用的本底不一致。
        """
        from grb_project.web_app import _parse_background_intervals

        default = ("-20--5", "100-150")

        # 三段必须原样保留。
        self.assertEqual(
            _parse_background_intervals("-24--5,100-150,350-400", default),
            ("-24--5", "100-150", "350-400"),
        )

        # 两段及以下的既有行为逐字不变。
        self.assertEqual(
            _parse_background_intervals("-24--5,100-150", default),
            ("-24--5", "100-150"),
        )
        self.assertEqual(
            _parse_background_intervals("-24--5", default),
            ("-24--5", "100-150"),
        )
        self.assertEqual(_parse_background_intervals("", default), default)

    def test_yaml_background_windows_survive_the_web_lightcurve_path(self) -> None:
        """special_bursts.yaml 中的每一段本底都要能走完网页光变这条路。"""
        import yaml

        from grb_project.lightcurves import (
            SPECIAL_BURSTS_YAML,
            parse_background_interval_tuple,
        )
        from grb_project.web_app import _parse_background_intervals

        with open(SPECIAL_BURSTS_YAML, encoding="utf-8") as handle:
            bursts = yaml.safe_load(handle)["special_bursts"]

        checked = 0
        for burst in bursts:
            raw = burst.get("background_interval")
            if not raw:
                continue
            expected = parse_background_interval_tuple(str(raw))
            parsed = _parse_background_intervals(str(raw), ("-20--5", "100-150"))
            self.assertEqual(
                parse_background_interval_tuple(",".join(parsed)),
                expected,
                f"{burst.get('name')} 的本底窗在网页分支被改动: {raw}",
            )
            checked += 1

        self.assertGreater(checked, 0, "special_bursts.yaml 中没有本底窗可供检查")

    def test_lat_energy_grid_extension_is_conditional(self) -> None:
        """向 100 MeV 以上延伸的能量网格必须是有条件的。

        GBM-only 时 emax 就是 1e5，np.logspace(5.0, 5.0, 100) 会退化成
        100 个完全相同的点，其中 99 个被无条件拼进 xs。这些重复采样会
        原样写进 save_model_curve_data 落盘的模型曲线文件。
        """
        import ast
        import inspect

        from grb_project.separate_spectr import discrete_spectr

        tree = ast.parse(inspect.getsource(discrete_spectr))

        def assigns_xs1(node) -> bool:
            return isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "xs1" for t in node.targets
            )

        # xs1 的赋值必须出现在某个 if 的分支里，而不是函数体的直线代码中。
        guarded = any(
            any(assigns_xs1(inner) for inner in ast.walk(node))
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
        )
        found = any(assigns_xs1(node) for node in ast.walk(tree))

        self.assertTrue(found, "discrete_spectr 中未找到 xs1 的赋值")
        self.assertTrue(guarded, "LAT 能量网格的延伸是无条件的，GBM-only 会产生重复采样点")

        # 同时钉死退化条件本身：logspace(5.0, 5.0, N) 确实是 N 个重复点。
        import numpy as np

        degenerate = np.logspace(5.0, np.log10(1e5), 100)
        self.assertEqual(np.unique(degenerate).size, 1)

    def test_special_burst_backgrounds_live_only_in_yaml(self) -> None:
        """逐暴的特例本底窗只能有 special_bursts.yaml 一个来源。

        gbm_core._build_background_interval_string 里曾按 bnname 硬编码过三个
        暴的本底窗，与 YAML 各存一份；改了一处忘了另一处会静默用错本底。三份
        配置已迁入 YAML，这里同时钉住两件事：函数里不能再出现按 bnname 的
        特例分支，以及迁移进 YAML 的取值必须与迁移前逐字一致。
        """
        import ast
        import inspect

        import yaml

        from grb_project.gbm_core import _build_background_interval_string
        from grb_project.lightcurves import SPECIAL_BURSTS_YAML

        # 1) 函数只吃目录表那一行，不再按暴名分支。
        self.assertEqual(
            list(inspect.signature(_build_background_interval_string).parameters),
            ["row"],
        )
        tree = ast.parse(inspect.getsource(_build_background_interval_string))
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("bn")
        ]
        self.assertEqual(literals, [], "本底窗函数里又出现了按 bnname 的硬编码特例")

        # 2) 迁移前这三个暴实际拿到的字符串，逐字钉死在 YAML 里。
        migrated = {
            "bn221023862": "-130.0--10.0,100.0-200.0",
            "bn250313607": "-24--5,100-150,350-400",
            "bn220921462": "-23.960--2.080,75-100,140-160",
        }
        with open(SPECIAL_BURSTS_YAML, encoding="utf-8") as handle:
            bursts = yaml.safe_load(handle)["special_bursts"]
        by_bnname = {str(b.get("bnname")): b for b in bursts if isinstance(b, dict)}

        for bnname, expected in migrated.items():
            self.assertIn(bnname, by_bnname, f"{bnname} 的特例本底窗没有留在 YAML 里")
            self.assertEqual(
                str(by_bnname[bnname].get("background_interval")),
                expected,
                f"{bnname} 的本底窗与迁移前不一致",
            )

    def test_migrated_special_bursts_only_carry_a_background_window(self) -> None:
        """迁移进来的两个暴不能顺手带上时间分段或活动区间。

        它们原先只有本底窗一项特例配置；YAML 条目里一旦多出 time_segments 或
        active_interval，project 就会改走特殊分段分支，分析结果随之变化。
        """
        import yaml

        from grb_project.lightcurves import SPECIAL_BURSTS_YAML

        with open(SPECIAL_BURSTS_YAML, encoding="utf-8") as handle:
            bursts = yaml.safe_load(handle)["special_bursts"]
        by_bnname = {str(b.get("bnname")): b for b in bursts if isinstance(b, dict)}

        for bnname in ("bn221023862", "bn220921462"):
            burst = by_bnname[bnname]
            self.assertFalse(burst.get("time_segments"), f"{bnname} 多出了 time_segments")
            self.assertFalse(burst.get("active_interval"), f"{bnname} 多出了 active_interval")

    def test_migrated_special_bursts_keep_their_result_directory_name(self) -> None:
        """YAML 里的 name 必须等于目录表 gcn_name 去掉空格后的结果。

        project 命中特例后会用 special_cfg["name"] 覆盖 grb_name，而 grb_name
        决定结果目录名。两个新条目若写错 name，输出目录就会换地方。
        """
        import yaml

        from grb_project.lightcurves import SPECIAL_BURSTS_YAML

        with open(SPECIAL_BURSTS_YAML, encoding="utf-8") as handle:
            bursts = yaml.safe_load(handle)["special_bursts"]
        by_bnname = {str(b.get("bnname")): b for b in bursts if isinstance(b, dict)}

        # 取自 fermilat-grb.xls 的 GCN 表，即 grb_name 的默认来源。
        gcn_names = {"bn221023862": "GRB 221023A", "bn220921462": "GRB 220921A"}
        for bnname, gcn_name in gcn_names.items():
            self.assertEqual(str(by_bnname[bnname].get("name")), gcn_name.replace(" ", ""))


    def test_li_ma_significance_reduces_correctly_when_alpha_is_one(self) -> None:
        """alpha=1 时 Li&Ma 退化成一个可以独立写出的闭式，用它交叉核对。"""
        import math

        from grb_project.lat_extended_three_ml import _li_ma_significance

        for n_on, n_off in ((30, 10), (100, 60), (7, 3)):
            total = n_on + n_off
            expected = math.sqrt(
                2.0
                * (
                    n_on * math.log(2.0 * n_on / total)
                    + n_off * math.log(2.0 * n_off / total)
                )
            )
            self.assertAlmostEqual(_li_ma_significance(n_on, n_off, 1.0), expected, places=9)

        # 没有超出就没有显著度。
        self.assertAlmostEqual(_li_ma_significance(50, 50, 1.0), 0.0, places=9)
        # 亏损记为负，排序时才不会和"刚好为零"混在一起。
        self.assertLess(_li_ma_significance(10, 50, 1.0), 0.0)
        # 本底不变时超出越多越显著。
        self.assertLess(
            _li_ma_significance(60, 100, 0.2),
            _li_ma_significance(90, 100, 0.2),
        )
        # 退化输入不能抛异常。
        self.assertEqual(_li_ma_significance(0, 0, 0.5), 0.0)
        self.assertEqual(_li_ma_significance(5, 5, 0.0), 0.0)

    def test_on_off_significance_counts_events_by_angular_separation(self) -> None:
        """on/off 计数按到源的角距划分，alpha 是立体角之比。"""
        import math
        import tempfile

        import numpy as np
        from astropy.io import fits

        from grb_project.lat_extended_three_ml import _on_off_significance

        ra0, dec0 = 100.0, 0.0
        # 6 个事例落在源上（角距 0），4 个落在 5 度外。
        ra = np.array([ra0] * 6 + [ra0 + 5.0] * 4, dtype=float)
        dec = np.array([dec0] * 10, dtype=float)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "events.fits")
            hdu = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="RA", format="E", array=ra),
                    fits.Column(name="DEC", format="E", array=dec),
                ],
                name="EVENTS",
            )
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

            result = _on_off_significance(path, ra0, dec0, roi=12.0, on_radius=1.0)

        self.assertEqual(result["n_on"], 6)
        self.assertEqual(result["n_off"], 4)

        omega_on = 1.0 - math.cos(math.radians(1.0))
        omega_off = math.cos(math.radians(1.0)) - math.cos(math.radians(12.0))
        self.assertAlmostEqual(result["alpha"], round(omega_on / omega_off, 6), places=6)
        self.assertGreater(result["sigma"], 0.0)

    def test_on_off_significance_gives_up_without_an_off_annulus(self) -> None:
        """roi 不大于 on 区半径时圆环退化，本底无从估计，只能返回空值。

        扫描范围里的 roi=1 正好落在这一档，不能让它把整个扫描弄挂。
        """
        from grb_project.lat_extended_three_ml import _on_off_significance

        result = _on_off_significance("/does/not/exist.fits", 10.0, 20.0, roi=1.0, on_radius=1.0)
        self.assertEqual(set(result), {"n_on", "n_off", "alpha", "excess", "sigma"})
        self.assertTrue(all(value is None for value in result.values()))

        # 文件读不出来时同样只能返回空值，不能抛出去。
        broken = _on_off_significance("/does/not/exist.fits", 10.0, 20.0, roi=12.0)
        self.assertIsNone(broken["sigma"])

    def test_roi_zmax_selection_still_goes_by_event_count(self) -> None:
        """显著度只是记录用的对照列，判据必须仍然是 nEvents 最大。

        构造一组数据：事例数最大的是 (roi=3, zmax=100)，显著度最大的是
        (roi=9, zmax=104)。返回值必须是前者。
        """
        from grb_project import lat_extended_three_ml as module

        class FakeDataset:
            ra = 10.0
            dec = 20.0
            filt_file = "/tmp/does-not-matter.fits"

            def __init__(self) -> None:
                self.nEvents = 0
                self.calls: list[tuple[int, int]] = []

            def extract_events(self, roi, zmax, irf, thetamax, strategy="time"):
                self.calls.append((roi, zmax))
                self.nEvents = 500 if (roi, zmax) == (3, 100) else roi + zmax

        def fake_significance(filt_file, ra, dec, roi, on_radius=1.0):
            sigma = 99.0 if roi == 9 else float(roi)
            return {"n_on": 1, "n_off": 1, "alpha": 0.1, "excess": 0.9, "sigma": sigma}

        dataset = FakeDataset()
        with mock.patch.object(module, "_on_off_significance", fake_significance):
            roi, zmax, rows = module._scan_best_roi_zmax(dataset, "p8_transient020e", 180.0)

        self.assertEqual((roi, zmax), (3, 100))
        self.assertEqual(len(dataset.calls), 72)
        self.assertEqual(len(rows), 72)
        # 对照列确实落进了每一行。
        self.assertTrue(all("sigma" in row for row in rows))
        best_by_sigma = max(rows, key=lambda row: row["sigma"])
        self.assertEqual(best_by_sigma["roi"], 9)
        self.assertNotEqual((best_by_sigma["roi"], best_by_sigma["zmax"]), (roi, zmax))


    # ---- discrete_spectr 的模型构造分支 ----

    @staticmethod
    def _discrete_spectr_branches():
        """按模型名取出 discrete_spectr 里给 modelTotal 赋值的那些 if 分支。

        直接解析源码而不是把构造逻辑抄一遍：抄一遍就等于又多了一份重复定义，
        而重复定义正是这些分支出问题的根源。
        """
        import ast

        from grb_project import separate_spectr

        source = Path(separate_spectr.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        func = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "discrete_spectr"
        )

        branches = {}
        for node in func.body:
            if not isinstance(node, ast.If):
                continue
            assigns_total = any(
                isinstance(target, ast.Name) and target.id == "modelTotal"
                for stmt in ast.walk(node)
                if isinstance(stmt, ast.Assign)
                for target in stmt.targets
            )
            if not assigns_total:
                continue
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    branches[sub.value] = node
        return branches

    def test_every_web_selectable_model_can_be_plotted(self) -> None:
        """网页上能选的模型，discrete_spectr 必须都有构造分支。

        SBPL 与 band+mbb 曾经只在 modelbuild 里有定义、在 discrete_spectr 里没有，
        于是 modelTotal 未绑定，分离谱在 bayesian_fit 的 except 里被吞成一行警告：
        拟合跑完了、AIC/BIC 进了表，图却永远不存在。
        """
        from grb_project.web_app import SUPPORTED_MODELS

        covered = set(self._discrete_spectr_branches())
        missing = sorted(set(SUPPORTED_MODELS) - covered)
        self.assertEqual(missing, [], f"这些模型能选能拟合，却画不出分离谱: {missing}")

    def test_discrete_spectr_reproduces_build_model_for_every_model(self) -> None:
        """同一组参数下，出图用的模型必须和拟合用的模型是同一条谱。

        两处各写一遍模型定义，历史上已经在 comp 的 pivot 上错出过约 1000 倍
        （pivot 默认 1 keV vs 拟合时的 100 keV）。这里把每个分支单独执行一遍，
        再和 modelbuild.build_model 的输出逐点比。
        """
        import ast

        import numpy as np

        from grb_project import separate_spectr
        from grb_project.modelbuild import build_model
        from grb_project.web_app import SUPPORTED_MODELS

        branches = self._discrete_spectr_branches()
        energies = np.logspace(np.log10(8.0), 5.0, 60)

        for model_str in SUPPORTED_MODELS:
            with self.subTest(model=model_str):
                node = branches[model_str]
                reference = build_model(model_str)
                values = [float(par.value) for par in reference.free_parameters.values()]

                namespace = dict(vars(separate_spectr))
                namespace.update(
                    model_str=model_str,
                    parameter_values=values,
                    analysis_mode="gbm",
                )
                exec(  # noqa: S102 - 就是要执行源码里的那一段
                    compile(ast.Module(body=[node], type_ignores=[]), "<branch>", "exec"),
                    namespace,
                )

                expected = np.asarray(reference(energies), dtype=float)
                actual = np.asarray(namespace["modelTotal"](energies), dtype=float)
                np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_sbpl_reads_the_break_energy_not_beta(self) -> None:
        """SBPL 的自由参数顺序是 K, alpha, break_energy, beta。

        转折能量在第 2 位；原先几处写的是第 3 位，取到的是 beta。beta 是负数，
        np.log10(beta*0.1) 直接给出 nan，加密采样和 y 轴范围会一起废掉。
        """
        import ast

        import numpy as np

        from grb_project import separate_spectr
        from grb_project.modelbuild import build_model

        free_names = [name.split(".")[-1] for name in build_model("SBPL").free_parameters]
        self.assertEqual(free_names, ["K", "alpha", "break_energy", "beta"])

        source = Path(separate_spectr.__file__).read_text(encoding="utf-8")
        func = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "discrete_spectr"
        )

        # 取出所有出现在 np.log10(...) 里的 parameter_values[N]
        indices = set()
        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "log10"):
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "parameter_values"
                    and isinstance(sub.slice, ast.Constant)
                ):
                    indices.add(sub.slice.value)

        self.assertEqual(
            indices,
            {free_names.index("break_energy")},
            "np.log10 只应作用在 break_energy 上，取到 beta 会得到 nan",
        )
        # 反过来钉一下：真按 beta 取，结果确实是 nan。
        beta = float(list(build_model("SBPL").free_parameters.values())[3].value)
        self.assertLess(beta, 0.0)
        with np.errstate(invalid="ignore"):
            self.assertTrue(np.isnan(np.log10(beta * 0.1)))

    def test_sbpl_only_resamples_the_break_once(self) -> None:
        """转折能量附近的加密采样只能做一次。

        原先有两段逐字相同的加密采样先后执行，同样的 100 个点被塞进 xs 两遍，
        落盘的模型曲线文件也跟着多出 100 行重复采样。
        """
        import ast

        from grb_project import separate_spectr

        source = Path(separate_spectr.__file__).read_text(encoding="utf-8")
        assignments = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "xs_around_E0"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)



if __name__ == "__main__":
    unittest.main()
