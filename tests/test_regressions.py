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
            bnname="bn123",
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


if __name__ == "__main__":
    unittest.main()
