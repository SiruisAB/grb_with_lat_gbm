from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock
import unittest

import pandas as pd


class ProjectFlowTests(unittest.TestCase):
    @staticmethod
    def _catalog_frame() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "trigger_name": "bn123",
                    "bnname": "bn123",
                    "gcn_name": "GRB123",
                    "t90_start": 0.5,
                    "t90": 8.0,
                    "ra": 12.3,
                    "dec": -45.6,
                    "back_interval_low_start": -20.0,
                    "back_interval_low_stop": -10.0,
                    "back_interval_high_start": 100.0,
                    "back_interval_high_stop": 120.0,
                }
            ]
        ).set_index("bnname", drop=False)

    FAKE_GBM_ROW = {
        "grb_name": "GRB123",
        "bnname": "bn123",
        "model": "band",
        "analysis_mode": "gbm+lat",
        "AIC": 1.0,
        "BIC": 2.0,
        "Flux(erg/cm2/s)": 3.0,
        "FTot(erg/cm2/s)": 3.0,
        "Fluence(erg/cm2)": 4.0,
        "FBB(erg/cm2/s)": 0.0,
        "log_marginal_likelihood": -5.0,
        "duration": 8.0,
        "num_time_bins": 1,
        "bin_start_time": 0.5,
        "bin_end_time": 8.5,
        "bin_duration": 8.0,
        "analysis_status": "completed",
        "analysis_note": "GBM+LAT 联合拟合已恢复",
        "result_dir": "/tmp/results/GRB123",
        "source_dir": "/tmp/data/GRB123/bn123",
        "segment_tag": "analysis_full",
        "detectors": "n0,n1,b0",
    }

    def test_build_result_metadata_includes_lightcurve_fields(self) -> None:
        from grb_project.project import _build_result_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir) / "GRB123"
            result_dir.mkdir(parents=True)
            payload = _build_result_metadata(
                grb_name="GRB123",
                bnname="bn123",
                analysis_mode="gbm+lat",
                result_dir=result_dir,
                t0=0.5,
                t1=8.5,
                ra=12.3,
                dec=-45.6,
                background_interval="-20--10,100-120",
                active_interval="0.5-8.5",
                time_segments=[{"tstart": 0.5, "tstop": 8.5, "tag": "analysis_full"}],
                models=["band", "comp"],
                special_yaml="special_bursts.yaml",
                special_burst_name="GRB123",
                lat_three_ml_full=False,
                plot_joint_lightcurve=True,
                gbm_start=-1.0,
                gbm_stop=12.0,
                gbm_display_pad_before_s=1.0,
                gbm_display_pad_after_s=2.0,
                lightcurve_include_lat=True,
                lightcurve_lat_prob_threshold=0.9,
                lightcurve_nai_bands_kev=((8.0, 50.0), (50.0, 300.0)),
                lightcurve_bgo_band_kev=(300.0, 38000.0),
                lightcurve_active_interval="0.5-8.5",
                lightcurve_background_intervals=("-20--10", "100-120"),
                lightcurve_path=str(result_dir / "GRB123_lightcurve.png"),
            )
            saved = json.loads((result_dir / "run_metadata.json").read_text(encoding="utf-8"))

        assert payload["plot_joint_lightcurve"] is True
        assert payload["gbm_start"] == -1.0
        assert payload["gbm_stop"] == 12.0
        assert payload["lightcurve_include_lat"] is True
        assert payload["lightcurve_lat_prob_threshold"] == 0.9
        assert payload["lightcurve_path"] == str(result_dir / "GRB123_lightcurve.png")
        assert saved == payload

    def test_plot_lightcurve_draws_special_time_segments(self):
        import matplotlib.pyplot as plt

        import types

        three_ml = types.ModuleType("threeML")
        three_ml.TimeSeriesBuilder = mock.MagicMock()
        three_ml.OGIPLike = mock.MagicMock()
        three_ml.DataList = mock.MagicMock()
        three_ml.silence_warnings = mock.MagicMock()
        three_ml.__path__ = []
        three_ml_config_pkg = types.ModuleType("threeML.config")
        three_ml_config_pkg.__path__ = []
        three_ml_config_mod = types.ModuleType("threeML.config.config")
        three_ml_config_mod.threeML_config = mock.MagicMock(
            time_series=mock.MagicMock(
                light_curve_color="C0",
                background_selection_color="C1",
                background_color="C2",
            )
        )
        three_ml_io_pkg = types.ModuleType("threeML.io")
        three_ml_io_pkg.__path__ = []
        three_ml_io_plotting_pkg = types.ModuleType("threeML.io.plotting")
        three_ml_io_plotting_pkg.__path__ = []
        three_ml_step_plot_mod = types.ModuleType("threeML.io.plotting.step_plot")
        three_ml_step_plot_mod.step_plot = mock.MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            "sys.modules",
            {
                "threeML": three_ml,
                "threeML.config": three_ml_config_pkg,
                "threeML.config.config": three_ml_config_mod,
                "threeML.io": three_ml_io_pkg,
                "threeML.io.plotting": three_ml_io_plotting_pkg,
                "threeML.io.plotting.step_plot": three_ml_step_plot_mod,
                "gt_apps": mock.MagicMock(),
            },
        ):
            import grb_project.runtime_env as runtime_env
            runtime_env._initialized = True
            import grb_project.lightcurves as lightcurves

            fake_ax = mock.MagicMock()
            fake_fig = mock.MagicMock()
            with mock.patch.object(lightcurves, "ensure_analysis_runtime"), mock.patch.object(
                lightcurves,
                "read_trigger_met_and_grb_name",
                return_value=(123.0, "GRB123"),
            ), mock.patch.object(lightcurves, "select_gbm_detectors", return_value=(("n0", "n1"), "b0")), mock.patch.object(
                lightcurves,
                "detectors_for_lightcurve",
                return_value=(("n0", "n1"), "b0"),
            ), mock.patch.object(
                lightcurves,
                "resolve_gbm_tte_rsp",
                return_value=(Path("/tmp/fake_tte.fit"), Path("/tmp/fake_rsp.rsp")),
            ), mock.patch.object(lightcurves, "kev_band_to_echan", return_value=(0, 1)), mock.patch.object(
                lightcurves,
                "plot_mean_lightcurve_on_ax",
            ), mock.patch.object(lightcurves, "shade_active_interval") as mock_shade_active, mock.patch.object(
                lightcurves,
                "shade_time_segments",
            ) as mock_shade_segments, mock.patch.object(
                lightcurves,
                "resolve_spectral_time_bins_for_lightcurve",
                return_value=None,
            ) as mock_resolve_bins, mock.patch.object(
                plt,
                "subplots",
                return_value=(fake_fig, [fake_ax, fake_ax, fake_ax]),
            ), mock.patch.object(plt, "close"):
                lightcurves.plot_gbm_lat_lightcurve_figure(
                    bnname="bn123",
                    grb_name="GRB123",
                    trigger_met=123.0,
                    data_dir=tmpdir,
                    gbm_start=0.0,
                    gbm_stop=6.0,
                    include_lat=False,
                    special_time_segments=[
                        {"name": "seg1", "start": 1.0, "stop": 2.0},
                        {"name": "seg2", "start": 2.0, "stop": 3.0},
                    ],
                    out_path=Path(tmpdir) / "out.png",
                )

        mock_shade_segments.assert_called_once()
        assert mock_shade_segments.call_args.args[1] == [
            {"name": "seg1", "start": 1.0, "stop": 2.0},
            {"name": "seg2", "start": 2.0, "stop": 3.0},
        ]
        mock_shade_active.assert_called_once()
        assert mock_resolve_bins.call_count == 1

    def test_shade_time_segments_draws_visible_spans(self):
        import matplotlib.pyplot as plt

        import types

        three_ml = types.ModuleType("threeML")
        three_ml.TimeSeriesBuilder = mock.MagicMock()
        three_ml.OGIPLike = mock.MagicMock()
        three_ml.DataList = mock.MagicMock()
        three_ml.silence_warnings = mock.MagicMock()
        three_ml.__path__ = []
        three_ml_config_pkg = types.ModuleType("threeML.config")
        three_ml_config_pkg.__path__ = []
        three_ml_config_mod = types.ModuleType("threeML.config.config")
        three_ml_config_mod.threeML_config = mock.MagicMock(time_series=mock.MagicMock())
        three_ml_io_pkg = types.ModuleType("threeML.io")
        three_ml_io_pkg.__path__ = []
        three_ml_io_plotting_pkg = types.ModuleType("threeML.io.plotting")
        three_ml_io_plotting_pkg.__path__ = []
        three_ml_step_plot_mod = types.ModuleType("threeML.io.plotting.step_plot")
        three_ml_step_plot_mod.step_plot = mock.MagicMock()

        with mock.patch.dict(
            "sys.modules",
            {
                "threeML": three_ml,
                "threeML.config": three_ml_config_pkg,
                "threeML.config.config": three_ml_config_mod,
                "threeML.io": three_ml_io_pkg,
                "threeML.io.plotting": three_ml_io_plotting_pkg,
                "threeML.io.plotting.step_plot": three_ml_step_plot_mod,
                "gt_apps": mock.MagicMock(),
            },
        ):
            import grb_project.runtime_env as runtime_env
            runtime_env._initialized = True
            import grb_project.lightcurves as lightcurves

            ax = mock.MagicMock()
            count = lightcurves.shade_time_segments(
                [ax],
                [
                    {"name": "seg1", "start": 1.0, "stop": 2.0},
                    {"name": "seg2", "start": 2.0, "stop": 3.0},
                ],
            )

        assert count == 2
        assert ax.axvspan.call_count == 2
        assert ax.axvline.call_count == 4


if __name__ == "__main__":
    unittest.main()
