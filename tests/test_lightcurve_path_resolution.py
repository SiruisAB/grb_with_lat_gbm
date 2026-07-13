from pathlib import Path
import importlib
import sys
import types
from unittest import mock

import pandas as pd

from grb_project.project import run_joint_lightcurve
from grb_project.web_app import _joint_gbm_lat_rows, _target_defaults


def _import_lightcurves_with_stubs():
    sys.modules.pop("grb_project.lightcurves", None)
    three_ml = types.ModuleType("threeML")
    three_ml.DataList = object
    three_ml.OGIPLike = object
    three_ml.TimeSeriesBuilder = object
    three_ml.silence_warnings = lambda: None
    three_ml_config_pkg = types.ModuleType("threeML.config")
    three_ml_config_mod = types.ModuleType("threeML.config.config")
    three_ml_config_mod.threeML_config = types.SimpleNamespace()
    three_ml_io_pkg = types.ModuleType("threeML.io")
    three_ml_plot_pkg = types.ModuleType("threeML.io.plotting")
    three_ml_step_plot_mod = types.ModuleType("threeML.io.plotting.step_plot")
    three_ml_step_plot_mod.step_plot = lambda *args, **kwargs: None
    with mock.patch.dict(
        "sys.modules",
        {
            "threeML": three_ml,
            "threeML.config": three_ml_config_pkg,
            "threeML.config.config": three_ml_config_mod,
            "threeML.io": three_ml_io_pkg,
            "threeML.io.plotting": three_ml_plot_pkg,
            "threeML.io.plotting.step_plot": three_ml_step_plot_mod,
        },
    ):
        return importlib.import_module("grb_project.lightcurves")


def test_target_defaults_keep_special_burst_time_windows():
    catalog_row = pd.Series(
        {
            "gcn_name": "GRB 231129C",
            "t90_start": 0.0,
            "t90": 8.0,
            "ra": 0.0,
            "dec": 0.0,
            "back_interval_low_start": -24,
            "back_interval_low_stop": -5,
            "back_interval_high_start": 350,
            "back_interval_high_stop": 400,
        }
    )
    lat_row = pd.Series({"gcn_name": "GRB 231129C", "trigger_met": 123.0})
    defaults = _target_defaults(catalog_row, lat_row, "bn231129799")

    assert defaults["active_interval"] == "0.1-8.5"
    assert defaults["background_low"] == "-130--10"
    assert defaults["background_high"] == "100-200"


def test_joint_gbm_lat_rows_keeps_only_catalog_intersection():
    gbm_catalog = pd.DataFrame(
        {"value": [1, 2, 3]},
        index=["bn230101001", "bn230102002", "bn230103003"],
    )
    lat_catalog = pd.DataFrame(
        {"trigger_met": [123.0, 456.0]},
        index=["bn230102002", "bn240101001"],
    )

    joint = _joint_gbm_lat_rows(gbm_catalog, lat_catalog)

    assert joint.index.tolist() == ["bn230102002"]
    assert joint["value"].tolist() == [2]


def test_expected_lightcurve_output_and_lat_input_layout(tmp_path):
    result_root = Path("/home/mxr/lee/gbmtest/results_sample")
    grb_name = "GRB231129C"

    result_dir = result_root / grb_name
    lat_dir = result_dir / "lat"
    lightcurve_path = result_root / grb_name / f"{grb_name}_lightcurve.png"

    assert result_dir == Path("/home/mxr/lee/gbmtest/results_sample/GRB231129C")
    assert lat_dir == Path("/home/mxr/lee/gbmtest/results_sample/GRB231129C/lat")
    assert lightcurve_path == Path("/home/mxr/lee/gbmtest/results_sample/GRB231129C/GRB231129C_lightcurve.png")


def test_discover_lat_prob_fit_files_uses_grb_lat_interval_layout(tmp_path):
    lightcurves = _import_lightcurves_with_stubs()
    lat_dir = tmp_path / "GRB231129C" / "lat"
    expected = lat_dir / "interval0.1-8.5" / "gll_ft1_tr_bn231129799_v00_filt_prob.fit"
    ignored = lat_dir / "bn231129799" / "interval0.1-8.5" / "gll_ft1_tr_bn231129799_v00_filt_prob.fit"
    expected.parent.mkdir(parents=True)
    ignored.parent.mkdir(parents=True)
    expected.write_text("", encoding="utf-8")
    ignored.write_text("", encoding="utf-8")

    assert lightcurves.discover_lat_prob_fit_files(lat_dir) == [expected]


def test_run_joint_lightcurve_passes_grb_lat_dir_to_prob_discovery(tmp_path):
    result_dir = tmp_path / "GRB231129C"
    lat_dir = result_dir / "lat"

    fake_lightcurves = types.ModuleType("grb_project.lightcurves")
    fake_lightcurves.parse_background_interval_tuple = lambda text: tuple(
        part.strip() for part in text.split(",") if part.strip()
    )
    fake_lightcurves.plot_gbm_lat_lightcurve_figure = mock.MagicMock()

    with mock.patch.dict("sys.modules", {"grb_project.lightcurves": fake_lightcurves}):
        out = run_joint_lightcurve(
            bnname="bn231129799",
            grb_name="GRB231129C",
            result_dir=result_dir,
            lat_dir=lat_dir,
            trigger_met=123.0,
            data_dir=str(tmp_path / "data"),
            active_interval="0.1-8.5",
            background_interval="-130--10,100-200",
            fixed_num_time_bins=None,
            gbm_start=0.0,
            gbm_stop=10.0,
            gbm_display_pad_before_s=1.0,
            gbm_display_pad_after_s=1.0,
            plot_joint_lightcurve=True,
            include_lat=True,
            lat_prob_threshold=0.9,
            nai_bands_kev=((8.0, 50.0), (50.0, 300.0)),
            bgo_band_kev=(300.0, 38000.0),
        )

    assert out == result_dir / "GRB231129C_lightcurve.png"
    assert fake_lightcurves.plot_gbm_lat_lightcurve_figure.call_args.kwargs["lat_prob_bn_dir"] == lat_dir
