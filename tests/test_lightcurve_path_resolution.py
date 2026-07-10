from pathlib import Path

import pandas as pd

from grb_project.web_app import _target_defaults


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


def test_expected_lightcurve_output_and_lat_input_layout(tmp_path):
    result_root = tmp_path / "results_sample"
    grb_name = "GRB231129C"
    bnname = "bn231129799"

    result_dir = result_root / grb_name / bnname
    lat_dir = result_dir / "lat" / bnname
    lightcurve_path = result_root / grb_name / f"{grb_name}_lightcurve.png"

    assert result_dir == Path("/home/mxr/lee/gbmtest/results_sample/GRB231129C/bn231129799")
    assert lat_dir == Path("/home/mxr/lee/gbmtest/results_sample/GRB231129C/bn231129799/lat/bn231129799")
    assert lightcurve_path == Path("/home/mxr/lee/gbmtest/results_sample/GRB231129C/GRB231129C_lightcurve.png")
