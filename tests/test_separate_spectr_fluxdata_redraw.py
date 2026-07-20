from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


def _write_fluxdata_file(path: Path, rows: list[tuple[float, float, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(" ".join(str(value) for value in row) + "\n")


def _write_fit_json(path: Path) -> None:
    path.write_text(
        """
{
  "GRB231129C": {
    "bins": {
      "bn231129799_bin_0.10_1.00": {
        "meta": {
          "time_bin_identifier": "bn231129799_bin_0.10_1.00",
          "bin_start_time": 0.1,
          "bin_end_time": 1.0
        },
        "band+bb": {
          "GRB.spectrum.main.composite.K_1_value": 0.6477081429398616,
          "GRB.spectrum.main.composite.alpha_1_value": 0.41591667797691745,
          "GRB.spectrum.main.composite.xp_1_value": 233.95285290292108,
          "GRB.spectrum.main.composite.beta_1_value": -2.807365626480064,
          "GRB.spectrum.main.composite.K_2_value": 1.167476890913784e-06,
          "GRB.spectrum.main.composite.kT_2_value": 155.14230259308331
        }
      }
    }
  }
}
""".strip(),
        encoding="utf-8",
    )


def test_public_axis_style_hides_top_and_right_ticks_and_spines():
    from grb_project.separate_spectr import _style_publication_axes

    fig, ax = plt.subplots()
    try:
        _style_publication_axes(ax)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["left"].get_visible()
        assert ax.spines["bottom"].get_visible()
    finally:
        plt.close(fig)


def test_bgo_rebin_axis_uses_active_range_and_keeps_centers_below_40_mev():
    from grb_project.separate_spectr import _build_bgo_rebin_axis

    energy_min = np.array([200.0, 300.0, 20_000.0])
    energy_max = np.array([300.0, 20_000.0, 42_000.0])
    centers, err_low, err_high = _build_bgo_rebin_axis((energy_min, energy_max))

    assert centers[-1] < 40_000.0
    assert centers[0] - err_low[0] == pytest.approx(200.0)
    assert centers[-1] + err_high[-1] == pytest.approx(40_000.0)
    assert np.all(err_low > 0)
    assert np.all(err_high > 0)


def test_group_fluxdata_files_by_timebin(tmp_path: Path):
    from grb_project.separate_spectr import discover_fluxdata_groups

    flux_dir = tmp_path / "fluxdata"
    _write_fluxdata_file(flux_dir / "band+bb_nai_n3_data_point_0.1-1.txt", [(1, 0.1, 0.1, 2, 0.2, 0.2)])
    _write_fluxdata_file(flux_dir / "band+bb_nai_n7_data_point_0.1-1.txt", [(1, 0.1, 0.1, 2, 0.2, 0.2)])
    _write_fluxdata_file(flux_dir / "band+bb_bgo_b0_data_point_0.1-1.txt", [(1, 0.1, 0.1, 2, 0.2, 0.2)])
    _write_fluxdata_file(flux_dir / "band+bb_lat_data_point_0.1-1.txt", [(1, 0.1, 0.1, 2, 0.2, 0.2)])

    groups = discover_fluxdata_groups(flux_dir)

    assert list(groups) == ["0.1-1"]
    assert set(groups["0.1-1"]) == {"nai_n3", "nai_n7", "bgo_b0", "lat"}


def test_load_bandbb_fit_params_from_json(tmp_path: Path):
    from grb_project.separate_spectr import load_bandbb_fit_params

    json_path = tmp_path / "GRB231129C_allmodel_nested_compact.json"
    _write_fit_json(json_path)

    params = load_bandbb_fit_params(json_path, bnname="GRB231129C", timebin="0.1-1")

    assert params["K_1"] == 0.6477081429398616
    assert params["alpha_1"] == 0.41591667797691745
    assert params["xp_1"] == 233.95285290292108
    assert params["beta_1"] == -2.807365626480064
    assert params["K_2"] == 1.167476890913784e-06
    assert params["kT_2"] == 155.14230259308331


def test_compute_bandbb_component_curves_returns_total_and_components():
    from grb_project.separate_spectr import compute_bandbb_curves

    xs = np.logspace(1, 3, 5)
    curves = compute_bandbb_curves(
        xs,
        K_1=1.0,
        alpha_1=-0.5,
        xp_1=200.0,
        beta_1=-2.5,
        K_2=1e-6,
        kT_2=100.0,
    )

    assert set(curves) == {"total", "band", "bb"}
    assert curves["total"].shape == xs.shape
    assert curves["band"].shape == xs.shape
    assert curves["bb"].shape == xs.shape


def test_redraw_single_timebin_creates_pdf(tmp_path: Path):
    from grb_project.separate_spectr import redraw_fluxdata_timebin

    flux_dir = tmp_path / "fluxdata"
    rows = [
        (10.0, 1.0, 1.2, 4.0e-8, 8.0e-9, 8.0e-9),
        (20.0, 2.0, 2.4, 6.0e-8, 1.0e-8, 1.0e-8),
    ]
    for prefix in ["nai_n3", "nai_n7", "bgo_b0", "lat"]:
        _write_fluxdata_file(flux_dir / f"band+bb_{prefix}_data_point_0.1-1.txt", rows)

    out_path = redraw_fluxdata_timebin(
        timebin="0.1-1",
        fluxdata_dir=flux_dir,
        output_dir=tmp_path / "out",
        bnname="GRB231129C",
    )

    assert out_path.name == "bs_GRB231129C_gbm_lat_spectra_band+bb_0.1-1.pdf"
    assert out_path.exists()


def test_redraw_single_timebin_overlays_fit_curves(tmp_path: Path):
    from grb_project.separate_spectr import redraw_fluxdata_timebin

    flux_dir = tmp_path / "fluxdata"
    rows = [(10.0, 1.0, 1.2, 4.0e-8, 8.0e-9, 8.0e-9)]
    for prefix in ["nai_n3", "nai_n7", "bgo_b0", "lat"]:
        _write_fluxdata_file(flux_dir / f"band+bb_{prefix}_data_point_0.1-1.txt", rows)
    json_path = tmp_path / "GRB231129C_allmodel_nested_compact.json"
    _write_fit_json(json_path)

    out_path = redraw_fluxdata_timebin(
        timebin="0.1-1",
        fluxdata_dir=flux_dir,
        output_dir=tmp_path / "out",
        bnname="GRB231129C",
        fit_json_path=json_path,
    )

    assert out_path.exists()


def test_batch_redraw_creates_individual_and_overview_outputs(tmp_path: Path):
    from grb_project.separate_spectr import redraw_all_fluxdata_spectra

    flux_dir = tmp_path / "fluxdata"
    rows = [(10.0, 1.0, 1.2, 4.0e-8, 8.0e-9, 8.0e-9)]
    for timebin in ["0.1-1", "1-3"]:
        for prefix in ["nai_n3", "nai_n7", "bgo_b0", "lat"]:
            _write_fluxdata_file(flux_dir / f"band+bb_{prefix}_data_point_{timebin}.txt", rows)

    outputs = redraw_all_fluxdata_spectra(
        fluxdata_dir=flux_dir,
        output_dir=tmp_path / "out",
        bnname="GRB231129C",
    )

    assert (tmp_path / "out" / "bs_GRB231129C_gbm_lat_spectra_band+bb_0.1-1.pdf").exists()
    assert (tmp_path / "out" / "bs_GRB231129C_gbm_lat_spectra_band+bb_1-3.pdf").exists()
    assert (tmp_path / "out" / "bs_GRB231129C_gbm_lat_spectra_band+bb_overview.pdf").exists()
    assert len(outputs["single_plots"]) == 2
    assert len(outputs["overview"]) == 1


def test_redraw_entrypoint_uses_default_bandbb_fluxdata_path(monkeypatch, tmp_path: Path):
    from grb_project import separate_spectr

    called = {}

    def fake_redraw_all(fluxdata_dir, output_dir, bnname, fit_json_path=None):
        called["fluxdata_dir"] = fluxdata_dir
        called["output_dir"] = output_dir
        called["bnname"] = bnname
        called["fit_json_path"] = fit_json_path
        return {"single_plots": [], "overview": []}

    monkeypatch.setattr(separate_spectr, "redraw_all_fluxdata_spectra", fake_redraw_all)

    separate_spectr.redraw_bandbb_fluxdata_outputs(bnname="GRB231129C", result_root=tmp_path)

    assert called["bnname"] == "GRB231129C"
    assert called["fluxdata_dir"] == tmp_path / "GRB231129C" / "band+bb" / "fluxdata"
    assert called["output_dir"] == tmp_path / "GRB231129C" / "band+bb" / "fluxdata"
    assert called["fit_json_path"] is None


def test_cli_main_creates_log_file_and_outputs(tmp_path: Path):
    from grb_project import separate_spectr

    flux_dir = tmp_path / "GRB231129C" / "band+bb" / "fluxdata"
    rows = [(10.0, 1.0, 1.2, 4.0e-8, 8.0e-9, 8.0e-9)]
    for prefix in ["nai_n3", "nai_n7", "bgo_b0", "lat"]:
        _write_fluxdata_file(flux_dir / f"band+bb_{prefix}_data_point_0.1-1.txt", rows)

    code = separate_spectr.main([
        "--bnname",
        "GRB231129C",
        "--result-root",
        str(tmp_path),
        "--log-file",
        str(tmp_path / "redraw.log"),
    ])

    assert code == 0
    assert (flux_dir / "bs_GRB231129C_gbm_lat_spectra_band+bb_0.1-1.pdf").exists()
    assert (flux_dir / "bs_GRB231129C_gbm_lat_spectra_band+bb_overview.pdf").exists()
    assert (tmp_path / "redraw.log").exists()
    assert "Fluxdata directory" in (tmp_path / "redraw.log").read_text(encoding="utf-8")


def test_cli_logs_json_and_output_paths(tmp_path: Path):
    from grb_project import separate_spectr

    flux_dir = tmp_path / "GRB231129C" / "band+bb" / "fluxdata"
    rows = [(10.0, 1.0, 1.2, 4.0e-8, 8.0e-9, 8.0e-9)]
    for prefix in ["nai_n3", "nai_n7", "bgo_b0", "lat"]:
        _write_fluxdata_file(flux_dir / f"band+bb_{prefix}_data_point_0.1-1.txt", rows)
    json_path = tmp_path / "GRB231129C_allmodel_nested_compact.json"
    _write_fit_json(json_path)

    code = separate_spectr.main([
        "--bnname",
        "GRB231129C",
        "--result-root",
        str(tmp_path),
        "--fit-json",
        str(json_path),
        "--save-result-dir",
        str(tmp_path / "custom_out"),
        "--log-file",
        str(tmp_path / "redraw.log"),
    ])

    assert code == 0
    assert (tmp_path / "custom_out" / "bs_GRB231129C_gbm_lat_spectra_band+bb_0.1-1.pdf").exists()
    log_text = (tmp_path / "redraw.log").read_text(encoding="utf-8")
    assert "Fit JSON" in log_text
    assert str(json_path) in log_text
    assert str(tmp_path / "custom_out") in log_text
