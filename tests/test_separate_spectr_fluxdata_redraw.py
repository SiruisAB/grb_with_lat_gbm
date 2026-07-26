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


def test_public_axis_style_shows_all_ticks_and_spines():
    from grb_project.separate_spectr import _style_publication_axes

    fig, ax = plt.subplots()
    try:
        _style_publication_axes(ax)
        for spine in ax.spines.values():
            assert spine.get_visible()
        assert ax.xaxis.majorTicks[0].tick2line.get_visible()
        assert ax.yaxis.majorTicks[0].tick2line.get_visible()
    finally:
        plt.close(fig)


def test_publication_style_uses_embeddable_fonts_and_compact_panel_size():
    import matplotlib as mpl

    from grb_project.publication_style import (
        SPECTRUM_PANEL_FIGSIZE,
        configure_publication_matplotlib,
    )

    configure_publication_matplotlib()

    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["svg.fonttype"] == "none"
    assert SPECTRUM_PANEL_FIGSIZE[0] < 4.0


def test_default_analysis_spectrum_matches_reference_canvas_and_legend():
    from grb_project.publication_style import normalize_plot_style

    style = normalize_plot_style()

    assert style["spectrum_width_in"] == pytest.approx(12.0)
    assert style["spectrum_height_in"] == pytest.approx(8.0)
    assert style["diagnostic_legend_position"] == "top"
    assert style["spectrum_legend_position"] == "upper left"
    assert style["spectrum_legend_columns"] == 2
    assert style["spectrum_legend_frame"] is True
    assert style["spectrum_font_mode"] == "reference"


def test_default_spectrum_legend_is_internal_vertical_and_framed():
    from grb_project.separate_spectr import _place_unfolded_spectrum_legend

    fig, ax = plt.subplots(figsize=(12.0, 8.0))
    try:
        ax.plot([1, 2], [2, 3], label="Band")
        ax.errorbar([1.5], [2.5], yerr=[0.1], label="LAT")
        _place_unfolded_spectrum_legend(fig, ax)

        legend = ax.get_legend()
        assert legend is not None
        assert not fig.legends
        assert legend._loc == 2
        assert legend._ncols == 2
        assert legend.get_frame_on()
        assert legend.get_texts()[0].get_fontsize() == pytest.approx(18.0)
        assert [text.get_text() for text in legend.get_texts()] == ["Band", "LAT"]
    finally:
        plt.close(fig)


def test_diagnostic_and_corner_profiles_set_expected_canvas_sizes():
    from grb_project.publication_style import (
        CORNER_PANEL_FIGSIZE,
        DIAGNOSTIC_PANEL_FIGSIZE,
        style_corner_figure,
        style_diagnostic_figure,
    )

    diagnostic_fig, _ = plt.subplots()
    corner_fig, _ = plt.subplots()
    try:
        style_diagnostic_figure(diagnostic_fig)
        style_corner_figure(corner_fig)

        assert tuple(diagnostic_fig.get_size_inches()) == DIAGNOSTIC_PANEL_FIGSIZE
        assert tuple(corner_fig.get_size_inches()) == CORNER_PANEL_FIGSIZE
    finally:
        plt.close(diagnostic_fig)
        plt.close(corner_fig)


def test_diagnostic_legend_is_collapsed_into_external_detector_strip():
    from grb_project.publication_style import style_diagnostic_figure

    fig, (data_ax, residual_ax) = plt.subplots(2, 1)
    try:
        for idx, detector in enumerate(("LAT", "n3", "n7", "b0")):
            color = f"C{idx}"
            data_ax.plot([1, 2], [idx + 1, idx + 2], color=color, label=f"{detector} Model")
            data_ax.errorbar([1.5], [idx + 1.5], yerr=[0.1], color=color, label=detector)
        data_ax.legend()

        style_diagnostic_figure(fig)

        assert data_ax.get_legend() is None
        assert len(fig.legends) == 1
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [
            "LAT",
            "n3",
            "n7",
            "b0",
        ]
        assert data_ax.get_position().y1 <= 0.81
    finally:
        plt.close(fig)


def test_custom_plot_style_controls_canvas_fonts_and_legend_position():
    from grb_project.publication_style import style_diagnostic_figure

    fig, ax = plt.subplots()
    try:
        ax.plot([1, 2], [2, 3], label="LAT Model")
        ax.errorbar([1.5], [2.5], yerr=[0.1], label="LAT")
        ax.legend()
        style_diagnostic_figure(
            fig,
            {
                "diagnostic_width_in": 6.2,
                "diagnostic_height_in": 4.4,
                "axis_label_pt": 9.5,
                "tick_label_pt": 9.0,
                "legend_pt": 8.5,
                "legend_position": "upper left",
                "legend_columns": 1,
            },
        )

        assert tuple(fig.get_size_inches()) == pytest.approx((6.2, 4.4))
        assert ax.get_legend() is not None
        assert not fig.legends
        assert ax.get_legend()._loc == 2
    finally:
        plt.close(fig)


def test_manuscript_scaled_typography_is_consistent_across_figure_types():
    from grb_project.publication_style import (
        AXIS_LABEL_SIZE,
        CORNER_LABEL_SIZE,
        CORNER_PANEL_FIGSIZE,
        DIAGNOSTIC_PANEL_FIGSIZE,
        FULL_WIDTH_IN,
        LIGHTCURVE_AXIS_LABEL_SIZE,
        LIGHTCURVE_FIGSIZE,
        MANUSCRIPT_DIAGNOSTIC_WIDTH_IN,
        MANUSCRIPT_LIGHTCURVE_WIDTH_IN,
        MANUSCRIPT_SPECTRUM_WIDTH_IN,
        SPECTRUM_PANEL_FIGSIZE,
    )

    del FULL_WIDTH_IN
    effective = [
        AXIS_LABEL_SIZE * MANUSCRIPT_SPECTRUM_WIDTH_IN / SPECTRUM_PANEL_FIGSIZE[0],
        CORNER_LABEL_SIZE * MANUSCRIPT_DIAGNOSTIC_WIDTH_IN / CORNER_PANEL_FIGSIZE[0],
        CORNER_LABEL_SIZE * MANUSCRIPT_DIAGNOSTIC_WIDTH_IN / DIAGNOSTIC_PANEL_FIGSIZE[0],
        LIGHTCURVE_AXIS_LABEL_SIZE * MANUSCRIPT_LIGHTCURVE_WIDTH_IN / LIGHTCURVE_FIGSIZE[0],
    ]
    assert max(effective) - min(effective) < 0.05
    assert 8.0 <= effective[0] <= 9.0


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


def test_nai_rebin_axis_uses_legacy_15_centers_and_fixed_boundaries():
    from grb_project.separate_spectr import _build_nai_rebin_axis

    mid_points = np.geomspace(7.0, 1_000.0, 128)
    centers, err_low, err_high = _build_nai_rebin_axis(mid_points)

    assert len(centers) == 15
    assert centers[0] == pytest.approx(mid_points[6])
    assert centers[-1] == pytest.approx(mid_points[-6])
    assert centers[0] - 2.0 * err_low[0] == pytest.approx(8.0)
    assert centers[-1] + 2.0 * err_high[-1] == pytest.approx(900.0)
    assert np.all(np.diff(centers) > 0)
    assert np.all(err_low > 0)
    assert np.all(err_high > 0)


def test_log_rebin_axis_limits_lat_to_five_points():
    from grb_project.separate_spectr import _build_log_rebin_axis

    mid_points = np.geomspace(3.0e4, 1.0e8, 30)
    centers, err_low, err_high = _build_log_rebin_axis(mid_points, bin_count=5)

    assert len(centers) == 5
    assert centers[0] == pytest.approx(mid_points[0])
    assert centers[-1] == pytest.approx(mid_points[-1])
    assert np.all(err_low > 0)
    assert np.all(err_high > 0)


def test_lat_rebin_axis_uses_fixed_outer_boundaries():
    """LAT 展示网格的外边界固定为 30 MeV 到 100 GeV。"""
    from grb_project.separate_spectr import _build_lat_rebin_axis

    mid_points = np.geomspace(5.0e4, 8.0e7, 40)
    centers, err_low, err_high = _build_lat_rebin_axis(mid_points, bin_count=5)

    assert len(centers) == 5
    assert centers[0] - 2.0 * err_low[0] == pytest.approx(3.0e4)
    assert centers[-1] + 2.0 * err_high[-1] == pytest.approx(1.0e8)
    assert np.all(np.diff(centers) > 0)


def test_lat_rebin_axis_does_not_depend_on_the_burst():
    """外边界不随暴变化：同样的输入必须给出同样的网格。

    历史上这里曾按暴名设置不同的外边界（例如 GRB090510 用 40 MeV 上界），
    现在的实现对所有暴一律用 30 MeV-100 GeV。这条用例把这一点钉死，
    以免再出现"形参存在但从未生效"的状态。
    """
    from grb_project.separate_spectr import _build_lat_rebin_axis

    mid_points = np.geomspace(1.0e5, 3.5e7, 30)
    centers, err_low, err_high = _build_lat_rebin_axis(mid_points, bin_count=6)

    assert centers[0] - 2.0 * err_low[0] == pytest.approx(3.0e4)
    assert centers[-1] + 2.0 * err_high[-1] == pytest.approx(1.0e8)

    # 网格只由 mid_points 与 bin_count 决定，不接受任何逐暴参数。
    import inspect

    signature = inspect.signature(_build_lat_rebin_axis)
    assert list(signature.parameters) == ["mid_points", "bin_count"]


def test_lat_display_uncertainty_is_capped_but_raw_values_are_unchanged():
    from grb_project.separate_spectr import _cap_lat_display_uncertainty

    flux = np.array([2.0, 2.0, 0.0])
    raw = np.array([1.0, 3.0, 4.0])

    display = _cap_lat_display_uncertainty(flux, raw)

    assert display.tolist() == pytest.approx([1.0, 1.9, 4.0])
    assert raw.tolist() == [1.0, 3.0, 4.0]


def test_lat_upper_limits_are_created_only_for_finite_nonpositive_flux():
    from grb_project.separate_spectr import _lat_upper_limit_values

    limits = _lat_upper_limit_values(
        flux=np.array([2.0, 0.0, -1.0, np.nan]),
        uncertainty=np.array([3.0, 4.0, 5.0, 6.0]),
    )

    assert np.isnan(limits[0])
    assert limits[1] == pytest.approx(1.645 * 4.0)
    assert limits[2] == pytest.approx(1.645 * 5.0)
    assert np.isnan(limits[3])


def test_saved_lat_upper_limits_are_drawn_with_a_distinct_legend(tmp_path: Path):
    from grb_project.separate_spectr import _plot_lat_upper_limits

    path = tmp_path / "band_lat_upper_limit_0.1-1.txt"
    _write_fluxdata_file(
        path,
        [(2.0e5, 2.0e4, 3.0e4, 6.0e-7, 0.0, 3.6e-7)],
    )
    fig, ax = plt.subplots()
    try:
        displayed = _plot_lat_upper_limits(ax, path, color="C3")
        assert displayed.shape == (1, 6)
        assert ax.get_legend_handles_labels()[1] == ["LAT 95% upper limit"]
    finally:
        plt.close(fig)


def test_log_rebin_axis_supports_one_available_channel():
    from grb_project.separate_spectr import _build_log_rebin_axis

    centers, err_low, err_high = _build_log_rebin_axis([100.0], bin_count=5)

    assert centers.tolist() == [100.0]
    assert err_low.tolist() == [20.0]
    assert err_high.tolist() == [20.0]


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


def test_group_fluxdata_files_accepts_single_component_model_name(tmp_path: Path):
    from grb_project.separate_spectr import discover_fluxdata_groups

    flux_dir = tmp_path / "fluxdata"
    _write_fluxdata_file(
        flux_dir / "band_nai_n3_data_point_0.1-1.txt",
        [(10, 1, 1, 2e-8, 1e-9, 1e-9)],
    )

    groups = discover_fluxdata_groups(flux_dir)

    assert set(groups["0.1-1"]) == {"nai_n3"}


def test_detector_plot_specs_follow_available_detectors():
    from grb_project.separate_spectr import _detector_plot_specs

    specs = _detector_plot_specs(
        {
            "nai_n6": Path("n6.txt"),
            "nai_n7": Path("n7.txt"),
            "bgo_b1": Path("b1.txt"),
            "lat": Path("lat.txt"),
        }
    )

    assert [tag for tag, _, _ in specs] == ["nai_n6", "nai_n7", "bgo_b1", "lat"]
    assert [label for _, label, _ in specs] == ["NaI (n6)", "NaI (n7)", "BGO (b1)", "LAT"]


def test_old_lat_fluxdata_is_limited_to_five_display_points(tmp_path: Path):
    from grb_project.separate_spectr import _plot_fluxdata_detector

    path = tmp_path / "band+bb_lat_data_point_0.1-1.txt"
    rows = [
        (energy, 1.0, 1.0, 1.0e-7, 1.0e-8, 1.0e-8)
        for energy in np.geomspace(3.0e4, 1.0e8, 20)
    ]
    _write_fluxdata_file(path, rows)
    fig, ax = plt.subplots()
    try:
        displayed = _plot_fluxdata_detector(
            ax,
            path,
            label="LAT",
            marker="v",
            color="C0",
            max_points=5,
        )
        assert displayed.shape[0] == 5
        assert displayed[0, 0] == pytest.approx(rows[0][0])
        assert displayed[-1, 0] == pytest.approx(rows[-1][0])
    finally:
        plt.close(fig)


def test_display_filter_rejects_uncertainty_equal_to_or_larger_than_flux():
    from grb_project.separate_spectr import _energy_flux_display_mask

    mask = _energy_flux_display_mask(
        energy=np.array([10.0, 20.0, 30.0, 40.0]),
        flux=np.array([4.0, 4.0, 4.0, 4.0]),
        flux_err_neg=np.array([1.0, 4.0, 5.0, 1.0]),
        flux_err_pos=np.array([1.0, 1.0, 1.0, 5.0]),
    )

    assert mask.tolist() == [True, False, False, False]


def test_detector_with_no_reliable_points_adds_no_legend_entry(tmp_path: Path):
    from grb_project.separate_spectr import _plot_fluxdata_detector

    path = tmp_path / "band_lat_data_point_0.1-1.txt"
    _write_fluxdata_file(
        path,
        [
            (1.0e5, 1.0e4, 1.0e4, 2.0e-7, 2.0e-7, 2.0e-7),
            (2.0e5, 1.0e4, 1.0e4, 2.0e-7, 3.0e-7, 3.0e-7),
        ],
    )
    fig, ax = plt.subplots()
    try:
        displayed = _plot_fluxdata_detector(
            ax,
            path,
            label="LAT",
            marker="D",
            color="C0",
            max_points=5,
        )
        assert displayed.size == 0
        assert ax.get_legend_handles_labels() == ([], [])
    finally:
        plt.close(fig)


def test_lat_display_keeps_sparse_positive_bins(tmp_path: Path):
    from grb_project.separate_spectr import _plot_fluxdata_detector

    path = tmp_path / "band+bb_lat_data_point_0.1-1.txt"
    rows = [
        (3.0e4, 1.0, 1.0, 0.0, 0.0, 0.0),
        (1.2e5, 1.0, 1.0, 4.5e-7, 4.0e-7, 4.0e-7),
        (1.9e5, 1.0, 1.0, 2.1e-7, 2.0e-7, 2.0e-7),
        (1.0e8, 1.0, 1.0, 0.0, 0.0, 0.0),
    ]
    _write_fluxdata_file(path, rows)
    fig, ax = plt.subplots()
    try:
        displayed = _plot_fluxdata_detector(
            ax,
            path,
            label="LAT",
            marker="v",
            color="C0",
            max_points=5,
        )
        assert displayed.shape[0] == 2
        assert displayed[:, 0].tolist() == pytest.approx([1.2e5, 1.9e5])
    finally:
        plt.close(fig)


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


def test_saved_model_curve_round_trip_preserves_labels_and_energy_flux(tmp_path: Path):
    from grb_project.separate_spectr import load_model_curve_data, save_model_curve_data

    energy = np.geomspace(8.0, 1.0e8, 12)
    total = np.geomspace(1.0e-5, 1.0e-10, 12)
    path = tmp_path / "band+bb_model_curve_0.1-1.txt"

    save_model_curve_data(
        path,
        energy,
        [("Band+BB", total), ("Band", total * 0.8), ("BB", total * 0.2)],
    )
    loaded_energy, curves = load_model_curve_data(path)

    assert np.allclose(loaded_energy, energy)
    assert [label for label, _ in curves] == ["Band+BB", "Band", "BB"]
    assert np.allclose(curves[0][1], total)


def test_fit_overlay_converts_photon_model_to_energy_flux(tmp_path: Path):
    from grb_project.separate_spectr import _overlay_fit_curves

    json_path = tmp_path / "GRB231129C_allmodel_nested_compact.json"
    _write_fit_json(json_path)
    fig, ax = plt.subplots()
    try:
        model_data = _overlay_fit_curves(
            ax,
            timebin="0.1-1",
            bnname="GRB231129C",
            fit_json_path=json_path,
        )
        assert model_data is not None
        energy, values = model_data
        assert energy[-1] == pytest.approx(1.0e8)
        assert values[0][0] > 0
        assert [line.get_label() for line in ax.lines] == ["Band+BB", "Band", "BB"]
    finally:
        plt.close(fig)


def test_redraw_prefers_saved_model_curves_without_fit_json(tmp_path: Path):
    from grb_project.separate_spectr import redraw_fluxdata_timebin, save_model_curve_data

    flux_dir = tmp_path / "fluxdata"
    rows = [(10.0, 1.0, 1.2, 4.0e-8, 8.0e-9, 8.0e-9)]
    for prefix in ["nai_n3", "nai_n7", "bgo_b0", "lat"]:
        _write_fluxdata_file(flux_dir / f"band+bb_{prefix}_data_point_0.1-1.txt", rows)
    energy = np.geomspace(8.0, 1.0e8, 20)
    total = np.geomspace(1.0e-5, 1.0e-10, 20)
    save_model_curve_data(
        flux_dir / "band+bb_model_curve_0.1-1.txt",
        energy,
        [("Band+BB", total), ("Band", total * 0.8), ("BB", total * 0.2)],
    )

    out_path = redraw_fluxdata_timebin(
        timebin="0.1-1",
        fluxdata_dir=flux_dir,
        output_dir=tmp_path / "out",
        bnname="GRB231129C",
    )

    assert out_path.exists()


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

    def fake_redraw_all(
        fluxdata_dir,
        output_dir,
        bnname,
        fit_json_path=None,
        show_lat_upper_limits=True,
    ):
        called["fluxdata_dir"] = fluxdata_dir
        called["output_dir"] = output_dir
        called["bnname"] = bnname
        called["fit_json_path"] = fit_json_path
        called["show_lat_upper_limits"] = show_lat_upper_limits
        return {"single_plots": [], "overview": []}

    monkeypatch.setattr(separate_spectr, "redraw_all_fluxdata_spectra", fake_redraw_all)

    separate_spectr.redraw_bandbb_fluxdata_outputs(bnname="GRB231129C", result_root=tmp_path)

    assert called["bnname"] == "GRB231129C"
    assert called["fluxdata_dir"] == tmp_path / "GRB231129C" / "band+bb" / "fluxdata"
    assert called["output_dir"] == tmp_path / "GRB231129C" / "band+bb" / "fluxdata"
    assert called["fit_json_path"] is None
    assert called["show_lat_upper_limits"] is True


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
