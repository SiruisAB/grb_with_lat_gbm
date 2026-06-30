from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _write_fluxdata_file(path: Path, rows: list[tuple[float, float, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(" ".join(str(value) for value in row) + "\n")


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

    def fake_redraw_all(fluxdata_dir, output_dir, bnname):
        called["fluxdata_dir"] = fluxdata_dir
        called["output_dir"] = output_dir
        called["bnname"] = bnname
        return {"single_plots": [], "overview": []}

    monkeypatch.setattr(separate_spectr, "redraw_all_fluxdata_spectra", fake_redraw_all)

    separate_spectr.redraw_bandbb_fluxdata_outputs(bnname="GRB231129C", result_root=tmp_path)

    assert called["bnname"] == "GRB231129C"
    assert called["fluxdata_dir"] == tmp_path / "GRB231129C" / "band+bb" / "fluxdata"
    assert called["output_dir"] == tmp_path / "GRB231129C" / "band+bb" / "fluxdata"


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
