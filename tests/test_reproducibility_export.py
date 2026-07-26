from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class _Parameter:
    def __init__(self, value, unit=""):
        self.value = value
        self.unit = unit
        self.min_value = 0.0
        self.max_value = 10.0
        self.prior = "uniform"


class _Component:
    def __init__(self, parameter, power):
        self.parameter = parameter
        self.power = power

    def __call__(self, energy):
        return self.parameter.value * np.asarray(energy) ** self.power


class _Composite:
    def __init__(self, functions):
        self.functions = functions

    def __call__(self, energy):
        return sum(function(energy) for function in self.functions)


def _fake_analysis():
    p1 = _Parameter(2.0, "keV-1 s-1 cm-2")
    p2 = _Parameter(0.5, "keV-1 s-1 cm-2")
    functions = [_Component(p1, -1.0), _Component(p2, -2.0)]
    shape = _Composite(functions)
    source = type("Source", (), {"spectrum": type("Spectrum", (), {"main": type("Main", (), {"shape": shape})()})()})()
    model = type(
        "Model",
        (),
        {
            "point_sources": {"GRB": source},
            "free_parameters": {"GRB.p1": p1, "GRB.p2": p2},
            "parameters": {"GRB.p1": p1, "GRB.p2": p2},
        },
    )()
    results = type(
        "Results",
        (),
        {
            "optimized_model": model,
            "samples": np.array([[1.8, 2.0, 2.2], [0.4, 0.5, 0.6]]),
            "log_probability": np.array([-3.0, -2.0, -2.5]),
            "statistical_measures": pd.Series({"AIC": 10.0, "BIC": 12.0}),
        },
    )()
    return type("Analysis", (), {"results": results})()


def test_export_bundle_saves_exact_curves_samples_and_metadata(tmp_path):
    from grb_project.reproducibility_export import KEV_TO_ERG, export_spectral_reproducibility_bundle

    output = export_spectral_reproducibility_bundle(
        _fake_analysis(),
        result_dir=tmp_path,
        grb_name="GRBTEST",
        bnname="bn000",
        model_name="band+bb",
        bin_start=0.1,
        bin_end=1.0,
        analysis_mode="gbm+lat",
        detector_names=["n0", "b0", "LAT"],
        energy_points=8,
        posterior_curves=3,
    )

    sed = np.loadtxt(output / "model_sed.csv", delimiter=",", skiprows=1)
    assert sed.shape[0] == 8
    assert sed[-1, 0] == 1.0e8
    expected = KEV_TO_ERG * sed[0, 0] ** 2 * (
        2.0 / sed[0, 0] + 0.5 / sed[0, 0] ** 2
    )
    assert sed[0, 2] == expected
    assert (output / "model_sed_posterior.csv").exists()
    samples = np.load(output / "posterior_samples.npz")
    assert samples["samples"].shape == (2, 3)
    metadata = json.loads((output / "fit_parameters.json").read_text(encoding="utf-8"))
    assert metadata["energy_max_keV"] == 1.0e8
    assert metadata["statistical_measures"]["BIC"] == 12.0


def test_export_counts_figure_data_saves_lines_and_errorbars(tmp_path):
    from grb_project.reproducibility_export import (
        export_counts_figure_data,
        redraw_counts_figure_data,
    )

    fig, (ax_data, ax_resid) = plt.subplots(2, 1)
    try:
        ax_data.errorbar(
            [10.0, 20.0],
            [4.0, 5.0],
            xerr=[1.0, 2.0],
            yerr=[0.5, 0.7],
            label="n0",
        )
        ax_data.plot([10.0, 20.0], [4.2, 4.8], label="n0 Model")
        ax_resid.axhline(0.0, linestyle="--")
        ax_resid.errorbar(
            [10.0, 20.0],
            [-0.2, 0.3],
            yerr=[0.1, 0.2],
            marker="o",
            linestyle="none",
        )
        ax_data.set_xscale("log")
        ax_data.set_yscale("log")
        ax_resid.set_xscale("log")
        ax_data.set_ylabel("Net rate")
        ax_resid.set_xlabel("Energy (keV)")
        ax_resid.set_ylabel("Residuals")
        output = export_counts_figure_data(fig, tmp_path)
    finally:
        plt.close(fig)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["axes"]) == 2
    assert payload["axes"][0]["lines"]
    assert any(collection["segments"] for collection in payload["axes"][0]["collections"])

    redrawn = redraw_counts_figure_data(output, tmp_path / "counts.png")
    assert redrawn.exists()
    assert redrawn.stat().st_size > 0
