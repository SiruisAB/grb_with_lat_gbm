"""Export self-contained products for reproducing spectral-fit figures."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

KEV_TO_ERG = 1.602176634e-9
DEFAULT_ENERGY_POINTS = 512
DEFAULT_POSTERIOR_CURVES = 256


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return text or "component"


def _time_tag(bin_start: float, bin_end: float) -> str:
    def fmt(value: float) -> str:
        return f"{float(value):.6f}".rstrip("0").rstrip(".")

    return f"{fmt(bin_start)}-{fmt(bin_end)}"


def _source_shape(model):
    point_sources = getattr(model, "point_sources", {})
    if "GRB" in point_sources:
        source = point_sources["GRB"]
    elif point_sources:
        source = next(iter(point_sources.values()))
    else:
        raise ValueError("The fitted model has no point source")
    return source.spectrum.main.shape


def _component_functions(shape) -> list[tuple[str, Any]]:
    functions = list(getattr(shape, "functions", []) or [])
    if not functions:
        return []
    counts: dict[str, int] = {}
    components: list[tuple[str, Any]] = []
    for function in functions:
        base = _safe_name(function.__class__.__name__)
        counts[base] = counts.get(base, 0) + 1
        name = base if counts[base] == 1 else f"{base}_{counts[base]}"
        components.append((name, function))
    return components


def _evaluate_curves(shape, components, energies: np.ndarray) -> dict[str, np.ndarray]:
    curves = {"total": np.asarray(shape(energies), dtype=float)}
    for name, function in components:
        curves[name] = np.asarray(function(energies), dtype=float)
    return curves


def _parameter_metadata(model) -> list[dict[str, Any]]:
    free_names = set(getattr(model, "free_parameters", {}).keys())
    rows: list[dict[str, Any]] = []
    for name, parameter in getattr(model, "parameters", {}).items():
        prior = getattr(parameter, "prior", None)
        rows.append(
            {
                "name": str(name),
                "value": _json_value(getattr(parameter, "value", None)),
                "unit": str(getattr(parameter, "unit", "")),
                "minimum": _json_value(getattr(parameter, "min_value", None)),
                "maximum": _json_value(getattr(parameter, "max_value", None)),
                "free": str(name) in free_names,
                "prior": None if prior is None else str(prior),
            }
        )
    return rows


def _save_posterior_samples(results, output: Path, parameter_names: list[str]) -> int:
    samples = np.asarray(results.samples, dtype=float)
    payload: dict[str, Any] = {
        "samples": samples,
        "parameter_names": np.asarray(parameter_names, dtype="U"),
    }
    log_probability = getattr(results, "log_probability", None)
    if log_probability is not None:
        payload["log_probability"] = np.asarray(log_probability, dtype=float)
    np.savez_compressed(output, **payload)
    return int(samples.shape[1]) if samples.ndim == 2 else 0


def _posterior_curve_quantiles(
    results,
    model,
    shape,
    components,
    energies: np.ndarray,
    max_curves: int,
) -> tuple[dict[str, np.ndarray], int]:
    samples = np.asarray(results.samples, dtype=float)
    free_parameters = list(getattr(model, "free_parameters", {}).values())
    if samples.ndim != 2 or samples.shape[0] != len(free_parameters) or samples.shape[1] == 0:
        return {}, 0

    count = min(int(max_curves), int(samples.shape[1]))
    indices = np.linspace(0, samples.shape[1] - 1, count, dtype=int)
    original = [float(parameter.value) for parameter in free_parameters]
    collected: dict[str, list[np.ndarray]] = {"total": []}
    for name, _ in components:
        collected[name] = []

    try:
        for index in indices:
            for parameter, value in zip(free_parameters, samples[:, index]):
                parameter.value = float(value)
            curves = _evaluate_curves(shape, components, energies)
            for name, values in curves.items():
                collected[name].append(values)
    finally:
        for parameter, value in zip(free_parameters, original):
            parameter.value = value

    quantiles: dict[str, np.ndarray] = {}
    for name, values in collected.items():
        stack = np.asarray(values, dtype=float)
        quantiles[name] = np.nanpercentile(stack, [16.0, 50.0, 84.0], axis=0)
    return quantiles, count


def export_spectral_reproducibility_bundle(
    analysis,
    *,
    result_dir: str | Path,
    grb_name: str,
    bnname: str,
    model_name: str,
    bin_start: float,
    bin_end: float,
    analysis_mode: str,
    detector_names: Iterable[str] = (),
    energy_points: int = DEFAULT_ENERGY_POINTS,
    posterior_curves: int = DEFAULT_POSTERIOR_CURVES,
) -> Path:
    """Save exact model curves, posterior samples, and fit metadata."""
    output = Path(result_dir) / "reproducibility" / _time_tag(bin_start, bin_end)
    output.mkdir(parents=True, exist_ok=True)

    results = analysis.results
    model = results.optimized_model
    shape = _source_shape(model)
    components = _component_functions(shape)
    energy_max = 1.0e8 if "lat" in analysis_mode.lower() else 4.0e4
    energies = np.logspace(np.log10(8.0), np.log10(energy_max), int(energy_points))
    median_curves = _evaluate_curves(shape, components, energies)

    parameter_names = list(getattr(model, "free_parameters", {}).keys())
    sample_count = _save_posterior_samples(results, output / "posterior_samples.npz", parameter_names)
    quantiles, curve_count = _posterior_curve_quantiles(
        results,
        model,
        shape,
        components,
        energies,
        max_curves=posterior_curves,
    )

    columns = [energies]
    headers = ["energy_keV"]
    for name, values in median_curves.items():
        columns.extend([values, KEV_TO_ERG * energies**2 * values])
        headers.extend([f"{name}_dnde_ph_cm-2_s-1_keV-1", f"{name}_sed_erg_cm-2_s-1"])
    np.savetxt(
        output / "model_sed.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
    )

    if quantiles:
        columns = [energies]
        headers = ["energy_keV"]
        for name, values in quantiles.items():
            for row, label in zip(values, ("p16", "p50", "p84")):
                columns.append(KEV_TO_ERG * energies**2 * row)
                headers.append(f"{name}_sed_{label}_erg_cm-2_s-1")
        np.savetxt(
            output / "model_sed_posterior.csv",
            np.column_stack(columns),
            delimiter=",",
            header=",".join(headers),
            comments="",
        )

    fit_metadata = {
        "grb_name": grb_name,
        "bnname": bnname,
        "model": model_name,
        "analysis_mode": analysis_mode,
        "bin_start_s": float(bin_start),
        "bin_end_s": float(bin_end),
        "energy_min_keV": 8.0,
        "energy_max_keV": energy_max,
        "parameters": _parameter_metadata(model),
        "statistical_measures": _json_value(getattr(results, "statistical_measures", {}).to_dict()),
    }
    (output / "fit_parameters.json").write_text(
        json.dumps(fit_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "format_version": 1,
        "description": "Self-contained spectral-fit products for figure reproduction",
        "grb_name": grb_name,
        "bnname": bnname,
        "model": model_name,
        "analysis_mode": analysis_mode,
        "time_bin_s": [float(bin_start), float(bin_end)],
        "detectors": [str(name) for name in detector_names],
        "component_columns": list(median_curves),
        "posterior_sample_count": sample_count,
        "posterior_curve_count": curve_count,
        "energy_grid_points": int(energy_points),
        "files": {
            "median_model_sed": "model_sed.csv",
            "posterior_model_sed": "model_sed_posterior.csv" if quantiles else None,
            "posterior_samples": "posterior_samples.npz",
            "fit_parameters": "fit_parameters.json",
            "folded_counts_plot": "folded_counts_plot.json",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def export_counts_figure_data(figure, output_dir: str | Path) -> Path:
    """Save the plotted folded-count and residual arrays from a threeML figure."""
    axes_payload: list[dict[str, Any]] = []
    for axis_index, ax in enumerate(figure.axes):
        lines = []
        for line_index, line in enumerate(ax.get_lines()):
            lines.append(
                {
                    "index": line_index,
                    "label": str(line.get_label()),
                    "x": _json_value(np.asarray(line.get_xdata(), dtype=float).tolist()),
                    "y": _json_value(np.asarray(line.get_ydata(), dtype=float).tolist()),
                    "color": _json_value(np.asarray(line.get_color()))
                    if not isinstance(line.get_color(), str)
                    else str(line.get_color()),
                    "linestyle": str(line.get_linestyle()),
                    "marker": str(line.get_marker()),
                }
            )
        collections = []
        for collection_index, collection in enumerate(ax.collections):
            segments = []
            if hasattr(collection, "get_segments"):
                for segment in collection.get_segments():
                    segments.append(_json_value(np.asarray(segment, dtype=float).tolist()))
            offsets = []
            if hasattr(collection, "get_offsets"):
                offsets = _json_value(np.asarray(collection.get_offsets(), dtype=float).tolist())
            collections.append({"index": collection_index, "segments": segments, "offsets": offsets})
        axes_payload.append(
            {
                "index": axis_index,
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "xscale": ax.get_xscale(),
                "yscale": ax.get_yscale(),
                "xlim": _json_value(list(ax.get_xlim())),
                "ylim": _json_value(list(ax.get_ylim())),
                "lines": lines,
                "collections": collections,
            }
        )

    output = Path(output_dir) / "folded_counts_plot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"format_version": 1, "axes": axes_payload}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def _saved_color(value) -> object:
    if isinstance(value, str) and value.startswith("["):
        parsed = np.fromstring(value.strip("[]"), sep=" ")
        if parsed.size in (3, 4):
            return parsed
    return value


def _segment_errors(segments, centers: np.ndarray, coordinate: int) -> np.ndarray:
    errors = np.zeros((2, len(centers)), dtype=float)
    for index, (segment, center) in enumerate(zip(segments, centers)):
        values = np.asarray(segment, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0:
            continue
        endpoints = values[:, coordinate]
        errors[0, index] = max(0.0, float(center) - float(np.nanmin(endpoints)))
        errors[1, index] = max(0.0, float(np.nanmax(endpoints)) - float(center))
    return errors


def _display_indices(x: np.ndarray, y: np.ndarray, max_points: int) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(x) & (x > 0) & np.isfinite(y))
    if valid.size <= max_points:
        return valid
    selected = np.linspace(0, valid.size - 1, max_points, dtype=int)
    return valid[selected]


def redraw_counts_figure_data(
    source_json: str | Path,
    output_path: str | Path,
    *,
    max_gbm_points: int = 60,
) -> Path:
    """Redraw a folded-count diagnostic from its saved figure arrays."""
    import matplotlib.pyplot as plt

    from .publication_style import (
        DIAGNOSTIC_PANEL_FIGSIZE,
        save_publication_figure,
        style_diagnostic_figure,
    )

    payload = json.loads(Path(source_json).read_text(encoding="utf-8"))
    axes_payload = payload.get("axes", [])
    if len(axes_payload) < 2:
        raise ValueError("Folded-count export must contain count and residual axes")

    fig, (count_ax, residual_ax) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=DIAGNOSTIC_PANEL_FIGSIZE,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    count_payload, residual_payload = axes_payload[:2]
    count_lines = count_payload["lines"]
    count_collections = count_payload["collections"]
    residual_lines = residual_payload["lines"]
    residual_collections = residual_payload["collections"]
    plotted_x: list[np.ndarray] = []
    plotted_y: list[np.ndarray] = []

    dataset_count = min(len(count_lines) // 2, len(count_collections) // 2)
    for dataset_index in range(dataset_count):
        data_line = count_lines[2 * dataset_index]
        model_line = count_lines[2 * dataset_index + 1]
        name = str(model_line["label"]).removesuffix(" Model")
        color = _saved_color(model_line["color"])
        model_x = np.asarray(model_line["x"], dtype=float)
        model_y = np.asarray(model_line["y"], dtype=float)
        count_ax.plot(model_x, model_y, color=color, linewidth=1.0, label=f"{name} Model")

        data_x = np.asarray(data_line["x"], dtype=float)
        data_y = np.asarray(data_line["y"], dtype=float)
        limit = len(data_x) if name.upper() == "LAT" else int(max_gbm_points)
        indices = _display_indices(data_x, data_y, limit)
        xerr = _segment_errors(
            count_collections[2 * dataset_index]["segments"], data_x, 0
        )[:, indices]
        yerr = _segment_errors(
            count_collections[2 * dataset_index + 1]["segments"], data_y, 1
        )[:, indices]
        count_ax.errorbar(
            data_x[indices],
            data_y[indices],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=3.2 if name.upper() == "LAT" else 2.2,
            color=color,
            markeredgewidth=0,
            elinewidth=0.55,
            capsize=0,
            label=name,
        )
        plotted_x.extend((model_x, data_x[indices]))
        plotted_y.extend((model_y, data_y[indices]))

        residual_line = residual_lines[2 * dataset_index + 1]
        residual_x = np.asarray(residual_line["x"], dtype=float)
        residual_y = np.asarray(residual_line["y"], dtype=float)
        residual_indices = _display_indices(residual_x, residual_y, limit)
        residual_yerr = _segment_errors(
            residual_collections[dataset_index]["segments"], residual_y, 1
        )[:, residual_indices]
        residual_ax.errorbar(
            residual_x[residual_indices],
            residual_y[residual_indices],
            yerr=residual_yerr,
            fmt="o",
            ms=3.0 if name.upper() == "LAT" else 2.0,
            color=color,
            markeredgewidth=0,
            elinewidth=0.55,
            capsize=0,
        )

    finite_x = np.concatenate(plotted_x)
    finite_x = finite_x[np.isfinite(finite_x) & (finite_x > 0)]
    finite_y = np.concatenate(plotted_y)
    finite_y = finite_y[np.isfinite(finite_y) & (finite_y > 0)]
    count_ax.set_xscale("log")
    count_ax.set_yscale("log")
    count_ax.set_xlim(float(finite_x.min()) * 0.65, float(finite_x.max()) * 1.6)
    count_ax.set_ylim(float(finite_y.min()) * 0.5, float(finite_y.max()) * 2.0)
    count_ax.set_ylabel(count_payload["ylabel"])
    residual_ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    residual_ax.set_xscale("log")
    residual_ax.set_ylim(*residual_payload["ylim"])
    residual_ax.set_xlabel(residual_payload["xlabel"])
    residual_ax.set_ylabel(residual_payload["ylabel"])
    count_ax.legend()
    style_diagnostic_figure(fig)
    saved = save_publication_figure(fig, output_path, bbox_inches=None)
    plt.close(fig)
    return saved
