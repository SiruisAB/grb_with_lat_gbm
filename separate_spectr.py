from __future__ import annotations

from pyexpat import model

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

try:
    from .publication_style import (
        ANNOTATION_SIZE,
        AXIS_LABEL_SIZE,
        LEGEND_SIZE,
        MINOR_TICK_LABEL_SIZE,
        MANUSCRIPT_SPECTRUM_WIDTH_IN,
        SPECTRUM_PANEL_FIGSIZE,
        TICK_LABEL_SIZE,
        TITLE_SIZE,
        configure_publication_matplotlib,
        save_publication_figure,
        normalize_plot_style,
        source_font_size,
        style_publication_axes,
    )
except ImportError:  # pragma: no cover - direct script execution
    from publication_style import (
        ANNOTATION_SIZE,
        AXIS_LABEL_SIZE,
        LEGEND_SIZE,
        MINOR_TICK_LABEL_SIZE,
        MANUSCRIPT_SPECTRUM_WIDTH_IN,
        SPECTRUM_PANEL_FIGSIZE,
        TICK_LABEL_SIZE,
        TITLE_SIZE,
        configure_publication_matplotlib,
        save_publication_figure,
        normalize_plot_style,
        source_font_size,
        style_publication_axes,
    )

try:
    import spectres
except ImportError:  # pragma: no cover - optional dependency in redraw-only environments
    spectres = None

try:
    from astropy import units as u
except ImportError:  # pragma: no cover - optional dependency in redraw-only environments
    u = None

try:
    from astromodels import (
        Band,
        Blackbody,
        Cutoff_powerlaw,
        Gaussian,
        Log_uniform_prior,
        Model,
        PointSource,
        Powerlaw,
        Uniform_prior,
        NonDissipativePhotosphere,
    )
except ImportError:  # pragma: no cover - optional dependency in redraw-only environments
    Band = Blackbody = Cutoff_powerlaw = Gaussian = Log_uniform_prior = Model = PointSource = Powerlaw = Uniform_prior = NonDissipativePhotosphere = None

try:
    from threeML import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - optional dependency in redraw-only environments
    pass

try:
    from .modelbuild import MultiColorBlackBody, build_model
except ImportError:  # pragma: no cover - optional dependency in redraw-only environments
    MultiColorBlackBody = build_model = None

configure_publication_matplotlib()

PAPER_NAI_STYLES = (
    {"marker": "P", "color": "#4e79a7"},
    {"marker": "X", "color": "#f28e2b"},
)
PAPER_BGO_STYLE = {"marker": "s", "color": "#59a14f"}
PAPER_LAT_STYLE = {"marker": "v", "color": "#b07aa1"}

PAPER_MODEL_STYLES = {
    "total": {"color": "#1f3b73", "linewidth": 1.3},
    "band": {"color": "#c44e52", "linewidth": 1.1, "linestyle": ":"},
    "bb": {"color": "#55a868", "linewidth": 1.1, "linestyle": "--"},
}

ENERGY_TO_ERG = 1.602e-9

PAPER_LEGEND_STYLE = {
    "frameon": True,
    "framealpha": 0.92,
    "edgecolor": "#d0d0d0",
    "facecolor": "white",
    "handlelength": 1.3,
    "handletextpad": 0.5,
    "borderpad": 0.5,
    "labelspacing": 0.45,
}

BGO_REBIN_MAX_KEV = 4.0e4
BGO_REBIN_BIN_COUNT = 21
NAI_REBIN_BIN_COUNT = 15
NAI_REBIN_LOWER_BOUNDARY_KEV = 8.0
NAI_REBIN_UPPER_BOUNDARY_KEV = 900.0
DEFAULT_LAT_DISPLAY_BINS = 5
LAT_UPPER_LIMIT_SIGMA = 1.645

REFERENCE_SPECTRUM_FONT_PT = {
    "axis_label_pt": 24.0,
    "tick_label_pt": 18.0,
    "legend_pt": 18.0,
}
REFERENCE_SPECTRUM_TARGET_PT = {
    "axis_label_pt": 8.5,
    "tick_label_pt": 8.0,
    "legend_pt": 8.0,
}


def _spectrum_font_size(config: dict, key: str, native_width: float) -> float:
    if config["spectrum_font_mode"] == "reference":
        return (
            REFERENCE_SPECTRUM_FONT_PT[key]
            * float(config[key])
            / REFERENCE_SPECTRUM_TARGET_PT[key]
        )
    return source_font_size(
        config[key], native_width, MANUSCRIPT_SPECTRUM_WIDTH_IN
    )


def _build_log_rebin_axis(
    mid_points,
    bin_count: int,
    trim_channels: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid_points = np.asarray(mid_points, dtype=float)
    mid_points = mid_points[np.isfinite(mid_points) & (mid_points > 0)]
    if mid_points.size == 0:
        raise ValueError("Energy mid-points must contain positive finite values")
    mid_points = np.unique(mid_points)

    trim = max(0, int(trim_channels))
    if 2 * trim < mid_points.size - 1:
        mid_points = mid_points[trim : mid_points.size - trim]

    count = max(1, min(int(bin_count), mid_points.size))
    centers = np.geomspace(mid_points[0], mid_points[-1], count)
    if count == 1:
        return centers, centers * 0.2, centers * 0.2

    edges = np.empty(count + 1, dtype=float)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0] ** 2 / edges[1]
    edges[-1] = centers[-1] ** 2 / edges[-2]
    return centers, centers - edges[:-1], edges[1:] - centers


def _build_nai_rebin_axis(
    mid_points,
    bin_count: int = NAI_REBIN_BIN_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the legacy NaI display grid and fixed outer boundaries."""
    mid_points = np.asarray(mid_points, dtype=float)
    mid_points = np.unique(
        mid_points[np.isfinite(mid_points) & (mid_points > 0)]
    )
    if mid_points.size < 13:
        raise ValueError("NaI rebinning requires at least 13 valid energy channels")

    lower_center = float(mid_points[6])
    upper_center = float(mid_points[-6])
    if not (
        NAI_REBIN_LOWER_BOUNDARY_KEV < lower_center < upper_center
        < NAI_REBIN_UPPER_BOUNDARY_KEV
    ):
        raise ValueError(
            "NaI trimmed energy centers must lie inside the fixed 8-900 keV "
            "display boundaries"
        )

    centers = np.logspace(
        np.log10(lower_center),
        np.log10(upper_center),
        max(1, int(bin_count)),
    )
    full_grid = np.concatenate(
        (
            [NAI_REBIN_LOWER_BOUNDARY_KEV],
            centers,
            [NAI_REBIN_UPPER_BOUNDARY_KEV],
        )
    )
    err_pos = (full_grid[2:] - full_grid[1:-1]) / 2.0
    err_neg = (full_grid[1:-1] - full_grid[:-2]) / 2.0
    if np.any(err_pos <= 0) or np.any(err_neg <= 0):
        raise ValueError("NaI display grid must increase strictly")
    return centers, err_neg, err_pos


def _build_lat_rebin_axis(
    mid_points,
    bin_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LAT display grid with fixed outer boundaries (30 MeV to 100 GeV).

    外边界对所有暴一致，不随暴名变化；原先的 ``burst_name`` 形参从未被
    读取，留着会让人误以为存在逐暴特例（测试里就是这么写的）。
    """
    lower_boundary = 3e4
    upper_boundary = 1e8
    mid_points = np.asarray(mid_points, dtype=float)
    usable = np.unique(
        mid_points[
            np.isfinite(mid_points)
            & (mid_points > lower_boundary)
            & (mid_points < upper_boundary)
        ]
    )
    if usable.size == 0:
        raise ValueError(
            "LAT energy mid-points do not overlap the fixed display range "
            f"{lower_boundary:g}-{upper_boundary:g} keV"
        )

    count = max(1, min(int(bin_count), usable.size))
    centers = np.geomspace(usable[0], usable[-1], count)
    full_grid = np.concatenate(([lower_boundary], centers, [upper_boundary]))
    err_pos = (full_grid[2:] - full_grid[1:-1]) / 2.0
    err_neg = (full_grid[1:-1] - full_grid[:-2]) / 2.0
    if np.any(err_pos <= 0) or np.any(err_neg <= 0):
        raise ValueError("LAT display grid must increase strictly")
    return centers, err_neg, err_pos


def _cap_lat_display_uncertainty(
    flux: np.ndarray,
    uncertainty: np.ndarray,
) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    display = np.asarray(uncertainty, dtype=float).copy()
    cap_mask = (
        np.isfinite(flux)
        & (flux > 0)
        & np.isfinite(display)
        & (display > flux)
    )
    display[cap_mask] = flux[cap_mask] * 0.95
    return display


def _lat_upper_limit_values(
    flux: np.ndarray,
    uncertainty: np.ndarray,
    sigma: float = LAT_UPPER_LIMIT_SIGMA,
) -> np.ndarray:
    """Return Gaussian one-sided upper limits for finite non-positive bins."""
    flux = np.asarray(flux, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    limits = np.full(np.broadcast_shapes(flux.shape, uncertainty.shape), np.nan)
    valid = (
        np.isfinite(flux)
        & np.isfinite(uncertainty)
        & (flux <= 0)
        & (uncertainty > 0)
    )
    limits[valid] = np.maximum(flux[valid], 0.0) + float(sigma) * uncertainty[valid]
    return limits


def _plugin_energy_boundaries(plugin, channel_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-channel energy bounds without depending on a plugin subtype."""
    candidates = [getattr(plugin, "energy_boundaries", None)]
    observed = getattr(plugin, "_observed_spectrum", None)
    if observed is not None:
        energy_min = getattr(observed, "energy_min", None)
        energy_max = getattr(observed, "energy_max", None)
        if energy_min is not None and energy_max is not None:
            candidates.append((energy_min, energy_max))

    for boundaries in candidates:
        if boundaries is None:
            continue
        if callable(boundaries):
            boundaries = boundaries()
        try:
            if len(boundaries) != 2:
                continue
            lower = np.asarray(boundaries[0], dtype=float).reshape(-1)
            upper = np.asarray(boundaries[1], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if lower.size == channel_count and upper.size == channel_count:
            return lower, upper

    missing = np.full(channel_count, np.nan, dtype=float)
    return missing.copy(), missing.copy()


def _build_bgo_rebin_axis(energy_boundaries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(energy_boundaries) != 2:
        raise ValueError("BGO energy boundaries must contain lower and upper arrays")
    energy_min = np.asarray(energy_boundaries[0], dtype=float)
    energy_max = np.asarray(energy_boundaries[1], dtype=float)
    if energy_min.ndim != 1 or energy_max.ndim != 1 or energy_min.size == 0:
        raise ValueError("BGO energy boundary arrays must be non-empty and one-dimensional")
    if energy_min.size != energy_max.size or np.any(energy_max <= energy_min):
        raise ValueError("BGO energy boundary arrays are inconsistent")

    lower_energy = float(energy_min[0])
    upper_energy = min(float(energy_max[-1]), BGO_REBIN_MAX_KEV)
    if lower_energy <= 0 or upper_energy <= lower_energy:
        raise ValueError("BGO active energy range is invalid")

    edges = np.logspace(
        np.log10(lower_energy),
        np.log10(upper_energy),
        BGO_REBIN_BIN_COUNT + 1,
    )
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, centers - edges[:-1], edges[1:] - centers


def _style_publication_axes(ax):
    style_publication_axes(ax, show_all_spines=True)
    ax.tick_params(axis="both", which="both", top=True, right=True)


def _place_unfolded_spectrum_legend(fig, ax, plot_style=None) -> None:
    config = normalize_plot_style(plot_style)
    position = config["spectrum_legend_position"]
    native_width = float(fig.get_size_inches()[0])
    legend_size = _spectrum_font_size(config, "legend_pt", native_width)
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    entries = list(zip(handles, labels))
    entries.sort(
        key=lambda item: str(item[1]).startswith(("NaI", "BGO", "LAT"))
    )
    handles, labels = map(list, zip(*entries))
    ncol = min(config["spectrum_legend_columns"], len(labels))
    if ncol == 2 and len(labels) > 2 and len(labels) % 2:
        # matplotlib 图例按列填充，奇数条目时左列会多分一项；
        # 在左列末尾插入不可见占位项，使靠右的一列条目更多。
        blank = Line2D([], [], linestyle="none", marker="none")
        n_left = len(labels) // 2
        handles = handles[:n_left] + [blank] + handles[n_left:]
        labels = labels[:n_left] + [""] + labels[n_left:]
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    legend_kwargs = {
        "ncol": ncol,
        "fontsize": legend_size,
        "frameon": bool(config["spectrum_legend_frame"]) if position != "top" else False,
        "framealpha": 0.92,
        "edgecolor": "#d0d0d0",
        "facecolor": "white",
        "columnspacing": 0.9,
        "handlelength": 1.5,
        "handletextpad": 0.4,
        "labelspacing": 0.35,
    }
    if position == "top":
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.57, 0.985),
            **legend_kwargs,
        )
        fig.subplots_adjust(left=0.20, right=0.98, bottom=0.18, top=0.73)
    else:
        ax.legend(handles, labels, loc=position, **legend_kwargs)
        fig.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.98)


_FLUXDATA_RE = re.compile(
    r"^.+?_(nai_[^_]+|bgo_[^_]+|lat)_data_point_(.+)\.txt$"
)

_JSON_TIMEBIN_RE = re.compile(r"^bn[^_]+_bin_(?P<start>[0-9.]+)_(?P<end>[0-9.]+)$")


def _model_curve_path(fluxdata_dir: Path, model_name: str, timebin: str) -> Path:
    return Path(fluxdata_dir) / f"{model_name}_model_curve_{timebin}.txt"


def save_model_curve_data(
    path: Path,
    energy_kev: np.ndarray,
    curves: list[tuple[str, np.ndarray]],
) -> Path:
    """Save model curves in energy-flux units for later redraws."""
    energy_kev = np.asarray(energy_kev, dtype=float)
    if not curves:
        raise ValueError("At least one model curve is required")
    labels = [str(label) for label, _ in curves]
    columns = [energy_kev]
    for _, values in curves:
        values = np.asarray(values, dtype=float)
        if values.shape != energy_kev.shape:
            raise ValueError("Model curves must match the energy grid")
        columns.append(values)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "labels_json: " + json.dumps(labels) + "\n"
        + "energy_keV," + ",".join(
            "total" if idx == 0 else f"component_{idx}"
            for idx in range(len(curves))
        )
    )
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=header)
    return path


def load_model_curve_data(path: Path) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    path = Path(path)
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    prefix = "# labels_json: "
    if not first_line.startswith(prefix):
        raise ValueError(f"Missing model-curve labels in {path}")
    labels = json.loads(first_line[len(prefix) :])
    data = np.loadtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = np.atleast_2d(data)
    if data.shape[1] != len(labels) + 1:
        raise ValueError(f"Model-curve columns do not match labels in {path}")
    energy = np.asarray(data[:, 0], dtype=float)
    curves = [
        (str(label), np.asarray(data[:, idx + 1], dtype=float))
        for idx, label in enumerate(labels)
    ]
    return energy, curves


def discover_fluxdata_groups(fluxdata_dir: Path) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {}
    for path in sorted(Path(fluxdata_dir).glob("*.txt")):
        match = _FLUXDATA_RE.match(path.name)
        if not match:
            continue
        detector_tag, timebin = match.groups()
        groups.setdefault(timebin, {})[detector_tag] = path
    return groups


def _detector_plot_specs(detector_files: dict[str, Path]):
    specs = []
    nai_tags = sorted(tag for tag in detector_files if tag.startswith("nai_"))
    for idx, tag in enumerate(nai_tags):
        style = PAPER_NAI_STYLES[min(idx, len(PAPER_NAI_STYLES) - 1)]
        specs.append((tag, f"NaI ({tag.split('_', 1)[1]})", style))

    for tag in sorted(tag for tag in detector_files if tag.startswith("bgo_")):
        specs.append((tag, f"BGO ({tag.split('_', 1)[1]})", PAPER_BGO_STYLE))

    if "lat" in detector_files:
        specs.append(("lat", "LAT", PAPER_LAT_STYLE))
    return specs


def _read_fluxdata_file(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = np.atleast_2d(data)
    return data


def _normalize_timebin(timebin: str) -> str:
    text = str(timebin).strip()
    if "-" not in text:
        return text
    start_s, end_s = text.split("-", 1)
    return f"{float(start_s):g}-{float(end_s):g}"


def _find_json_bin_entry(json_payload: dict, bnname: str, timebin: str) -> tuple[str, dict]:
    burst_root = json_payload.get(bnname, {})
    bins = burst_root.get("bins", {})
    timebin_norm = _normalize_timebin(timebin)
    for bin_key, entry in bins.items():
        meta = entry.get("meta", {})
        if meta.get("time_bin_identifier") == timebin_norm:
            return bin_key, entry
        match = _JSON_TIMEBIN_RE.match(bin_key)
        if match:
            start = float(match.group("start"))
            end = float(match.group("end"))
            if _normalize_timebin(f"{start}-{end}") == timebin_norm:
                return bin_key, entry
            if f"{start:.2f}-{end:.2f}" == timebin_norm:
                return bin_key, entry
        if meta.get("bin_start_time") is not None and meta.get("bin_end_time") is not None:
            cand = _normalize_timebin(f"{float(meta['bin_start_time'])}-{float(meta['bin_end_time'])}")
            if cand == timebin_norm:
                return bin_key, entry
    raise KeyError(f"No fit entry found for {bnname} {timebin}")


def load_bandbb_fit_params(json_path: Path, bnname: str, timebin: str) -> dict[str, float]:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    _, entry = _find_json_bin_entry(payload, bnname, timebin)
    bandbb = entry.get("band+bb")
    if not bandbb:
        raise KeyError(f"No band+bb fit found for {bnname} {timebin}")
    return {
        "K_1": float(bandbb["GRB.spectrum.main.composite.K_1_value"]),
        "alpha_1": float(bandbb["GRB.spectrum.main.composite.alpha_1_value"]),
        "xp_1": float(bandbb["GRB.spectrum.main.composite.xp_1_value"]),
        "beta_1": float(bandbb["GRB.spectrum.main.composite.beta_1_value"]),
        "K_2": float(bandbb["GRB.spectrum.main.composite.K_2_value"]),
        "kT_2": float(bandbb["GRB.spectrum.main.composite.kT_2_value"]),
    }


def _energy_flux_display_mask(
    energy: np.ndarray,
    flux: np.ndarray,
    flux_err_neg: np.ndarray,
    flux_err_pos: np.ndarray,
) -> np.ndarray:
    energy = np.asarray(energy, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err_neg = np.asarray(flux_err_neg, dtype=float)
    flux_err_pos = np.asarray(flux_err_pos, dtype=float)
    return (
        np.isfinite(energy)
        & np.isfinite(flux)
        & np.isfinite(flux_err_neg)
        & np.isfinite(flux_err_pos)
        & (energy > 0)
        & (flux > 0)
        & (flux_err_neg >= 0)
        & (flux_err_pos >= 0)
        & (flux_err_neg < flux)
        & (flux_err_pos < flux)
    )


def _limit_display_points(data: np.ndarray, max_points: int | None) -> np.ndarray:
    valid = np.all(np.isfinite(data[:, :6]), axis=1) & _energy_flux_display_mask(
        data[:, 0], data[:, 3], data[:, 4], data[:, 5]
    )
    data = data[valid]
    if max_points is None or len(data) <= max_points:
        return data
    indices = np.unique(np.rint(np.linspace(0, len(data) - 1, max_points)).astype(int))
    return data[indices]


def _plot_fluxdata_detector(
    ax,
    path: Path,
    label: str,
    marker: str,
    color: str,
    max_points: int | None = None,
) -> np.ndarray:
    data = _read_fluxdata_file(path)
    data = _limit_display_points(data, max_points)
    if data.size == 0:
        return data
    x = data[:, 0]
    xerr = np.vstack((data[:, 1], data[:, 2]))
    y = data[:, 3]
    yerr = np.vstack((data[:, 4], data[:, 5]))
    ax.errorbar(
        x,
        y,
        xerr=np.abs(xerr),
        yerr=np.abs(yerr),
        fmt=marker,
        label=label,
        ms=4.5,
        color=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.55,
        elinewidth=0.75,
        capthick=0.75,
        capsize=1.8,
        alpha=0.95,
    )
    return data


def _matching_lat_upper_limit_path(lat_data_path: Path) -> Path:
    path = Path(lat_data_path)
    return path.with_name(
        path.name.replace("_lat_data_point_", "_lat_upper_limit_")
    )


def _plot_lat_upper_limits(
    ax,
    path: Path,
    color: str,
) -> np.ndarray:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 6), dtype=float)
    try:
        data = np.loadtxt(path)
    except ValueError:
        return np.empty((0, 6), dtype=float)
    if np.size(data) == 0:
        return np.empty((0, 6), dtype=float)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    if data.shape[1] < 4:
        raise ValueError(f"LAT upper-limit file has fewer than four columns: {path}")
    valid = (
        np.all(np.isfinite(data[:, :4]), axis=1)
        & (data[:, 0] > 0)
        & (data[:, 1] >= 0)
        & (data[:, 2] >= 0)
        & (data[:, 3] > 0)
    )
    data = data[valid]
    if data.size == 0:
        return data
    upper_limit = data[:, 3]
    ax.errorbar(
        data[:, 0],
        upper_limit,
        xerr=np.abs(np.vstack((data[:, 1], data[:, 2]))),
        yerr=upper_limit * 0.25,
        uplims=True,
        fmt="none",
        label="LAT 95% upper limit",
        color=color,
        elinewidth=0.85,
        capthick=0.85,
        capsize=2.2,
        alpha=0.95,
    )
    return data


def _format_timebin_title(bnname: str, timebin: str) -> str:
    return f"{bnname}_spectra_{timebin}"


def compute_bandbb_curves(
    xs: np.ndarray,
    K_1: float,
    alpha_1: float,
    xp_1: float,
    beta_1: float,
    K_2: float,
    kT_2: float,
) -> dict[str, np.ndarray]:
    xs = np.asarray(xs, dtype=float)
    band = np.empty_like(xs)
    ebreak = (alpha_1 - beta_1) * xp_1 / (2.0 + alpha_1)
    band_low = xs <= ebreak
    band_high = ~band_low
    band[band_low] = K_1 * (xs[band_low] / 100.0) ** alpha_1 * np.exp(-xs[band_low] * (2.0 + alpha_1) / xp_1)
    band[band_high] = (
        K_1
        * ((alpha_1 - beta_1) * xp_1 / (100.0 * (2.0 + alpha_1))) ** (alpha_1 - beta_1)
        * np.exp(beta_1 - alpha_1)
        * (xs[band_high] / 100.0) ** beta_1
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        bb = K_2 * xs**2 / np.expm1(xs / kT_2)
    bb = np.where(np.isfinite(bb), bb, 0.0)
    total = band + bb
    return {"total": total, "band": band, "bb": bb}


def _collect_errorbar_y_values(ax) -> np.ndarray:
    y_values: list[float] = []
    for line in ax.get_lines():
        marker = line.get_marker()
        linestyle = line.get_linestyle()
        if marker not in (None, "None", "") or linestyle in ("None", ""):
            y_values.extend(np.asarray(line.get_ydata(), dtype=float).ravel().tolist())

    for container in getattr(ax, "containers", []):
        lines = getattr(container, "lines", None)
        if not lines:
            continue
        data_line = lines[0] if len(lines) > 0 else None
        if data_line is not None:
            y_values.extend(np.asarray(data_line.get_ydata(), dtype=float).ravel().tolist())
        barlinecols = lines[2] if len(lines) > 2 else []
        for barcol in barlinecols or []:
            if not hasattr(barcol, "get_segments"):
                continue
            for seg in barcol.get_segments():
                if len(seg):
                    y_values.extend(np.asarray(seg[:, 1], dtype=float).ravel().tolist())

    y = np.asarray(y_values, dtype=float)
    return y[np.isfinite(y) & (y > 0)]


def _auto_tighten_log_ylim(ax, lower_pad: float = 0.65, upper_pad: float = 1.35) -> None:
    y = _collect_errorbar_y_values(ax)
    if y.size == 0:
        return
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min <= 0 or y_max <= 0:
        return
    if y_min == y_max:
        y_min *= 0.8
        y_max *= 1.25
    ax.set_ylim(y_min * lower_pad, y_max * upper_pad)


def _overlay_fit_curves(
    ax,
    timebin: str,
    bnname: str,
    fit_json_path: Path | None,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    if fit_json_path is None:
        return None
    try:
        params = load_bandbb_fit_params(fit_json_path, bnname=bnname, timebin=timebin)
    except KeyError:
        return None
    xs = np.logspace(np.log10(8.0), np.log10(1e8), 600)
    photon_curves = compute_bandbb_curves(xs, **params)
    energy_curves = {
        name: ENERGY_TO_ERG * xs**2 * values
        for name, values in photon_curves.items()
    }
    ax.plot(xs, energy_curves["total"], **PAPER_MODEL_STYLES["total"], label="Band+BB")
    ax.plot(xs, energy_curves["band"], **PAPER_MODEL_STYLES["band"], label="Band")
    ax.plot(xs, energy_curves["bb"], **PAPER_MODEL_STYLES["bb"], label="BB")
    return xs, [energy_curves["total"], energy_curves["band"], energy_curves["bb"]]


def _overlay_saved_model_curves(
    ax,
    curve_path: Path,
) -> tuple[np.ndarray, list[np.ndarray]]:
    energy, curves = load_model_curve_data(curve_path)
    plotted: list[np.ndarray] = []
    for idx, (label, values) in enumerate(curves):
        style_key = "total" if idx == 0 else str(label).lower()
        style = PAPER_MODEL_STYLES.get(
            style_key,
            {
                "color": f"C{idx - 1}",
                "linewidth": 1.1,
                "linestyle": (":", "--", "-.")[(idx - 1) % 3],
            },
        )
        valid = (
            np.isfinite(energy)
            & (energy > 0)
            & np.isfinite(values)
            & (values > 0)
        )
        if not np.any(valid):
            continue
        ax.plot(energy[valid], values[valid], label=label, **style)
        plotted.append(values[valid])
    return energy, plotted


def _set_redrawn_spectrum_limits(
    ax,
    x_values: list[np.ndarray],
    y_values: list[np.ndarray],
) -> None:
    finite_x = np.concatenate(x_values)
    finite_x = finite_x[np.isfinite(finite_x) & (finite_x > 0)]
    finite_y = np.concatenate(y_values)
    finite_y = finite_y[np.isfinite(finite_y) & (finite_y > 0)]
    if finite_x.size:
        ax.set_xlim(max(5.0, float(finite_x.min()) * 0.65), float(finite_x.max()) * 1.6)
    if finite_y.size:
        ax.set_ylim(float(finite_y.min()) * 0.08, float(finite_y.max()) * 12.0)


def redraw_fluxdata_timebin(
    timebin: str,
    fluxdata_dir: Path,
    output_dir: Path,
    bnname: str,
    fit_json_path: Path | None = None,
    show_lat_upper_limits: bool = True,
) -> Path:
    groups = discover_fluxdata_groups(fluxdata_dir)
    if timebin not in groups:
        raise FileNotFoundError(f"No fluxdata files found for time bin {timebin}")

    detector_files = groups[timebin]
    fig, ax = plt.subplots(figsize=SPECTRUM_PANEL_FIGSIZE)

    detector_specs = _detector_plot_specs(detector_files)
    plotted_any = False
    plotted_x: list[np.ndarray] = []
    plotted_y: list[np.ndarray] = []
    for detector_tag, label, style in detector_specs:
        path = detector_files.get(detector_tag)
        if path is None:
            continue
        detector_data = _plot_fluxdata_detector(
            ax,
            path,
            label=label,
            marker=style["marker"],
            color=style["color"],
            max_points=DEFAULT_LAT_DISPLAY_BINS if detector_tag == "lat" else None,
        )
        if detector_data.size:
            plotted_x.append(detector_data[:, 0])
            plotted_y.append(detector_data[:, 3])
            plotted_any = True
        if detector_tag == "lat" and show_lat_upper_limits:
            upper_data = _plot_lat_upper_limits(
                ax,
                _matching_lat_upper_limit_path(path),
                color=style["color"],
            )
            if upper_data.size:
                plotted_x.append(upper_data[:, 0])
                plotted_y.append(upper_data[:, 3])
                plotted_any = True

    if not plotted_any:
        raise FileNotFoundError(f"No recognized detector files found for time bin {timebin}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [keV]", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$E^{2} dN/dE$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=AXIS_LABEL_SIZE)
    curve_path = _model_curve_path(fluxdata_dir, "band+bb", timebin)
    if curve_path.exists():
        model_data = _overlay_saved_model_curves(ax, curve_path)
    else:
        model_data = _overlay_fit_curves(
            ax,
            timebin=timebin,
            bnname=bnname,
            fit_json_path=fit_json_path,
        )
        if model_data is not None:
            model_x, model_y = model_data
            save_model_curve_data(
                curve_path,
                model_x,
                list(zip(("Band+BB", "Band", "BB"), model_y)),
            )
    if model_data is not None:
        model_x, model_y = model_data
        plotted_x.append(model_x)
        if model_y:
            plotted_y.append(model_y[0])
    ax.minorticks_on()
    _style_publication_axes(ax)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="both", which="minor", labelsize=MINOR_TICK_LABEL_SIZE)
    _set_redrawn_spectrum_limits(ax, plotted_x, plotted_y)
    _place_unfolded_spectrum_legend(fig, ax)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bs_{bnname}_gbm_lat_spectra_band+bb_{timebin}.pdf"
    save_publication_figure(fig, out_path)
    plt.close(fig)
    return out_path


def redraw_saved_model_spectrum(
    *,
    timebin: str,
    fluxdata_dir: Path,
    model_sed_path: Path,
    output_path: Path,
    bnname: str,
    model_name: str,
    show_lat_upper_limits: bool = True,
) -> Path:
    groups = discover_fluxdata_groups(fluxdata_dir)
    if timebin not in groups:
        raise FileNotFoundError(f"No fluxdata files found for time bin {timebin}")

    fig, ax = plt.subplots(figsize=SPECTRUM_PANEL_FIGSIZE)
    plotted_y: list[np.ndarray] = []
    plotted_x: list[np.ndarray] = []
    for detector_tag, label, style in _detector_plot_specs(groups[timebin]):
        path = groups[timebin][detector_tag]
        data = _plot_fluxdata_detector(
            ax,
            path,
            label=label,
            marker=style["marker"],
            color=style["color"],
            max_points=DEFAULT_LAT_DISPLAY_BINS if detector_tag == "lat" else None,
        )
        if data.size:
            plotted_x.append(data[:, 0])
            plotted_y.append(data[:, 3])
        if detector_tag == "lat" and show_lat_upper_limits:
            upper_data = _plot_lat_upper_limits(
                ax,
                _matching_lat_upper_limit_path(path),
                color=style["color"],
            )
            if upper_data.size:
                plotted_x.append(upper_data[:, 0])
                plotted_y.append(upper_data[:, 3])

    sed = np.loadtxt(model_sed_path, delimiter=",", skiprows=1)
    energy = np.asarray(sed[:, 0], dtype=float)
    energy_flux = np.asarray(sed[:, 2], dtype=float)
    valid_model = (
        np.isfinite(energy)
        & (energy > 0)
        & np.isfinite(energy_flux)
        & (energy_flux > 0)
    )
    if not np.any(valid_model):
        raise ValueError(f"No finite model SED values found in {model_sed_path}")
    ax.plot(
        energy[valid_model],
        energy_flux[valid_model],
        color=PAPER_MODEL_STYLES["total"]["color"],
        linewidth=1.3,
        label=model_name,
        zorder=2,
    )
    plotted_x.append(energy[valid_model])
    plotted_y.append(energy_flux[valid_model])

    finite_x = np.concatenate(plotted_x)
    finite_x = finite_x[np.isfinite(finite_x) & (finite_x > 0)]
    finite_y = np.concatenate(plotted_y)
    finite_y = finite_y[np.isfinite(finite_y) & (finite_y > 0)]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(max(5.0, float(finite_x.min()) * 0.65), float(finite_x.max()) * 1.6)
    ax.set_ylim(float(finite_y.min()) * 0.35, float(finite_y.max()) * 3.0)
    ax.set_xlabel(r"$E$ [keV]", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(
        r"$E^{2} dN/dE$ [erg s$^{-1}$ cm$^{-2}$]",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.minorticks_on()
    _style_publication_axes(ax)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="both", which="minor", labelsize=MINOR_TICK_LABEL_SIZE)
    _place_unfolded_spectrum_legend(fig, ax)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, output_path)
    plt.close(fig)
    return output_path


def redraw_fluxdata_overview(
    groups: dict[str, dict[str, Path]],
    fluxdata_dir: Path,
    output_dir: Path,
    bnname: str,
    fit_json_path: Path | None = None,
    show_lat_upper_limits: bool = True,
) -> Path:
    timebins = list(groups)
    n_plots = len(timebins)
    if n_plots == 0:
        raise FileNotFoundError(f"No fluxdata groups found in {fluxdata_dir}")

    ncols = 2 if n_plots > 1 else 1
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(SPECTRUM_PANEL_FIGSIZE[0] * ncols, SPECTRUM_PANEL_FIGSIZE[1] * nrows),
        squeeze=False,
    )
    fig.set_facecolor("white")

    for idx, timebin in enumerate(timebins):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        detector_files = groups[timebin]
        for detector_tag, label, style in _detector_plot_specs(detector_files):
            path = detector_files.get(detector_tag)
            if path is None:
                continue
            _plot_fluxdata_detector(
                ax,
                path,
                label=label,
                marker=style["marker"],
                color=style["color"],
                max_points=DEFAULT_LAT_DISPLAY_BINS if detector_tag == "lat" else None,
            )
            if detector_tag == "lat" and show_lat_upper_limits:
                _plot_lat_upper_limits(
                    ax,
                    _matching_lat_upper_limit_path(path),
                    color=style["color"],
                )
        curve_path = _model_curve_path(fluxdata_dir, "band+bb", timebin)
        if curve_path.exists():
            _overlay_saved_model_curves(ax, curve_path)
        else:
            _overlay_fit_curves(
                ax,
                timebin=timebin,
                bnname=bnname,
                fit_json_path=fit_json_path,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(timebin, fontsize=TITLE_SIZE, pad=4)
        ax.minorticks_on()
        _style_publication_axes(ax)
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="both", which="minor", labelsize=MINOR_TICK_LABEL_SIZE)
        ax.set_xlim(8.0, 1.2e5)
        _auto_tighten_log_ylim(ax)
        ax.grid(which="major", alpha=0.12, linestyle="-")
        ax.grid(which="minor", alpha=0.05, linestyle=":")
        if row == nrows - 1:
            ax.set_xlabel("Energy [keV]", fontsize=AXIS_LABEL_SIZE)
        if col == 0:
            ax.set_ylabel(r"$E^{2} dN/dE$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=AXIS_LABEL_SIZE)
        ax.legend(loc="upper left", fontsize=LEGEND_SIZE, **PAPER_LEGEND_STYLE)

    for idx in range(n_plots, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    fig.suptitle(f"{bnname} fluxdata overview", y=0.995, fontsize=TITLE_SIZE, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bs_{bnname}_gbm_lat_spectra_band+bb_overview.pdf"
    save_publication_figure(fig, out_path)
    plt.close(fig)
    return out_path


def redraw_all_fluxdata_spectra(
    fluxdata_dir: Path,
    output_dir: Path,
    bnname: str,
    fit_json_path: Path | None = None,
    show_lat_upper_limits: bool = True,
) -> dict[str, list[Path]]:
    groups = discover_fluxdata_groups(fluxdata_dir)
    single_plots = [
        redraw_fluxdata_timebin(timebin, fluxdata_dir, output_dir, bnname, fit_json_path=fit_json_path, show_lat_upper_limits=show_lat_upper_limits)
        for timebin in groups
    ]
    overview = redraw_fluxdata_overview(groups, fluxdata_dir, output_dir, bnname, fit_json_path=fit_json_path, show_lat_upper_limits=show_lat_upper_limits)
    return {"single_plots": single_plots, "overview": [overview]}


def redraw_bandbb_fluxdata_outputs(
    bnname: str,
    result_root: Path,
    fit_json_path: Path | None = None,
    show_lat_upper_limits: bool = True,
) -> dict[str, list[Path]]:
    fluxdata_dir = Path(result_root) / bnname / "band+bb" / "fluxdata"
    return redraw_all_fluxdata_spectra(
        fluxdata_dir=fluxdata_dir,
        output_dir=fluxdata_dir,
        bnname=bnname,
        fit_json_path=fit_json_path,
        show_lat_upper_limits=show_lat_upper_limits,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redraw band+bb spectra from saved fluxdata files."
    )
    parser.add_argument("--bnname", required=True, help="Burst name, e.g. GRB231129C")
    parser.add_argument(
        "--result-root",
        required=True,
        type=Path,
        help="Root directory containing <bnname>/band+bb/fluxdata",
    )
    parser.add_argument(
        "--fluxdata-dir",
        type=Path,
        default=None,
        help="Optional explicit fluxdata directory. Overrides --result-root when set.",
    )
    parser.add_argument(
        "--save-result-dir",
        type=Path,
        default=None,
        help="Optional directory for PDF output. Defaults to the fluxdata directory.",
    )
    parser.add_argument(
        "--timebin",
        default=None,
        help="Redraw only one time bin, e.g. 0.1-1.",
    )
    parser.add_argument(
        "--fit-json",
        type=Path,
        default=None,
        help="Optional JSON file with best-fit band+bb parameters.",
    )
    parser.add_argument(
        "--show-lat-upper-limits",
        action="store_true",
        default=True,
        dest="show_lat_upper_limits",
        help="Draw LAT 95 percent upper-limit arrows (default: enabled).",
    )
    parser.add_argument(
        "--no-lat-upper-limits",
        action="store_false",
        dest="show_lat_upper_limits",
        help="Suppress LAT 95 percent upper-limit arrows.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to save a log file.",
    )
    return parser


def _resolve_fluxdata_dir(
    result_root: Path,
    bnname: str,
    fluxdata_dir: Optional[Path],
) -> Path:
    return fluxdata_dir if fluxdata_dir is not None else Path(result_root) / bnname / "band+bb" / "fluxdata"


def _resolve_output_dir(fluxdata_dir: Path, save_result_dir: Optional[Path]) -> Path:
    return save_result_dir if save_result_dir is not None else fluxdata_dir


def _setup_logger(log_file: Optional[Path]) -> logging.Logger:
    logger = logging.getLogger("grb_project.separate_spectr.redraw")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

    return logger


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    fluxdata_dir = _resolve_fluxdata_dir(args.result_root, args.bnname, args.fluxdata_dir)
    output_dir = _resolve_output_dir(fluxdata_dir, args.save_result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or (output_dir / "fluxdata_redraw.log")
    logger = _setup_logger(log_file)
    logger.info("Fluxdata directory: %s", fluxdata_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Fit JSON: %s", args.fit_json if args.fit_json else "<none>")
    logger.info("Log file: %s", log_file)

    if args.timebin:
        logger.info("Redrawing time bin: %s", args.timebin)
        out_path = redraw_fluxdata_timebin(
            timebin=args.timebin,
            fluxdata_dir=fluxdata_dir,
            output_dir=output_dir,
            bnname=args.bnname,
            fit_json_path=args.fit_json,
        )
        logger.info("Saved: %s", out_path)
    else:
        logger.info("Redrawing all fluxdata spectra")
        outputs = redraw_all_fluxdata_spectra(
            fluxdata_dir=fluxdata_dir,
            output_dir=output_dir,
            bnname=args.bnname,
            fit_json_path=args.fit_json,
        )
        for path in outputs["single_plots"]:
            logger.info("Saved: %s", path)
        for path in outputs["overview"]:
            logger.info("Saved: %s", path)

    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def discrete_spectr(
    fluence_plugins,
    lat,
    model_str,
    parameter_values,
    bs,
    result_dir,
    bnname,
    gbm_detectors,
    bin_start=None,
    bin_end=None,
    analysis_mode: str = "gbm+lat",
    output_dir: Optional[str] = None,
    plot_style: Optional[dict] = None,
):
    # log(f"{bnname}: 探测器 {gbm_detectors}")
    # if model_str  in ['pl','band','blackbody','comp','SBPL']:
    #     model1=build_model(model_str, parameter_values)
    #     modelTotal=model1
    #     model_str1=model_str
    if model_str == 'pl':
        model1 = Powerlaw(piv=1E2)
        # model1.K.min_value, model1.K.max_value = 1e-8, 1e3
        # model1.index.min_value, model1.index.max_value = -5.0, 0.0
        model1.K ,model1.index = parameter_values[:2]
        modelTotal=model1
        model_str1='PL'

    if model_str == 'blackbody':
        model1 = Blackbody()
        model1.K ,model1.kT = parameter_values[:2]
        modelTotal=model1
        model_str1= 'BB'

    if model_str == 'comp':
        model1 = Cutoff_powerlaw(piv=1E2)
        model1.K.min_value, model1.K.max_value = 1e-7, 1e6
        model1.index.min_value, model1.index.max_value = -10.0, 10.0
        model1.xc.min_value, model1.xc.max_value = 1e-99, 1e7
        model1.K ,model1.index,model1.xc = parameter_values[:3]
        modelTotal=model1
        model_str1='comp'

    if  model_str == 'band':
        model1 = Band(piv=1E2)
        # model1.alpha.min_value = -2.0    # 改硬边界
        # model1.alpha.max_value =  5.0
        # model1.xp.min_value,model1.xp.max_value = 1.0, 1e8
        model1.K ,model1.alpha,model1.xp ,model1.beta = parameter_values[:4]
        modelTotal=model1
        model_str1='Band'

    if model_str == 'comp+pl':

        model1 = Cutoff_powerlaw(piv=1E2)
        # model1.K.min_value, model1.K.max_value = 1e-7, 1e2
        # model1.index.min_value, model1.index.max_value = -5.0, 0.0
        # model1.xc.min_value, model1.xc.max_value = 1, 1e4
        model1.K,model1.index,model1.xc = parameter_values[:3]

        model2 = Powerlaw(piv=1E2)
        # model2.K.min_value, model2.K.max_value = 1e-7, 1e2
        # model2.index.min_value, model2.index.max_value = -5.0, 0.0
        model2.K, model2.index = parameter_values[3:5]
        modelTotal=model1+model2
        model_str1='comp'
        model_str2='PL'

    if model_str == 'NDP':
        model1 = NonDissipativePhotosphere(piv=1E2)
        model1.k, model1.ec = parameter_values[:2]
        modelTotal=model1
        model_str1='NDP'

    if model_str == 'mbb':
        # MBB: K, kT_min, kT_max, m（与 modelbuild.py 一致）
        model1 = MultiColorBlackBody()
        model1.K, model1.kT_min, model1.kT_max, model1.m = parameter_values[:4]
        modelTotal = model1
        model_str1 = 'mBB'

    if model_str == 'mbb+pl':
        # mBB + PL: K, kT_min, kT_max, m, K_pl, index（与 modelbuild.py 一致）
        model1 = MultiColorBlackBody()
        model1.K, model1.kT_min, model1.kT_max, model1.m = parameter_values[:4]
        model2 = Powerlaw(piv=1E2)
        model2.K, model2.index = parameter_values[4:6]
        modelTotal = model1 + model2
        model_str1 = 'mBB'
        model_str2 = 'PL'

    if model_str == 'band+bb':
        model1 = Band(K=parameter_values[0],alpha=parameter_values[1],xp=parameter_values[2],beta=parameter_values[3],piv=1E2)          # 先用默认值构造出来
        # model1.alpha.min_value = -5.0    # 改硬边界
        # model1.alpha.max_value =  3.0
        model1.xp.min_value,model1.xp.max_value = 1.0, 1e8
        # model1.k,model1.alpha,model1.xp,model1.beta  = parameter_values[:4]    

        model2 = Blackbody(K=parameter_values[4],kT=parameter_values[5])           # 返回 Blackbody 实例
        # model2.K.min_value, model2.K.max_value = 1e-9, 1e1
        # model2.kT.min_value, model2.kT.max_value = 0, 1e6
        # model2.k,model2.kT = parameter_values[4],parameter_values[5]
        modelTotal = model1 + model2
        model_str1 = 'Band'
        model_str2 = 'BB'

    if model_str == 'band+pl':
        model1 = Band(piv=1E2)                  # 先用默认值构造出来
        model1.alpha.min_value = -2.0    # 改硬边界
        model1.alpha.max_value =  5.0
        # model1.xp.min_value,model1.xp.max_value = 1.0, 1e8
        model1.K , model1.alpha , model1.xp , model1.beta  = parameter_values[:4]
        model2=Powerlaw(piv=1E2)
        # model2.K.min_value, model2.K.max_value = 1e-7, 1e6
        # model2.index.min_value, model2.index.max_value = -5.0, 5.0
        model2.K,model2.index = parameter_values[4],parameter_values[5]
        modelTotal=model1+model2
        model_str1='Band'
        model_str2='PL'
    
    if model_str == 'comp+bb':
        model1 = Cutoff_powerlaw(piv=1E2)
        # model1.K.min_value, model1.K.max_value = 1e-7, 1e2
        # model1.index.min_value, model1.index.max_value = -4.0, 0.0
        # model1.xc.min_value, model1.xc.max_value = 1.0, 1e4
        model1.K ,model1.index,model1.xc = parameter_values[:3]
        model2 = Blackbody()
        # model2.K.min_value, model2.K.max_value = 1e-6, 1
        # model2.kT.min_value, model2.kT.max_value = 0, 1e3
        model2.K ,model2.kT = parameter_values[3:5]
        modelTotal=model1+model2
        model_str1='comp'
        model_str2='BB'

    if model_str == 'pl+bb':
        model1=Blackbody(K=parameter_values[0],kT=parameter_values[1])
        model2=Powerlaw(K=parameter_values[2],index=parameter_values[3],piv=1E2)
        modelTotal=model1+model2
        model_str1='BB'
        model_str2='PL'

    if model_str == 'band+bb+pl':
        model1 = Band(piv=1E2)
        model1.K , model1.alpha , model1.xp , model1.beta  = parameter_values[:4]
        model2 = Blackbody()
        model2.K ,model2.kT = parameter_values[4:6]
        model3 = Powerlaw(piv=1E2)
        model3.K,model3.index = parameter_values[6:8]
        model3.K.min_value, model3.K.max_value = 1e-7, 1e6
        model3.index.min_value, model3.index.max_value = -5.0, 5.0
        modelTotal=model1+model2+model3
        model_str1='Band'
        model_str2='BB'
        model_str3='PL'
    
    if model_str == 'band+gauss':
        model1 = Band(piv=1E2)
        model1.K , model1.alpha , model1.xp , model1.beta  = parameter_values[:4]
        model2 = Gaussian()
        model2.F, model2.mu , model2.sigma = parameter_values[4:7]
        modelTotal=model1+model2
        model_str1='Band'
        model_str2='Gauss'

    conversion = u.Unit("keV2/(cm2 s keV)").to("erg/(cm2 s)")

    
    k0 = ENERGY_TO_ERG             ## convert keV to erg

    def _fmt_time(t):
        try:
            return f"{float(t):.2f}".rstrip("0").rstrip(".")
        except Exception:
            return str(t)

    time_bin_tag = (
        f"{_fmt_time(bin_start)}-{_fmt_time(bin_end)}"
        if bin_start is not None and bin_end is not None
        else None
    )
    time_bin_suffix = f"_{time_bin_tag}" if time_bin_tag else ""

    if output_dir is None:
        output_dir = result_dir
    os.makedirs(output_dir, exist_ok=True)

    style_config = normalize_plot_style(plot_style)
    spectrum_size = (
        style_config["spectrum_width_in"],
        style_config["spectrum_height_in"],
    )
    axis_label_size = _spectrum_font_size(
        style_config, "axis_label_pt", spectrum_size[0]
    )
    tick_label_size = _spectrum_font_size(
        style_config, "tick_label_pt", spectrum_size[0]
    )
    figModel = plt.figure(figsize=spectrum_size)
    # threeML plot_spectra(bs.results, ...) 已暂时关闭；若要与库自带 folded/总谱叠图，取消下面注释即可。
    # # 修复：bs可能是JointLikelihood或BayesianAnalysis，需要统一处理
    # try:
    #     if hasattr(bs, "results"):
    #         # 根据是否包含 LAT，调整谱图最高能量
    #         ene_max = 100 * u.GeV if "lat" in analysis_mode.lower() else 100 * u.MeV
    #         fig = plot_spectra(
    #             bs.results,
    #             ene_min=1 * u.keV,
    #             ene_max=ene_max,
    #             flux_unit="erg/(cm2 s)",
    #             use_components=True,
    #             components_to_use="total",
    #         )
    #     else:
    #         # 如果没有results属性，创建一个空图
    #         fig = plt.figure(figsize=(18, 12))
    # except Exception:
    #     # 如果plot_spectra失败，创建一个空图
    #     fig = plt.figure(figsize=(18, 12))
    nai_styles = (
        {"fmt": "P", "color": "#4e79a7"},
        {"fmt": "o", "color": "#f28e2b"},
    )
    nai_count = 0
    for detector_name, plugin in zip(gbm_detectors, fluence_plugins):
        detector_name = str(detector_name)
        is_bgo = detector_name.lower().startswith("b")
        if is_bgo:
            detector_kind = "bgo"
            label = f"BGO ({detector_name})"
            style = {"fmt": "s", "color": "#59a14f"}
        else:
            detector_kind = "nai"
            label = f"NaI ({detector_name})"
            style = nai_styles[min(nai_count, len(nai_styles) - 1)]
            nai_count += 1

        wavelength = np.asarray(plugin._observed_spectrum.mid_points, dtype=float)
        model_values = np.asarray(modelTotal(wavelength), dtype=float)
        expected_rate = np.asarray(plugin.expected_model_rate, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            area = expected_rate / model_values
            data_origin = np.asarray(plugin.source_rate, dtype=float) / area
            data_origin_err = np.asarray(plugin.source_rate_error, dtype=float) / area

        valid = (
            np.isfinite(wavelength)
            & (wavelength > 0)
            & np.isfinite(data_origin)
            & np.isfinite(data_origin_err)
        )
        wavelength = wavelength[valid]
        data_origin = data_origin[valid]
        data_origin_err = data_origin_err[valid]
        if wavelength.size == 0:
            continue

        if is_bgo:
            rebinned_wavelength, wavelength_err_neg, wavelength_err_pos = (
                _build_bgo_rebin_axis(plugin.energy_boundaries)
            )
            inside = (
                (rebinned_wavelength >= wavelength[0])
                & (rebinned_wavelength <= wavelength[-1])
            )
            rebinned_wavelength = rebinned_wavelength[inside]
            wavelength_err_neg = wavelength_err_neg[inside]
            wavelength_err_pos = wavelength_err_pos[inside]
        else:
            rebinned_wavelength, wavelength_err_neg, wavelength_err_pos = (
                _build_nai_rebin_axis(
                    wavelength,
                    bin_count=NAI_REBIN_BIN_COUNT,
                )
            )

        rebinned_flux, rebinned_flux_err = spectres.spectres(
            rebinned_wavelength,
            wavelength,
            data_origin,
            spec_errs=data_origin_err,
        )
        rebinned_flux = np.asarray(rebinned_flux, dtype=float)
        rebinned_flux_err_raw = np.asarray(rebinned_flux_err, dtype=float).copy()
        if is_bgo:
            rebinned_flux_err_display = rebinned_flux_err_raw.copy()
        else:
            rebinned_flux_err_display = _cap_lat_display_uncertainty(
                rebinned_flux,
                rebinned_flux_err_raw,
            )
        flux_err_pos = rebinned_flux_err_display.copy()
        flux_err_neg = rebinned_flux_err_display.copy()

        energy_flux = k0 * rebinned_wavelength**2 * rebinned_flux
        energy_flux_err_neg = k0 * rebinned_wavelength**2 * flux_err_neg
        energy_flux_err_pos = k0 * rebinned_wavelength**2 * flux_err_pos
        display_mask = _energy_flux_display_mask(
            rebinned_wavelength,
            energy_flux,
            energy_flux_err_neg,
            energy_flux_err_pos,
        )
        if np.any(display_mask):
            plt.errorbar(
                rebinned_wavelength[display_mask],
                energy_flux[display_mask],
                xerr=np.abs(
                    [
                        wavelength_err_neg[display_mask],
                        wavelength_err_pos[display_mask],
                    ]
                ),
                yerr=np.abs(
                    [
                        energy_flux_err_neg[display_mask],
                        energy_flux_err_pos[display_mask],
                    ]
                ),
                fmt=style["fmt"],
                label=label,
                ms=9.0 if is_bgo else 8.5,
                color=style["color"],
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.9,
                elinewidth=1.45,
                capthick=1.45,
                capsize=3,
            )

        text_file = os.path.join(
            output_dir,
            f"{model_str}_{detector_kind}_{detector_name}_data_point"
            f"{time_bin_suffix}.txt",
        )
        np.savetxt(
            text_file,
            np.column_stack(
                (
                    rebinned_wavelength,
                    wavelength_err_neg,
                    wavelength_err_pos,
                    energy_flux,
                    energy_flux_err_neg,
                    energy_flux_err_pos,
                )
            ),
            delimiter=" ",
        )

        raw_uncertainty_file = os.path.join(
            output_dir,
            f"{model_str}_{detector_kind}_{detector_name}_raw_uncertainty"
            f"{time_bin_suffix}.txt",
        )
        raw_energy_flux_err = (
            k0 * rebinned_wavelength**2 * rebinned_flux_err_raw
        )
        np.savetxt(
            raw_uncertainty_file,
            np.column_stack(
                (
                    rebinned_wavelength,
                    wavelength_err_neg,
                    wavelength_err_pos,
                    energy_flux,
                    raw_energy_flux_err,
                    raw_energy_flux_err,
                )
            ),
            delimiter=" ",
        )


    ###################  convert to origin data and rebin for LLE if has LLE data

    # if LLElabel !='no LLE':
    #     print '#### plot LLE data     ####'

    #     area = lat_lle.expected_model_rate/modelTotal(lat_lle._observed_spectrum.mid_points)
    #     dataOrigin = lat_lle.source_rate/area
    #     dataOriginErr = lat_lle.source_rate_error/area

    #     # select data x-range
    #     wavelength = lat_lle._observed_spectrum.mid_points
    #     rebinnedWavelength = np.logspace(np.log10(wavelength[4]),np.log10(wavelength[15]),5)
    #     #rebinnedWavelength = wavelength

    #     # rebin x-axis
    #     rebinnedWavelengthFull = np.insert(rebinnedWavelength, 0, 2e4)
    #     rebinnedWavelengthFull = np.append(rebinnedWavelengthFull, 1e5)
    #     rebinnedWavelengthErrPos = (rebinnedWavelengthFull[2:] - rebinnedWavelengthFull[1:-1])/2
    #     rebinnedWavelengthErrNeg = (rebinnedWavelengthFull[1:-1] - rebinnedWavelengthFull[:-2])/2
    #     # rebin y-axis
    #     rebinned = spectres.spectres(rebinnedWavelength, wavelength, dataOrigin, spec_errs=dataOriginErr)
    #     rebinnedFlux = rebinned[0]
    #     rebinnedFluxErrPos = rebinned[1]
    #     rebinnedFluxErrNeg = rebinned[1]
    #     rebinnedFluxErrNeg[rebinnedFluxErrNeg > rebinnedFlux] = rebinnedFlux[rebinnedFluxErrNeg > rebinnedFlux]*0.95

    #     #rebinnedWavelength[-1]=0

    #     plt.errorbar(rebinnedWavelength,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
    #                 xerr=[rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos], \
    #                 yerr=[k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos],marker='d',label='LLE',fmt='.',ms=12,color='#aaaaaa')

    #     text_file = model_str+'_lle_data_point.txt'
    #     data = np.column_stack((rebinnedWavelength, rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
    # k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos))
    #     np.savetxt(text_file, data, delimiter=' ')

    ###################  convert to origin data and rebin for LAT if has LAT data
    if "lat" in analysis_mode.lower() and lat is not None:
        raw_wavelength = np.asarray(
            lat._observed_spectrum.mid_points,
            dtype=float,
        )
        raw_model_flux = np.asarray(modelTotal(raw_wavelength), dtype=float)
        raw_expected_rate = np.asarray(lat.expected_model_rate, dtype=float)
        raw_source_rate = np.asarray(lat.source_rate, dtype=float)
        raw_source_rate_error = np.asarray(lat.source_rate_error, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_area = raw_expected_rate / raw_model_flux
            raw_data_origin = raw_source_rate / raw_area
            raw_data_origin_error = raw_source_rate_error / raw_area

        energy_min, energy_max = _plugin_energy_boundaries(
            lat,
            raw_wavelength.size,
        )
        raw_channels_file = os.path.join(
            output_dir,
            model_str + "_lat_raw_channels" + time_bin_suffix + ".txt",
        )
        np.savetxt(
            raw_channels_file,
            np.column_stack(
                (
                    raw_wavelength,
                    energy_min,
                    energy_max,
                    raw_source_rate,
                    raw_source_rate_error,
                    raw_expected_rate,
                    raw_model_flux,
                    raw_area,
                    raw_data_origin,
                    raw_data_origin_error,
                )
            ),
            delimiter=" ",
            header=(
                "energy_mid_keV energy_min_keV energy_max_keV source_rate "
                "source_rate_error expected_model_rate model_photon_flux "
                "effective_area unfolded_photon_flux "
                "unfolded_photon_flux_error"
            ),
        )

        wavelength = raw_wavelength
        valid = (
            np.isfinite(wavelength)
            & (wavelength > 0)
            & np.isfinite(raw_data_origin)
            & np.isfinite(raw_data_origin_error)
        )
        wavelength = wavelength[valid]
        dataOrigin = raw_data_origin[valid]
        dataOriginErr = raw_data_origin_error[valid]
        rebinnedWavelength, rebinnedWavelengthErrNeg, rebinnedWavelengthErrPos = (
            _build_lat_rebin_axis(
                wavelength,
                bin_count=style_config["lat_plot_bins"],
            )
        )
        # rebin y-axis
        rebinned = spectres.spectres(
            rebinnedWavelength,
            wavelength,
            dataOrigin,
            spec_errs=dataOriginErr,
        )
        rebinnedFlux = rebinned[0]
        rebinnedFluxErrRaw = np.asarray(rebinned[1], dtype=float).copy()
        rebinnedFluxErrDisplay = _cap_lat_display_uncertainty(
            rebinnedFlux, rebinnedFluxErrRaw
        )
        rebinnedFluxErrPos = rebinnedFluxErrDisplay
        rebinnedFluxErrNeg = rebinnedFluxErrDisplay
        lat_energy_flux = k0 * rebinnedWavelength**2 * rebinnedFlux
        raw_energy_flux_err = (
            k0 * rebinnedWavelength**2 * rebinnedFluxErrRaw
        )
        lat_energy_flux_err_neg = (
            k0 * rebinnedWavelength**2 * rebinnedFluxErrNeg
        )
        lat_energy_flux_err_pos = (
            k0 * rebinnedWavelength**2 * rebinnedFluxErrPos
        )
        display_mask = _energy_flux_display_mask(
            rebinnedWavelength,
            lat_energy_flux,
            lat_energy_flux_err_neg,
            lat_energy_flux_err_pos,
        )
        if np.any(display_mask):
            plt.errorbar(
                rebinnedWavelength[display_mask],
                lat_energy_flux[display_mask],
                xerr=np.abs(
                    [
                        rebinnedWavelengthErrNeg[display_mask],
                        rebinnedWavelengthErrPos[display_mask],
                    ]
                ),
                yerr=np.abs(
                    [
                        lat_energy_flux_err_neg[display_mask],
                        lat_energy_flux_err_pos[display_mask],
                    ]
                ),
                fmt="D",
                label="LAT",
                ms=9.5,
                color="#e15759",
                markerfacecolor="#e15759",
                markeredgecolor="white",
                markeredgewidth=0.9,
                elinewidth=1.45,
                capthick=1.45,
                capsize=3,
            )

        if style_config.get("show_lat_upper_limits", True):
            lat_upper_limits = _lat_upper_limit_values(
                lat_energy_flux,
                raw_energy_flux_err,
            )
            upper_limit_mask = np.isfinite(lat_upper_limits) & (lat_upper_limits > 0)
            if np.any(upper_limit_mask):
                plt.errorbar(
                    rebinnedWavelength[upper_limit_mask],
                    lat_upper_limits[upper_limit_mask],
                    xerr=np.abs(
                        [
                            rebinnedWavelengthErrNeg[upper_limit_mask],
                            rebinnedWavelengthErrPos[upper_limit_mask],
                        ]
                    ),
                    yerr=lat_upper_limits[upper_limit_mask] * 0.25,
                    uplims=True,
                    fmt="none",
                    label="LAT 95% upper limit",
                    color="#e15759",
                    elinewidth=1.3,
                    capthick=1.3,
                    capsize=3,
                )

            upper_limit_file = os.path.join(
                output_dir,
                model_str + "_lat_upper_limit" + time_bin_suffix + ".txt",
            )
            upper_limit_data = np.column_stack(
                (
                    rebinnedWavelength[upper_limit_mask],
                    rebinnedWavelengthErrNeg[upper_limit_mask],
                    rebinnedWavelengthErrPos[upper_limit_mask],
                    lat_upper_limits[upper_limit_mask],
                    lat_energy_flux[upper_limit_mask],
                    raw_energy_flux_err[upper_limit_mask],
                )
            )
            np.savetxt(
                upper_limit_file,
                upper_limit_data,
                delimiter=" ",
                header=(
                    "energy_keV xerr_neg_keV xerr_pos_keV upper_limit_95pct "
                    "measured_energy_flux raw_energy_flux_error"
                ),
            )

        text_file = os.path.join(
            output_dir,
            model_str + "_lat_data_point" + time_bin_suffix + ".txt",
        )
        data = np.column_stack(
            (
                rebinnedWavelength,
                rebinnedWavelengthErrNeg,
                rebinnedWavelengthErrPos,
                lat_energy_flux,
                lat_energy_flux_err_neg,
                lat_energy_flux_err_pos,
            )
        )
        np.savetxt(text_file, data, delimiter=" ")

        raw_text_file = os.path.join(
            output_dir,
            model_str + "_lat_raw_uncertainty" + time_bin_suffix + ".txt",
        )
        raw_data = np.column_stack(
            (
                rebinnedWavelength,
                rebinnedWavelengthErrNeg,
                rebinnedWavelengthErrPos,
                lat_energy_flux,
                raw_energy_flux_err,
                raw_energy_flux_err,
            )
        )
        np.savetxt(raw_text_file, raw_data, delimiter=" ")

        


    # 能量上限：GBM-only 用 1e5 keV（100 MeV），GBM+LAT 扩展到 1e8 keV（100 GeV）
    if "lat" in analysis_mode.lower() and lat is not None:
        emax = 1e8
    else:
        emax = 1e5
    xs = np.logspace(np.log10(8.),5.0,100)
    if emax > 1e5:
        # 仅在 GBM+LAT 模式下才需要向 100 MeV 以上延伸。GBM-only 时
        # emax 就是 1e5，np.logspace(5.0, 5.0, 100) 会退化成 100 个完全
        # 相同的点，凭空给曲线和落盘的模型曲线文件塞进 99 行重复采样。
        xs1 = np.logspace(5.0,np.log10(emax),100)
        xs=np.append(xs, xs1[1:])
    fluxPL = k0*xs*xs*modelTotal(xs)
    if model_str in ['pl','band','blackbody','comp','SBPL','NDP','mbb']:
        plt.loglog(xs,k0*xs*xs*modelTotal(xs),'-',linewidth=2,label=model_str1, color='b')
        if model_str in ['blackbody']:
            Epeak = 3.92*parameter_values[1]
            fluxEpeak = max(k0*xs*xs*modelTotal(xs))
            flux100mev = fluxPL[np.where( xs == 100000.0)]
        elif model_str in ['mbb']:
            kT_min, kT_max = parameter_values[1], parameter_values[2]
            Epeak = 3.92 * np.sqrt(kT_min * kT_max)
            fluxEpeak = max(k0*xs*xs*modelTotal(xs))
            flux100mev = fluxPL[np.where(xs == 100000.0)]
        elif model_str in ['pl']:
            Epeak = 0
            fluxEpeak = 0
            flux100mev = fluxPL[np.where( xs == 100000.0)]
        elif model_str in ['NDP']:
            # NDP (NonDissipativePhotosphere) 模型的特征能量使用 ec
            Epeak = parameter_values[1]
            fluxEpeak = max(k0*xs*xs*modelTotal(xs))
            flux100mev = fluxPL[np.where( xs == 100000.0)]

        # elif model_str in ['comp']:
        #     # 康普顿化模型的特征能量计算
        #     # comp 模型通常峰值在 Ep 附近
        #     Epeak = parameter_values[2]  # 直接用 Ep 作为峰值能量
        #     fluxEpeak = max(k0*xs*xs*modelTotal(xs))
        #     flux100mev = fluxPL[np.where( xs == 100000.0)]
        elif model_str in ['SBPL']:
            Epeak = parameter_values[3]  # E0作为参考峰值
            fluxEpeak = max(k0*xs*xs*modelTotal(xs))
            flux100mev = fluxPL[np.where( xs == 100000.0)]
        else:
            Epeak = (2+parameter_values[1])*parameter_values[2]
            fluxEpeak = max(k0*xs*xs*modelTotal(xs))
            flux100mev = fluxPL[np.where( xs == 100000.0)]

    else:
        # 对于组合模型，先分别计算各个组件的flux，然后相加
        # 这样可以确保模型计算正确
        if model_str in ['band+bb+pl']:
            # 三模型组合: Band + Blackbody + Powerlaw
            flux1 = k0*xs*xs*model1(xs)  # Band component
            flux2 = k0*xs*xs*model2(xs)  # Blackbody component
            flux3 = k0*xs*xs*model3(xs)  # Powerlaw component
            fluxTotal = flux1 + flux2 + flux3
            
            # 绘制总模型和各个组件
            plt.loglog(xs, fluxTotal, '-', linewidth=2, label=model_str1+'+'+model_str2+'+'+model_str3, color='b')
            plt.loglog(xs, flux1, ':', linewidth=1.5, label=model_str1, color='r')
            plt.loglog(xs, flux2, '--', linewidth=2, label=model_str2, color='g')
            plt.loglog(xs, flux3, '-.', linewidth=2, label=model_str3, color='m')
            
            # 计算峰值能量和通量
            Epeak = (2+parameter_values[1])*parameter_values[2]  # 基于Band模型参数计算
            fluxEpeak = max(flux1)  # 使用Band组件的峰值
            idx_100mev = np.argmin(np.abs(xs - 100000.0))
            flux100mev = fluxTotal[idx_100mev]  # 使用总模型的100MeV通量
        else:
            # 对于其他组合模型（双模型）
            flux1 = k0*xs*xs*model1(xs)
            flux2 = k0*xs*xs*model2(xs)
            # modelTotal应该是两个组件的和
            fluxTotal = flux1 + flux2
            
            # 绘制总模型和各个组件
            plt.loglog(xs, fluxTotal, '-', linewidth=2, label=model_str1+'+'+model_str2, color='b')
            plt.loglog(xs, flux1, ':', linewidth=1.5, label=model_str1, color='r')
            plt.loglog(xs, flux2, '--', linewidth=2, label=model_str2, color='g')

            
            if model_str in ['pl+bb']:
                Epeak = 3.92*parameter_values[1]

            elif model_str in ['comp+pl']:
                # cutoff power-law峰值，pl用于高能延拓
                Epeak = (2 + parameter_values[1]) * parameter_values[2]
                fluxEpeak = max(flux1)  # 使用comp组件的峰值
                idx_100mev = np.argmin(np.abs(xs - 100000.0))
                flux100mev = fluxTotal[idx_100mev]  # 使用总模型的100MeV通量

            elif model_str in ['band+gauss']:
                # Band + Gaussian 组合模型
                Epeak = (2+parameter_values[1])*parameter_values[2]  # 基于Band模型参数计算
                fluxEpeak = max(flux1)  # 使用Band组件的峰值
                idx_100mev = np.argmin(np.abs(xs - 100000.0))
                flux100mev = fluxTotal[idx_100mev]  # 使用总模型的100MeV通量
            
            elif model_str in ['band+bb']:
                # 查找最接近 100 MeV 的点
                Epeak = (2+parameter_values[1])*parameter_values[2]
                # idx_100mev = np.argmin(np.abs(xs - 100000.0))
                # flux100mev = fluxPL[idx_100mev]
                fluxEpeak = max(k0*xs*xs*model1(xs))
                fluxPL = k0*xs*xs*model2(xs)
                flux100mev = fluxPL[np.where( xs == 100000.0)]
            elif model_str in ['band+pl']:
                # Band + Power-law 组合模型
                Epeak = (2 + parameter_values[1]) * parameter_values[2]
                fluxEpeak = max(k0 * xs * xs * model1(xs))  # 使用Band组件的峰值
                fluxPL = k0 * xs * xs * model2(xs)
                idx_100mev = np.argmin(np.abs(xs - 100000.0))
                flux100mev = fluxPL[idx_100mev]  # 使用pl组件在100 MeV的通量
            elif model_str in ['mbb+pl']:
                kT_min, kT_max = parameter_values[1], parameter_values[2]
                Epeak = 3.92 * np.sqrt(kT_min * kT_max)
                fluxEpeak = max(flux1)
                fluxPL = k0 * xs * xs * model2(xs)
                idx_100mev = np.argmin(np.abs(xs - 100000.0))
                flux100mev = fluxPL[idx_100mev]
            else:
                Epeak = (2+parameter_values[1])*parameter_values[2]
                fluxEpeak = max(k0*xs*xs*model1(xs))
                fluxPL = k0*xs*xs*model2(xs)
                flux100mev = fluxPL[np.where( xs == 100000.0)]

    plt.xlim([5e0, emax*2.0])

    if model_str == 'SBPL':
        # 在转折能量附近多采样一些点以更好地显示转折特征
        xs_around_E0 = np.logspace(np.log10(parameter_values[3]*0.1), 
                                    np.log10(parameter_values[3]*10), 100)
        xs = np.sort(np.append(xs, xs_around_E0))

    plt.ylim([k0*min(xs*xs*modelTotal(xs))*0.1, k0*max(xs*xs*modelTotal(xs))*500.0])
    if model_str == 'pl':
        plt.ylim([k0*min(xs*xs*modelTotal(xs))*0.1, k0*max(xs*xs*modelTotal(xs))*500.0])
    if model_str == 'band':
        plt.ylim([k0*min(xs*xs*modelTotal(xs))*0.1, k0*max(xs*xs*modelTotal(xs))*500.0])
    if model_str == 'comp':
        xc=parameter_values[2]
        plt.ylim([k0*xc*xc*modelTotal(xc)*0.005, k0*max(xs*xs*modelTotal(xs))*600.0])
    if model_str == 'NDP':
        ec = parameter_values[1]
        plt.ylim([k0*ec*ec*modelTotal(ec)*0.005, k0*max(xs*xs*modelTotal(xs))*600.0])
    if model_str == 'blackbody':
        kT = parameter_values[1]
        plt.ylim([k0*kT*kT*modelTotal(kT)*0.005, k0*max(xs*xs*modelTotal(xs))*600.0])
    if model_str == 'mbb':
        kT_min, kT_max = parameter_values[1], parameter_values[2]
        kT_ref = np.sqrt(kT_min * kT_max)
        plt.ylim(
            [
                k0 * kT_ref * kT_ref * modelTotal(kT_ref) * 0.005,
                k0 * max(xs * xs * modelTotal(xs)) * 600.0,
            ]
        )
    if model_str in ['mbb+pl']:
        kT_min, kT_max = parameter_values[1], parameter_values[2]
        kT_ref = np.sqrt(kT_min * kT_max)
        plt.ylim(
            [
                k0 * kT_ref * kT_ref * modelTotal(kT_ref) * 0.005,
                k0 * max(xs * xs * modelTotal(xs)) * 600.0,
            ]
        )
    if model_str in ['band+bb']:
        plt.ylim([k0*min(xs*xs*modelTotal(xs))*0.05, 
                  k0*max(xs*xs*modelTotal(xs))*600.0])

    if model_str == 'comp+bb':
        plt.ylim(1e-10, 1e-5)
        # plt.ylim([k0*min(xs*xs*modelTotal(xs))*0.01, 
        #           k0*max(xs*xs*modelTotal(xs))*100.0])

    if model_str == 'SBPL':
        # 在转折能量附近多采样一些点以更好地显示转折特征
        xs_around_E0 = np.logspace(np.log10(parameter_values[3]*0.1), 
                                    np.log10(parameter_values[3]*10), 100)
        xs = np.sort(np.append(xs, xs_around_E0))
        
        # 自适应设置 y 轴范围
        y_values = k0*xs*xs*modelTotal(xs)
        y_min = np.min(y_values[y_values > 0])
        y_max = np.max(y_values)
        plt.ylim([y_min*0.7, y_max*100.0])

    component_models = [(model_str1, model1)]
    if "model2" in locals():
        component_models.append((model_str2, model2))
    if "model3" in locals():
        component_models.append((model_str3, model3))
    total_label = "+".join(label for label, _ in component_models)
    saved_curves = [(total_label, k0 * xs**2 * modelTotal(xs))]
    if len(component_models) > 1:
        saved_curves.extend(
            (label, k0 * xs**2 * component(xs))
            for label, component in component_models
        )
    save_model_curve_data(
        _model_curve_path(output_dir, model_str, time_bin_tag or "full"),
        xs,
        saved_curves,
    )

    ax = plt.gca()
    ax.set_xlabel('$E$ [keV]', fontsize=axis_label_size)
    ax.set_ylabel(
        '$E^{2} dN/dE$ [erg s$^{-1}$cm$^{-2}$]',
        fontsize=axis_label_size,
    )
    # ax.set_title(f"{bnname}_spectra{time_bin_suffix}")
    ax.minorticks_on()
    style_publication_axes(
        ax,
        label_size=axis_label_size,
        tick_size=tick_label_size,
        minor_tick_size=tick_label_size,
        show_all_spines=True,
    )
    ax.tick_params(axis="both", which="both", top=True, right=True)
    _place_unfolded_spectrum_legend(figModel, ax, plot_style)
    suffix = "gbm_lat" if "lat" in analysis_mode.lower() and lat is not None else "gbm"
    out_name = f"bs_{bnname}_{suffix}_spectra_{model_str}{time_bin_suffix}.pdf"
    save_publication_figure(figModel, os.path.join(output_dir, out_name))
