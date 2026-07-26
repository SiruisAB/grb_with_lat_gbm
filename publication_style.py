"""Shared matplotlib settings for manuscript figures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import matplotlib as mpl
from matplotlib.legend_handler import HandlerTuple

FULL_WIDTH_IN = 7.10
LIGHTCURVE_FIGSIZE = (FULL_WIDTH_IN, 8.40)
SPECTRUM_PANEL_FIGSIZE = (3.35, 3.00)
MANUSCRIPT_BODY_FONT_PT = 10.0
MANUSCRIPT_SPECTRUM_WIDTH_IN = 0.327 * FULL_WIDTH_IN
MANUSCRIPT_LIGHTCURVE_WIDTH_IN = 0.48 * FULL_WIDTH_IN
MANUSCRIPT_DIAGNOSTIC_WIDTH_IN = 0.48 * FULL_WIDTH_IN
# Generate every manuscript panel close to its final physical size. This keeps
# typography stable and avoids shrinking slide-sized threeML canvases in LaTeX.
DIAGNOSTIC_PANEL_FIGSIZE = (5.0, 3.80)
CORNER_PANEL_FIGSIZE = (5.0, 5.0)

TARGET_AXIS_LABEL_PT = 8.5
TARGET_TICK_LABEL_PT = 8.0
TARGET_LEGEND_PT = 8.0

DEFAULT_PLOT_STYLE = {
    "spectrum_width_in": 12.0,
    "spectrum_height_in": 8.0,
    "diagnostic_width_in": DIAGNOSTIC_PANEL_FIGSIZE[0],
    "diagnostic_height_in": DIAGNOSTIC_PANEL_FIGSIZE[1],
    "corner_width_in": CORNER_PANEL_FIGSIZE[0],
    "corner_height_in": CORNER_PANEL_FIGSIZE[1],
    "axis_label_pt": TARGET_AXIS_LABEL_PT,
    "tick_label_pt": TARGET_TICK_LABEL_PT,
    "legend_pt": TARGET_LEGEND_PT,
    "legend_position": "top",
    "diagnostic_legend_position": "top",
    "spectrum_legend_position": "upper left",
    "spectrum_legend_frame": True,
    "spectrum_font_mode": "reference",
    "diagnostic_legend_columns": 4,
    "spectrum_legend_columns": 2,
    "lat_plot_bins": 5,
    "show_lat_upper_limits": True,
}

LEGEND_POSITIONS = {
    "top",
    "upper right",
    "upper left",
    "lower right",
    "lower left",
}


def normalize_plot_style(style: Optional[Mapping] = None) -> dict:
    normalized = dict(DEFAULT_PLOT_STYLE)
    if style:
        provided = {key: value for key, value in style.items() if value is not None}
        if "legend_position" in provided:
            provided.setdefault("diagnostic_legend_position", provided["legend_position"])
            provided.setdefault("spectrum_legend_position", provided["legend_position"])
        if "legend_columns" in provided:
            provided.setdefault("diagnostic_legend_columns", provided["legend_columns"])
            provided.setdefault("spectrum_legend_columns", provided["legend_columns"])
        normalized.update(provided)
    for key in (
        "spectrum_width_in",
        "spectrum_height_in",
        "diagnostic_width_in",
        "diagnostic_height_in",
        "corner_width_in",
        "corner_height_in",
        "axis_label_pt",
        "tick_label_pt",
        "legend_pt",
    ):
        normalized[key] = max(1.0, float(normalized[key]))
    normalized["diagnostic_legend_columns"] = max(
        1, int(normalized["diagnostic_legend_columns"])
    )
    normalized["spectrum_legend_columns"] = max(
        1, int(normalized["spectrum_legend_columns"])
    )
    normalized["lat_plot_bins"] = max(2, int(normalized["lat_plot_bins"]))
    normalized["show_lat_upper_limits"] = bool(normalized.get("show_lat_upper_limits", True))
    for key, fallback in (
        ("legend_position", "top"),
        ("diagnostic_legend_position", "top"),
        ("spectrum_legend_position", "upper left"),
    ):
        position = str(normalized[key]).strip().lower()
        normalized[key] = position if position in LEGEND_POSITIONS else fallback
    normalized["spectrum_legend_frame"] = bool(normalized["spectrum_legend_frame"])
    font_mode = str(normalized["spectrum_font_mode"]).strip().lower()
    normalized["spectrum_font_mode"] = (
        font_mode if font_mode in {"reference", "manuscript"} else "reference"
    )
    return normalized


def source_font_size(target_pt: float, native_width_in: float, placed_width_in: float) -> float:
    return _source_size(float(target_pt), float(native_width_in), float(placed_width_in))


def _source_size(target_pt: float, native_width_in: float, placed_width_in: float) -> float:
    return target_pt * native_width_in / placed_width_in


FONT_SIZE = 11.0
AXIS_LABEL_SIZE = _source_size(TARGET_AXIS_LABEL_PT, SPECTRUM_PANEL_FIGSIZE[0], MANUSCRIPT_SPECTRUM_WIDTH_IN)
TITLE_SIZE = _source_size(TARGET_TICK_LABEL_PT, SPECTRUM_PANEL_FIGSIZE[0], MANUSCRIPT_SPECTRUM_WIDTH_IN)
TICK_LABEL_SIZE = _source_size(TARGET_TICK_LABEL_PT, SPECTRUM_PANEL_FIGSIZE[0], MANUSCRIPT_SPECTRUM_WIDTH_IN)
MINOR_TICK_LABEL_SIZE = TICK_LABEL_SIZE
LEGEND_SIZE = _source_size(TARGET_LEGEND_PT, SPECTRUM_PANEL_FIGSIZE[0], MANUSCRIPT_SPECTRUM_WIDTH_IN)
ANNOTATION_SIZE = LEGEND_SIZE
LIGHTCURVE_AXIS_LABEL_SIZE = _source_size(TARGET_AXIS_LABEL_PT, LIGHTCURVE_FIGSIZE[0], MANUSCRIPT_LIGHTCURVE_WIDTH_IN)
LIGHTCURVE_TICK_LABEL_SIZE = _source_size(TARGET_TICK_LABEL_PT, LIGHTCURVE_FIGSIZE[0], MANUSCRIPT_LIGHTCURVE_WIDTH_IN)
LIGHTCURVE_LEGEND_SIZE = _source_size(TARGET_LEGEND_PT, LIGHTCURVE_FIGSIZE[0], MANUSCRIPT_LIGHTCURVE_WIDTH_IN)
LIGHTCURVE_ANNOTATION_SIZE = LIGHTCURVE_LEGEND_SIZE
REFERENCE_AXIS_LABEL_SIZE = _source_size(TARGET_AXIS_LABEL_PT, DIAGNOSTIC_PANEL_FIGSIZE[0], MANUSCRIPT_DIAGNOSTIC_WIDTH_IN)
REFERENCE_TICK_SIZE = _source_size(TARGET_TICK_LABEL_PT, DIAGNOSTIC_PANEL_FIGSIZE[0], MANUSCRIPT_DIAGNOSTIC_WIDTH_IN)
REFERENCE_LEGEND_SIZE = _source_size(TARGET_LEGEND_PT, DIAGNOSTIC_PANEL_FIGSIZE[0], MANUSCRIPT_DIAGNOSTIC_WIDTH_IN)
CORNER_LABEL_SIZE = REFERENCE_AXIS_LABEL_SIZE
CORNER_TICK_SIZE = REFERENCE_TICK_SIZE
CORNER_TITLE_SIZE = 0.85 * REFERENCE_LEGEND_SIZE

AXES_LINEWIDTH = 0.8
MAJOR_TICK_WIDTH = 0.8
MINOR_TICK_WIDTH = 0.6
MAJOR_TICK_LENGTH = 3.5
MINOR_TICK_LENGTH = 2.0
RASTER_DPI = 300


def configure_publication_matplotlib() -> None:
    """Install the project-wide typography and export defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FONT_SIZE,
            "mathtext.fontset": "dejavusans",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.linewidth": AXES_LINEWIDTH,
            "lines.linewidth": 1.0,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "xtick.major.size": MAJOR_TICK_LENGTH,
            "ytick.major.size": MAJOR_TICK_LENGTH,
            "xtick.major.width": MAJOR_TICK_WIDTH,
            "ytick.major.width": MAJOR_TICK_WIDTH,
            "xtick.minor.size": MINOR_TICK_LENGTH,
            "ytick.minor.size": MINOR_TICK_LENGTH,
            "xtick.minor.width": MINOR_TICK_WIDTH,
            "ytick.minor.width": MINOR_TICK_WIDTH,
            "legend.fontsize": LEGEND_SIZE,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": RASTER_DPI,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_publication_axes(
    ax,
    *,
    label_size: float = AXIS_LABEL_SIZE,
    tick_size: float = TICK_LABEL_SIZE,
    minor_tick_size: float = MINOR_TICK_LABEL_SIZE,
    show_all_spines: bool = False,
) -> None:
    """Apply consistent labels, ticks, and spine weights to one axes."""
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.title.set_fontsize(TITLE_SIZE)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=False,
        right=False,
        length=MAJOR_TICK_LENGTH,
        width=MAJOR_TICK_WIDTH,
        labelsize=tick_size,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=False,
        right=False,
        length=MINOR_TICK_LENGTH,
        width=MINOR_TICK_WIDTH,
        labelsize=minor_tick_size,
    )
    for name, spine in ax.spines.items():
        spine.set_linewidth(AXES_LINEWIDTH)
        spine.set_visible(show_all_spines or name in {"left", "bottom"})


def _style_legend(ax, font_size: float) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontsize(font_size)
    legend.get_frame().set_linewidth(0.6)


def _combined_fit_data_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    grouped: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for handle, raw_label in zip(handles, labels):
        label = str(raw_label).strip()
        if label.endswith(" Model"):
            name, kind = label.removesuffix(" Model"), "fit"
        elif label.endswith(" fit"):
            name, kind = label.removesuffix(" fit"), "fit"
        else:
            name, kind = label, "data"
        if name not in grouped:
            grouped[name] = {}
            order.append(name)
        grouped[name][kind] = handle

    combined_handles = []
    combined_labels = []
    for name in order:
        fit_handle = grouped[name].get("fit")
        data_handle = grouped[name].get("data")
        if fit_handle is not None and data_handle is not None:
            combined_handles.append((fit_handle, data_handle))
        else:
            combined_handles.append(fit_handle or data_handle)
        combined_labels.append(name)
    return combined_handles, combined_labels


def style_diagnostic_figure(fig, plot_style: Optional[Mapping] = None) -> None:
    """Normalize figures returned by threeML count/SED plotting helpers."""
    config = normalize_plot_style(plot_style)
    figure_size = (
        config["diagnostic_width_in"],
        config["diagnostic_height_in"],
    )
    label_size = source_font_size(
        config["axis_label_pt"], figure_size[0], MANUSCRIPT_DIAGNOSTIC_WIDTH_IN
    )
    tick_size = source_font_size(
        config["tick_label_pt"], figure_size[0], MANUSCRIPT_DIAGNOSTIC_WIDTH_IN
    )
    legend_size = source_font_size(
        config["legend_pt"], figure_size[0], MANUSCRIPT_DIAGNOSTIC_WIDTH_IN
    )
    fig.set_size_inches(*figure_size, forward=True)
    for ax in fig.axes:
        xlabel = ax.get_xlabel()
        if "\n" in xlabel:
            ax.xaxis.label.set_text(xlabel.replace("\n", " "))
        style_publication_axes(
            ax,
            label_size=label_size,
            tick_size=tick_size,
            minor_tick_size=tick_size,
            show_all_spines=True,
        )
        _style_legend(ax, legend_size)
    external_legend = False
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            handles, labels = _combined_fit_data_legend(ax)
            legend.remove()
            legend_kwargs = {
                "handler_map": {tuple: HandlerTuple(ndivide=None)},
                "ncol": min(config["diagnostic_legend_columns"], len(labels)),
                "fontsize": legend_size,
                "frameon": False,
                "columnspacing": 1.0,
                "handlelength": 2.2,
                "handletextpad": 0.5,
            }
            if config["diagnostic_legend_position"] == "top":
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.575, 0.985),
                    **legend_kwargs,
                )
                external_legend = True
            else:
                ax.legend(
                    handles,
                    labels,
                    loc=config["diagnostic_legend_position"],
                    **legend_kwargs,
                )
    fig.subplots_adjust(
        left=0.17,
        right=0.98,
        bottom=0.21,
        top=0.81 if external_legend else 0.97,
        hspace=0.10,
    )


def style_lightcurve_axes(ax, *, show_all_spines: bool = True) -> None:
    """Apply source sizes that become 8--9 pt after manuscript scaling."""
    style_publication_axes(
        ax,
        label_size=LIGHTCURVE_AXIS_LABEL_SIZE,
        tick_size=LIGHTCURVE_TICK_LABEL_SIZE,
        minor_tick_size=LIGHTCURVE_TICK_LABEL_SIZE,
        show_all_spines=show_all_spines,
    )
    _style_legend(ax, LIGHTCURVE_LEGEND_SIZE)


def style_corner_figure(fig, plot_style: Optional[Mapping] = None) -> None:
    """Normalize a posterior matrix for a half-page-width appendix panel."""
    config = normalize_plot_style(plot_style)
    figure_size = (config["corner_width_in"], config["corner_height_in"])
    label_size = source_font_size(
        config["axis_label_pt"], figure_size[0], MANUSCRIPT_DIAGNOSTIC_WIDTH_IN
    )
    tick_size = source_font_size(
        config["tick_label_pt"], figure_size[0], MANUSCRIPT_DIAGNOSTIC_WIDTH_IN
    )
    title_size = 0.85 * source_font_size(
        config["legend_pt"], figure_size[0], MANUSCRIPT_DIAGNOSTIC_WIDTH_IN
    )
    fig.set_size_inches(*figure_size, forward=True)
    label_replacements = {
        "K_1": r"$K_{\rm B}$",
        "alpha_1": r"$\alpha$",
        "xp_1": r"$E_{p}$",
        "beta_1": r"$\beta$",
        "K_2": r"$K_{\rm BB}$",
        "kT_2": r"$kT$",
    }
    for ax in fig.axes:
        style_publication_axes(
            ax,
            label_size=label_size,
            tick_size=tick_size,
            minor_tick_size=tick_size,
        )
        ax.tick_params(axis="x", which="major", labelrotation=55)
        ax.tick_params(axis="y", which="major", labelrotation=0)
        if ax.get_xlabel() in label_replacements:
            ax.set_xlabel(label_replacements[ax.get_xlabel()])
        if ax.get_ylabel() in label_replacements:
            ax.set_ylabel(label_replacements[ax.get_ylabel()])
        if ax.get_xlabel():
            ax.set_xlabel("")
        if ax.get_ylabel():
            ax.set_ylabel("")
        title = ax.get_title()
        if " = " in title:
            name, value = title.split(" = ", 1)
            ax.set_title(f"{label_replacements.get(name, name)}\n{value}", pad=1.0)
        ax.title.set_fontsize(title_size)
        for text in ax.texts:
            text.set_fontsize(title_size)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.99, wspace=0.04, hspace=0.04)


def save_publication_figure(
    fig,
    path: Union[str, Path],
    *,
    dpi: int = RASTER_DPI,
    bbox_inches: Optional[str] = "tight",
) -> Path:
    """Save with a white background and deterministic raster resolution."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches=bbox_inches, facecolor="white")
    return output


configure_publication_matplotlib()
