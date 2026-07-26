"""Standalone Plotly rendering for hoverable GRB light curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


def interactive_lightcurve_path(png_path: Union[str, Path]) -> Path:
    """Return the standalone interactive HTML path paired with a PNG."""

    return Path(png_path).with_suffix(".html")


def write_interactive_lightcurve_html(
    output_path: Union[str, Path],
    *,
    gbm_panels: Sequence[Dict[str, Any]],
    lat_data: Optional[Dict[str, np.ndarray]],
    x_range: Tuple[float, float],
) -> Path:
    """Write an offline Plotly light curve with hoverable data points."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panel_count = len(gbm_panels) + (1 if lat_data is not None else 0)
    if panel_count == 0:
        raise ValueError("交互光变图没有可绘制的数据")
    specs = [[{"secondary_y": False}] for _ in gbm_panels]
    if lat_data is not None:
        specs.append([{"secondary_y": True}])
    figure = make_subplots(rows=panel_count, cols=1, shared_xaxes=True, vertical_spacing=0.035, specs=specs)

    for row, panel in enumerate(gbm_panels, start=1):
        figure.add_trace(
            go.Scatter(
                x=np.asarray(panel["time"], dtype=float),
                y=np.asarray(panel["rate"], dtype=float),
                mode="lines+markers",
                line={"shape": "hv"},
                marker={"size": 4},
                name=str(panel["title"]),
                hovertemplate=(
                    "Series: %{fullData.name}<br>Time − T0: %{x:.4f} s"
                    "<br>Rate: %{y:.4f} cnts/s<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
        background = panel.get("background")
        if background is not None:
            figure.add_trace(
                go.Scatter(
                    x=np.asarray(panel["time"], dtype=float),
                    y=np.asarray(background, dtype=float),
                    mode="lines",
                    line={"shape": "hv", "dash": "dot"},
                    name=f"{panel['title']} background",
                    hovertemplate=(
                        "Series: %{fullData.name}<br>Time − T0: %{x:.4f} s"
                        "<br>Background: %{y:.4f} cnts/s<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
        figure.update_yaxes(title_text="Rate (cnts/s)", row=row, col=1)

    if lat_data is not None:
        row = panel_count
        edges = np.asarray(lat_data["bin_edges"], dtype=float)
        counts = np.asarray(lat_data["counts"], dtype=float)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = edges[1:] - edges[:-1]
        figure.add_trace(
            go.Bar(
                x=centers,
                y=counts,
                width=widths,
                name="LAT event count",
                opacity=0.35,
                hovertemplate="Time − T0: %{x:.4f} s<br>Events: %{y:.0f}<extra></extra>",
            ),
            row=row,
            col=1,
            secondary_y=False,
        )
        probability = np.asarray(lat_data["probability"], dtype=float)
        figure.add_trace(
            go.Scatter(
                x=np.asarray(lat_data["time"], dtype=float),
                y=np.asarray(lat_data["energy"], dtype=float),
                customdata=probability[:, None],
                mode="markers",
                marker={"color": "black", "size": 7},
                name="LAT photon",
                hovertemplate=(
                    "Time − T0: %{x:.4f} s<br>Energy: %{y:.4f} MeV"
                    "<br>GRB probability: %{customdata[0]:.4f}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
            secondary_y=True,
        )
        figure.update_yaxes(title_text="LAT events", row=row, col=1, secondary_y=False)
        figure.update_yaxes(title_text="Energy [MeV]", type="log", row=row, col=1, secondary_y=True)

    figure.update_xaxes(range=[float(x_range[0]), float(x_range[1])], title_text="Time − T0 [s]", row=panel_count, col=1)
    figure.update_layout(
        height=max(420, 280 * panel_count),
        hovermode="closest",
        template="plotly_white",
        margin={"l": 75, "r": 75, "t": 35, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
    )
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs=True, full_html=True)
    return output
