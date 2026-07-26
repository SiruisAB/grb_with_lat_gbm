from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np


def test_interactive_path_uses_same_stem_as_png(tmp_path: Path) -> None:
    from grb_project import interactive_lightcurves

    assert interactive_lightcurves.interactive_lightcurve_path(tmp_path / "burst.png") == tmp_path / "burst.html"


def test_interactive_html_contains_hoverable_gbm_and_lat_data(tmp_path: Path) -> None:
    from grb_project import interactive_lightcurves
    output = tmp_path / "burst.html"

    interactive_lightcurves.write_interactive_lightcurve_html(
        output,
        gbm_panels=[
            {
                "title": "NaI 8–50 keV",
                "time": np.array([0.05, 0.15]),
                "rate": np.array([10.0, 12.0]),
                "background": np.array([2.0, 2.5]),
            }
        ],
        lat_data={
            "time": np.array([0.1]),
            "energy": np.array([120.0]),
            "probability": np.array([0.95]),
            "bin_edges": np.array([0.0, 0.2]),
            "counts": np.array([1]),
        },
        x_range=(0.0, 0.2),
    )

    html = output.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "Time \\u2212 T0" in html
    assert "Rate" in html
    assert "GRB probability" in html
    assert "120" in html
    assert html.count("plotly.js v") == 1


def test_web_prefers_interactive_html_and_keeps_png_fallback(tmp_path: Path) -> None:
    from grb_project.web_app import _show_lightcurve_plot

    png_path = tmp_path / "burst.png"
    html_path = tmp_path / "burst.html"
    html_path.write_text("<html>interactive</html>", encoding="utf-8")
    fake_st = mock.MagicMock()

    returned = _show_lightcurve_plot(fake_st, png_path, caption="Burst")

    assert returned == html_path
    fake_st.components.v1.html.assert_called_once_with(
        "<html>interactive</html>", height=900, scrolling=True
    )
    fake_st.image.assert_not_called()

    html_path.unlink()
    returned = _show_lightcurve_plot(fake_st, png_path, caption="Burst")

    assert returned == png_path
    fake_st.image.assert_called_once_with(str(png_path), caption="Burst", use_container_width=True)
