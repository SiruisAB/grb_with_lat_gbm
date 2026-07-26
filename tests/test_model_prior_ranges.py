from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path


class _Prior:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Parameter:
    def __init__(self):
        self.prior = None
        self.min_value = None
        self.max_value = None
        self.value = None


class _ModelShape:
    parameter_names: tuple[str, ...] = ()

    def __init__(self, **_kwargs):
        for name in self.parameter_names:
            object.__setattr__(self, name, _Parameter())

    def __setattr__(self, name, value):
        current = self.__dict__.get(name)
        if isinstance(current, _Parameter) and not isinstance(value, _Parameter):
            current.value = value
            return
        object.__setattr__(self, name, value)

    def __add__(self, other):
        return (self, other)


class _Band(_ModelShape):
    parameter_names = ("K", "alpha", "xp", "beta")


class _CutoffPowerlaw(_ModelShape):
    parameter_names = ("K", "index", "xc")


class _Blackbody(_ModelShape):
    parameter_names = ("K", "kT")


class _NDP(_ModelShape):
    parameter_names = ("K", "k", "ec")


class _Powerlaw(_ModelShape):
    parameter_names = ("K", "index")


def _load_modelbuild(monkeypatch):
    astromodels = types.ModuleType("astromodels")
    astromodels.Band = _Band
    astromodels.Blackbody = _Blackbody
    astromodels.Cutoff_powerlaw = _CutoffPowerlaw
    astromodels.Function1D = object
    astromodels.FunctionMeta = type
    astromodels.Log_uniform_prior = _Prior
    astromodels.Model = object
    astromodels.NonDissipativePhotosphere = _NDP
    astromodels.PointSource = object
    astromodels.Powerlaw = _Powerlaw
    astromodels.Uniform_prior = _Prior

    astropy = types.ModuleType("astropy")
    units = types.ModuleType("astropy.units")
    units.Quantity = type("Quantity", (), {})
    units.dimensionless_unscaled = object()
    astropy.units = units

    monkeypatch.setitem(sys.modules, "astromodels", astromodels)
    monkeypatch.setitem(sys.modules, "astropy", astropy)
    monkeypatch.setitem(sys.modules, "astropy.units", units)
    monkeypatch.setitem(sys.modules, "threeML", types.ModuleType("threeML"))

    path = Path(__file__).parents[1] / "modelbuild.py"
    spec = importlib.util.spec_from_file_location("modelbuild_prior_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_energy_parameter(parameter, lower, upper):
    assert parameter.prior.lower_bound == lower
    assert parameter.prior.upper_bound == upper
    assert parameter.min_value == lower
    assert parameter.max_value == upper


def test_gbm_models_use_gbm_detector_energy_range(monkeypatch):
    module = _load_modelbuild(monkeypatch)

    _assert_energy_parameter(module.build_model("band", analysis_mode="gbm").xp, 8.0, 40000.0)
    _assert_energy_parameter(module.build_model("comp", analysis_mode="gbm").xc, 8.0, 40000.0)
    _assert_energy_parameter(module.build_model("blackbody", analysis_mode="gbm").kT, 8.0, 40000.0)


def test_joint_models_expand_energy_range_to_lat(monkeypatch):
    module = _load_modelbuild(monkeypatch)

    _assert_energy_parameter(module.build_model("band", analysis_mode="gbm+lat").xp, 8.0, 1e8)
    _assert_energy_parameter(module.build_model("comp", analysis_mode="gbm+lat").xc, 8.0, 1e8)
    _assert_energy_parameter(module.build_model("blackbody", analysis_mode="gbm+lat").kT, 8.0, 1e8)


def test_bayesian_fit_passes_analysis_mode_to_model_builder():
    path = Path(__file__).parents[1] / "bayesian_fit.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "build_model"
    ]
    assert len(calls) == 1
    assert any(
        keyword.arg == "analysis_mode"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "analysis_mode"
        for keyword in calls[0].keywords
    )


def test_energy_models_assign_real_astromodel_parameter_names():
    path = Path(__file__).parents[1] / "modelbuild.py"
    source = path.read_text(encoding="utf-8")

    assert "m.K, m.ec = parameters[0], parameters[1]" in source
    assert "m.K, m.alpha, m.break_energy, m.beta, m.break_scale =" in source
    assert "band.k," not in source


def test_mbb_temperatures_each_use_the_full_selected_energy_range():
    path = Path(__file__).parents[1] / "modelbuild.py"
    source = path.read_text(encoding="utf-8")

    assert "_mbb_temperature_bounds" not in source
    assert source.count("_set_energy_prior(m.kT_min, energy_bounds)") == 1
    assert source.count("_set_energy_prior(m.kT_max, energy_bounds)") == 1
    assert source.count("_set_energy_prior(mbb.kT_min, energy_bounds)") == 2
    assert source.count("_set_energy_prior(mbb.kT_max, energy_bounds)") == 2


def test_gaussian_width_uses_the_selected_detector_lower_bound():
    path = Path(__file__).parents[1] / "modelbuild.py"
    source = path.read_text(encoding="utf-8")

    assert "_set_energy_prior(gauss.sigma, energy_bounds)" in source
    assert "_set_energy_prior(gauss.sigma, energy_bounds, lower_bound=1.0)" not in source
