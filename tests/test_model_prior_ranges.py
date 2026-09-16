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


class _Composite(tuple):
    """桩组合模型：支持 a + b + c 链式相加，摊平成一个扁平元组。"""

    def __add__(self, other):
        return _Composite(tuple(self) + (other,))


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
        return _Composite((self, other))


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
    astromodels.DoubleSmoothlyBrokenPowerlaw = _ModelShape
    astromodels.Uniform_prior = _Prior

    astropy = types.ModuleType("astropy")
    units = types.ModuleType("astropy.units")
    units.Quantity = type("Quantity", (), {})
    units.dimensionless_unscaled = object()
    astropy.units = units

    monkeypatch.setitem(sys.modules, "astromodels", astromodels)
    monkeypatch.setitem(sys.modules, "astropy", astropy)
    monkeypatch.setitem(sys.modules, "astropy.units", units)
    three_ml = types.ModuleType("threeML")
    # modelbuild 通过 from threeML import * 取得 Truncated_gaussian
    three_ml.Truncated_gaussian = _Prior
    monkeypatch.setitem(sys.modules, "threeML", three_ml)

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


def test_thermal_temperatures_do_not_follow_lat_expansion(monkeypatch):
    """热成分的 kT 只能由 GBM 波段约束，LAT(>100 MeV) 对它没有约束力。

    若 kT 跟着 gbm+lat 放宽到 1e8 keV，多出的 3.4 个数量级全是与连续谱
    退化的平坦似然区，dynesty 会在里面空转（实测单 bin 满核 3.5 小时未收敛）。
    """
    module = _load_modelbuild(monkeypatch)

    # 单模型黑体：直接取参数验证
    _assert_energy_parameter(
        module.build_model("blackbody", analysis_mode="gbm+lat").kT, 8.0, 4.0e4
    )
    # 无论哪种模式，热成分温度都封顶在 GBM 波段
    _assert_energy_parameter(
        module.build_model("blackbody", analysis_mode="gbm").kT, 8.0, 4.0e4
    )
    # 复合模型（band+bb / band+mbb 等）在本文件的桩里被摊平为元组，
    # 属性不可达，其温度站点由下面的源码字符串测试逐处覆盖。


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


def test_mbb_temperatures_use_thermal_bounds_not_energy_bounds():
    """多色黑体的 kT_min/kT_max 与单色黑体一样走 thermal_bounds。"""
    path = Path(__file__).parents[1] / "modelbuild.py"
    source = path.read_text(encoding="utf-8")

    assert "_mbb_temperature_bounds" not in source
    assert source.count("_set_energy_prior(m.kT_min, thermal_bounds)") == 1
    assert source.count("_set_energy_prior(m.kT_max, thermal_bounds)") == 1
    assert source.count("_set_energy_prior(mbb.kT_min, thermal_bounds)") == 2
    assert source.count("_set_energy_prior(mbb.kT_max, thermal_bounds)") == 2
    # 不得有温度参数漏回 energy_bounds
    for leaked in (
        "m.kT, energy_bounds",
        "m.kT_min, energy_bounds",
        "m.kT_max, energy_bounds",
        "mbb.kT_min, energy_bounds",
        "mbb.kT_max, energy_bounds",
        "bb.kT, energy_bounds",
        "m.kT_1, energy_bounds",
    ):
        assert leaked not in source, f"{leaked} 仍在使用 energy_bounds"


def test_gaussian_width_uses_the_selected_detector_lower_bound():
    path = Path(__file__).parents[1] / "modelbuild.py"
    source = path.read_text(encoding="utf-8")

    assert "_set_energy_prior(gauss.sigma, energy_bounds)" in source
    assert "_set_energy_prior(gauss.sigma, energy_bounds, lower_bound=1.0)" not in source


def _iter_shapes(built):
    """把 build_model 返回的（可能嵌套的）组合模型摊平成单个分量。"""
    if isinstance(built, tuple):
        for item in built:
            yield from _iter_shapes(item)
    else:
        yield built


def test_truncated_gaussian_priors_contain_their_own_centre(monkeypatch):
    """截断高斯先验的 mu 必须落在 [lower_bound, upper_bound] 区间内。

    band+bb+pl 曾把 PL 谱指数的先验区间写成 parameters[4]（黑体归一化
    1E-6）而不是 parameters[7]（-2.0），区间因此塌成 (-0.5, 0.5)，而
    mu=-2 落在区间之外。结果是 PL 谱指数被锁死在物理上不合理的范围，
    且先验密度在整个区间上都只是高斯的极小尾巴。
    """
    module = _load_modelbuild(monkeypatch)

    checked = 0
    # comp+pl 用 m = Cutoff_powerlaw() + Powerlaw() 后按 K_1/index_2 访问，
    # 本文件的桩模型无法模拟这种合并命名，故不列入（其索引已人工核对无误）。
    for mstr in ("band", "comp", "band+bb", "band+pl", "band+bb+pl"):
        built = module.build_model(mstr, analysis_mode="gbm")
        for shape in _iter_shapes(built):
            for name in getattr(shape, "parameter_names", ()):
                prior = getattr(getattr(shape, name), "prior", None)
                mu = getattr(prior, "mu", None)
                lower = getattr(prior, "lower_bound", None)
                upper = getattr(prior, "upper_bound", None)
                if mu is None or lower is None or upper is None:
                    continue
                assert lower <= mu <= upper, (
                    f"{mstr} 的 {type(shape).__name__}.{name} 先验区间 "
                    f"[{lower}, {upper}] 不包含中心值 mu={mu}"
                )
                checked += 1

    assert checked > 0, "没有检查到任何带 mu 的截断高斯先验"
