"""Add the doubly smoothly broken power law (2SBPL) to the model registry.

Two variants are registered:

  2SBPL      free low-energy index alpha1, cooling index alpha2, both breaks and
             the high-energy index; six free parameters. Tests whether the data
             prefer a low-energy break at all.
  2SBPL_syn  alpha1 and alpha2 fixed at the fast-cooling synchrotron prediction
             (-2/3 and -3/2). Four free parameters, the same count as Band, so
             the comparison with Band is not confounded by model flexibility.

Reference: Ravasio et al. 2018, A&A 613, A16; Ravasio et al. 2019, A&A 625, A60.
Run this on the server against grb_project/modelbuild.py and web_app.py.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

MODEL_CODE = '''
    elif mstr in ("2SBPL", "2SBPL_syn"):
        # Doubly smoothly broken power law (Ravasio et al. 2018, A&A 613, A16).
        # alpha1 applies below the cooling break xb, alpha2 between xb and the
        # nuFnu peak xp, and beta above xp. The two curvature parameters n1 and
        # n2 stay fixed at their defaults, as in the reference implementation.
        m = DoubleSmoothlyBrokenPowerlaw(piv=1E2)
        synchrotron = (mstr == "2SBPL_syn")

        m.K.prior = Log_uniform_prior(lower_bound=1E-7, upper_bound=1E1)
        m.K.value = 1E-2
        m.K.min_value, m.K.max_value = 1e-12, 1e3

        # xb is the cooling break and is expected well below the peak; keeping
        # its prior below the xp prior avoids the label-swapping degeneracy that
        # arises if both breaks may occupy the same range.
        _set_energy_prior(m.xp, energy_bounds)
        m.xp.value = 300.0
        xb_upper = min(1.0e4, float(energy_bounds[1]))
        m.xb.min_value, m.xb.max_value = float(energy_bounds[0]), xb_upper
        m.xb.prior = Log_uniform_prior(
            lower_bound=float(energy_bounds[0]), upper_bound=xb_upper
        )
        m.xb.value = 100.0

        m.beta.prior = Uniform_prior(lower_bound=-5.0, upper_bound=-1.6)
        m.beta.min_value, m.beta.max_value = -5.0, -1.6
        m.beta.value = -2.5

        if synchrotron:
            # fast-cooling synchrotron: alpha1 = -2/3 below the cooling break,
            # alpha2 = -3/2 above it. Both held fixed, leaving four free
            # parameters (K, xb, xp, beta), the same number as Band.
            m.alpha1.value = -2.0 / 3.0
            m.alpha2.value = -1.5
            m.alpha1.fix = True
            m.alpha2.fix = True
        else:
            m.alpha1.prior = Uniform_prior(lower_bound=-1.5, upper_bound=1.0)
            m.alpha1.min_value, m.alpha1.max_value = -1.5, 1.0
            m.alpha1.value = -2.0 / 3.0
            m.alpha2.prior = Uniform_prior(lower_bound=-3.0, upper_bound=-0.5)
            m.alpha2.min_value, m.alpha2.max_value = -3.0, -0.5
            m.alpha2.value = -1.5

'''


def patch_modelbuild(path):
    src = path.read_text(encoding="utf-8")
    if "2SBPL_syn" in src:
        print("modelbuild.py: already patched")
        return False

    # import the astromodels function alongside the others
    if "DoubleSmoothlyBrokenPowerlaw" not in src:
        src = src.replace(
            "from astromodels import (\n    Band,",
            "from astromodels import (\n    Band,\n    DoubleSmoothlyBrokenPowerlaw,",
            1,
        )

    # insert the new branch immediately before the unknown-model fallback
    anchor = '    else:\n        raise ValueError(f"未知模型类型: {mstr}")'
    if anchor not in src:
        raise SystemExit("modelbuild.py: could not find the fallback branch anchor")
    src = src.replace(anchor, MODEL_CODE + anchor, 1)

    path.write_text(src, encoding="utf-8")
    print("modelbuild.py: patched")
    return True


def patch_supported(path):
    src = path.read_text(encoding="utf-8")
    if '"2SBPL_syn"' in src:
        print("web_app.py: already patched")
        return False
    src = src.replace('    "SBPL",\n', '    "SBPL",\n    "2SBPL",\n    "2SBPL_syn",\n', 1)
    path.write_text(src, encoding="utf-8")
    print("web_app.py: patched")
    return True


if __name__ == "__main__":
    patch_modelbuild(ROOT / "modelbuild.py")
    patch_supported(ROOT / "web_app.py")
