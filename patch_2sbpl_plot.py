"""Teach separate_spectr.py that 2SBPL is a single-component model.

Without this the model name falls through to the composite branch, which
dereferences model2 and raises NameError before any spectrum is written.
Adds the plotting dispatch, the single-component membership test, and the peak
energy definition (the nuFnu peak of the 2SBPL is the fitted xp).
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
PATH = ROOT / "separate_spectr.py"

DISPATCH = '''
    if model_str in ('2SBPL', '2SBPL_syn'):
        # Doubly smoothly broken power law. The free-parameter order differs
        # between the two variants because 2SBPL_syn holds alpha1 and alpha2
        # fixed at the fast-cooling synchrotron values, so they are absent from
        # the fitted-parameter vector.
        model1 = DoubleSmoothlyBrokenPowerlaw(piv=1E2)
        model1.beta.min_value, model1.beta.max_value = -5.0, -1.6
        _lo, _hi = _energy_bounds_for_mode(analysis_mode)
        model1.xp.min_value, model1.xp.max_value = _lo, _hi
        model1.xb.min_value, model1.xb.max_value = _lo, min(1.0e4, _hi)
        if model_str == '2SBPL_syn':
            model1.alpha1.value = -2.0 / 3.0
            model1.alpha2.value = -1.5
            model1.K, model1.xb, model1.xp, model1.beta = parameter_values[:4]
        else:
            model1.alpha1.min_value, model1.alpha1.max_value = -1.5, 1.0
            model1.alpha2.min_value, model1.alpha2.max_value = -3.0, -0.5
            (model1.K, model1.alpha1, model1.xb,
             model1.alpha2, model1.xp, model1.beta) = parameter_values[:6]
        modelTotal = model1
        model_str1 = model_str

'''

SINGLE_LIST_OLD = "if model_str in ['pl','band','blackbody','comp','SBPL','NDP','mbb']:"
SINGLE_LIST_NEW = ("if model_str in ['pl','band','blackbody','comp','SBPL','NDP','mbb',"
                   "'2SBPL','2SBPL_syn']:")

EPEAK_OLD = """        elif model_str in ['SBPL']:"""
EPEAK_NEW = """        elif model_str in ['2SBPL', '2SBPL_syn']:
            # the nuFnu peak of the 2SBPL is the fitted xp; its index in the
            # free-parameter vector depends on whether alpha1/alpha2 are fixed
            Epeak = parameter_values[2] if model_str == '2SBPL_syn' else parameter_values[4]
            fluxEpeak = max(k0*xs*xs*modelTotal(xs))
            flux100mev = fluxPL[np.where( xs == 100000.0)]
        elif model_str in ['SBPL']:"""


def main():
    src = PATH.read_text(encoding="utf-8")
    if "2SBPL_syn" in src:
        print("separate_spectr.py: already patched")
        return

    if "DoubleSmoothlyBrokenPowerlaw" not in src.split("def ")[0]:
        # add the import next to the other astromodels spectral shapes
        marker = "        NonDissipativePhotosphere,\n"
        if marker not in src:
            raise SystemExit("could not find the astromodels import block")
        src = src.replace(
            marker, marker + "        DoubleSmoothlyBrokenPowerlaw,\n", 1
        )
        # keep the ImportError fallback assignment consistent
        src = src.replace(
            "NonDissipativePhotosphere = None",
            "NonDissipativePhotosphere = DoubleSmoothlyBrokenPowerlaw = None",
            1,
        )

    anchor = "    if model_str == 'SBPL':"
    if anchor not in src:
        raise SystemExit("could not find the SBPL dispatch anchor")
    src = src.replace(anchor, DISPATCH + anchor, 1)

    if SINGLE_LIST_OLD not in src:
        raise SystemExit("could not find the single-component membership test")
    src = src.replace(SINGLE_LIST_OLD, SINGLE_LIST_NEW, 1)

    if EPEAK_OLD not in src:
        raise SystemExit("could not find the SBPL Epeak branch")
    src = src.replace(EPEAK_OLD, EPEAK_NEW, 1)

    PATH.write_text(src, encoding="utf-8")
    print("separate_spectr.py: patched")


if __name__ == "__main__":
    main()
