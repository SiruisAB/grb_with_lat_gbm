import pytest

from grb_project.web_app import SUPPORTED_MODELS, _validate_selected_models


def test_supported_models_default_starts_with_band():
    assert SUPPORTED_MODELS[0] == "band"


def test_validate_selected_models_preserves_selected_models():
    assert _validate_selected_models(["band", "comp+bb"]) == ["band", "comp+bb"]


def test_validate_selected_models_rejects_empty_selection():
    with pytest.raises(ValueError, match="至少选择一个模型"):
        _validate_selected_models([])
