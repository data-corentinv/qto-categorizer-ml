# %% IMPORTS

from qto_categorizer_ml.io import registries

# %% HELPERS


def test_uri_for_model() -> None:
    # given
    name = "testing"
    version = "1"
    # when
    uri = registries.uri_for_model(model_name=name, version_or_stage=version)
    # then
    assert uri == f"models:/{name}/{version}", "The model URI should be valid!"
