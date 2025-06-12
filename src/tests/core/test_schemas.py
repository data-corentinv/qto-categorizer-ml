# %% IMPORTS

from qto_categorizer_ml.core import schemas
from qto_categorizer_ml.io import datasets

# %% SCHEMAS


def test_inputs_schema(inputs_reader: datasets.Reader) -> None:
    # given
    schema = schemas.InputsSchema
    # when
    data = inputs_reader.read()
    # then
    assert schema.check(data) is not None, "Inputs data should be valid!"


def test_targets_schema(targets_reader: datasets.Reader) -> None:
    # given
    schema = schemas.TargetsSchema
    # when
    data = targets_reader.read()
    # then
    assert schema.check(data) is not None, "Targets data should be valid!"
