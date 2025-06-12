# %% IMPORTS

import pytest
from qto_categorizer_ml.core import schemas
from qto_categorizer_ml.utils import signers

# %% SIGNERS

def test_manual_signer(inputs: schemas.Inputs, outputs: schemas.Outputs) -> None:
    def specs_to_dict(specs: list[dict[str, str]]) -> dict[str, str]:
        """Convert a list of mlflow signature specs to a dict."""
        return {spec["name"]: spec["type"] for spec in specs}

    # given
    inputs_specs = [
        {"type": "boolean", "name": "c1"},
        {"type": "integer", "name": "c2"},
        {"type": "long", "name": "c3"},
        {"type": "float", "name": "c4"},
        {"type": "double", "name": "c5"},
        {"type": "string", "name": "c6"},
        {"type": "binary", "name": "c7"},
        {"type": "datetime", "name": "c8"},
    ]
    outputs_specs = [
        {"type": "long", "name": "out"},
    ]
    signer = signers.ManualSigner(inputs_specs=inputs_specs, outputs_specs=outputs_specs)
    # when
    signature = signer.sign(inputs=inputs, outputs=outputs)
    # then
    assert specs_to_dict(signature.inputs.to_dict()) == specs_to_dict(
        inputs_specs
    ), "Signature inputs should match inputs specs."
    assert specs_to_dict(signature.outputs.to_dict()) == specs_to_dict(
        outputs_specs
    ), "Signature outputs should match outputs specs."

def test_pandera_signer__invalid_schema() -> None:
    # given
    bad_schema_name = "BadSchema"
    # when
    with pytest.raises(ValueError) as error:
        signers.PanderaSigner(
            inputs_schema_name=bad_schema_name, outputs_schema_name=bad_schema_name
        )
    # then
    assert error.match(f"Schema name was not found in module. Got: {bad_schema_name}.")
