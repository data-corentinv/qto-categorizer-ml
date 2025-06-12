"""Generate signatures for AI/ML models."""

# %% IMPORTS

import abc
import json
import typing as T

import mlflow
import pydantic as pdt
import typing_extensions as TX
from mlflow.models import signature as ms
from mlflow.types import schema as mlflow_schemas

from qto_categorizer_ml.core import schemas

# %% TYPES

Signature: T.TypeAlias = ms.ModelSignature

# %% SIGNERS


class Signer(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for generating model signatures.

    Note: allow to switch between signing strategies.
    e.g., automatic inference, manual signatures, pandera schemas.

    https://mlflow.org/docs/latest/models.html#model-signature-and-input-example
    """

    KIND: str

    @abc.abstractmethod
    def sign(self, inputs: schemas.Inputs, outputs: schemas.Outputs) -> Signature:
        """Generate a model signature from its inputs/outputs.

        Args:
            inputs (schemas.Inputs): inputs data.
            outputs (schemas.Outputs): outputs data.

        Returns:
            Signature: signature of the model.
        """


class InferSigner(Signer):
    """Generate model signatures from inputs/outputs data."""

    KIND: T.Literal["InferSigner"] = "InferSigner"

    @TX.override
    def sign(self, inputs: schemas.Inputs, outputs: schemas.Outputs) -> Signature:
        return mlflow.models.infer_signature(model_input=inputs, model_output=outputs)


class ManualSigner(Signer):
    """Generate model signatures from manual specifications.

    https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.Signature.from_dict

    Parameters:
        inputs_specs (list[dict[str, T.Any]]): inputs specifications.
        outputs_specs (list[dict[str, T.Any]]): outputs specifications.
    """

    KIND: T.Literal["ManualSigner"] = "ManualSigner"

    inputs_specs: list[dict[str, T.Any]]
    outputs_specs: list[dict[str, T.Any]]

    @TX.override
    def sign(self, inputs: schemas.Inputs, outputs: schemas.Outputs) -> Signature:
        # note: mlflow only supports JSON strings
        inputs_json = json.dumps(self.inputs_specs)
        outputs_json = json.dumps(self.outputs_specs)
        signature_dict = {"inputs": inputs_json, "outputs": outputs_json}
        signature = Signature.from_dict(signature_dict=signature_dict)
        return signature


class PanderaSigner(Signer):
    """Generate model signatures from Pandera inputs/outputs schemas.

    Mapping from Pandera to Mlflow dtypes (only for supported types)
    - pandera: https://github.com/unionai-oss/pandera/blob/main/pandera/dtypes.py
    - mlflow: https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.DataType

    Parameters:
        inputs_schema_name (str): name of the inputs schema.
        outputs_schema_name (str): name of the outputs schema.
    """

    KIND: T.Literal["PanderaSigner"] = "PanderaSigner"

    inputs_schema_name: str = schemas.InputsSchema.__name__
    outputs_schema_name: str = schemas.OutputsSchema.__name__

    PANDERA_MLFLOW_MAPPING: dict[str, str] = {
        "bool": "boolean",
        "int": "long",
        "int8": "integer",
        "int16": "integer",
        "int32": "integer",
        "int64": "long",
        "uint": "long",
        "uint8": "integer",
        "uint16": "integer",
        "uint32": "integer",
        "uint64": "long",
        "float": "float",
        "float16": "float",
        "float32": "float",
        "float64": "double",
        "category": "string",
        "string": "string",
        "date": "datetime",
        "timestamp": "datetime",
        "datetime": "datetime",
        "binary": "binary",
    }

    @pdt.field_validator("inputs_schema_name", "outputs_schema_name")
    @classmethod
    def check_schema_name(cls, value: str) -> str:
        """Validate the schema name exists and is a valid schema subclasses.

        Args:
            value (str): value to validate.

        Raises:
            ValueError: schema does not exists or is not a valid schema subclass.

        Returns:
            str: validated value.
        """
        schema = getattr(schemas, value, None)
        if schema is None or not issubclass(schema, schemas.Schema):
            raise ValueError(f"Schema name was not found in module. Got: {value}.")
        return value

    @TX.override
    def sign(self, inputs: schemas.Inputs, outputs: schemas.Outputs) -> Signature:
        pandera_inputs_schema = getattr(schemas, self.inputs_schema_name)
        pandera_outputs_schema = getattr(schemas, self.outputs_schema_name)
        mlflow_inputs_schema = self._convert_pandera_to_mlflow(pandera_inputs_schema)
        mlflow_outputs_schema = self._convert_pandera_to_mlflow(pandera_outputs_schema)
        signature = Signature(inputs=mlflow_inputs_schema, outputs=mlflow_outputs_schema)
        return signature

    def _convert_pandera_to_mlflow(self, pandera_schema: schemas.Schema) -> mlflow_schemas.Schema:
        """Convert Pandera schema to Mlflow schema using dtype mapping.

        Args:
            pandera_schema (schemas.Schema): schema to convert.

        Raises:
            ValueError: schema contains unsupported dtypes.

        Returns:
            mlflow_schemas.Schema: converted schema.
        """
        mlflow_col_specs = []
        pandera_col_specs = pandera_schema.to_schema().columns
        for name, pandera_col_spec in pandera_col_specs.items():
            optional = not pandera_col_spec.required
            pandera_dtype_name = str(pandera_col_spec.dtype)
            mlflow_dtype_name = self.PANDERA_MLFLOW_MAPPING[pandera_dtype_name]
            mlflow_col_spec = mlflow_schemas.ColSpec(
                name=name, optional=optional, type=mlflow_dtype_name
            )
            mlflow_col_specs.append(mlflow_col_spec)
        mlflow_schema = mlflow_schemas.Schema(mlflow_col_specs)
        return mlflow_schema


SignerKind = InferSigner | ManualSigner | PanderaSigner
