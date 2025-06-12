"""Define and validate dataframe schemas."""

# %% IMPORTS

import datetime
import typing as T

import pandas as pd
import pandera as pa
import pandera.typing as papd
import pandera.typing.common as padt

# %% TYPES

TSchema = T.TypeVar("TSchema", bound="pa.DataFrameModel")

# %% SCHEMAS


class Schema(pa.DataFrameModel):
    """Base class for a dataframe schema.

    Use a schema to type your dataframe object.
    e.g., to communicate and validate its fields.
    """

    class Config:
        """Default configurations for all schemas.

        Parameters:
            coerce (bool): convert data type if possible.
            strict (bool): ensure the data type is correct.
        """

        coerce: bool = True
        strict: bool = True

    @classmethod
    def check(cls: T.Type[TSchema], data: pd.DataFrame) -> papd.DataFrame[TSchema]:
        """Check the dataframe with this schema.

        Args:
            data (pd.DataFrame): dataframe to check.

        Returns:
            papd.DataFrame[TSchema]: validated dataframe.
        """
        return T.cast(papd.DataFrame[TSchema], cls.validate(data))


class InputsSchema(Schema):
    """Schema for the project inputs."""

    TRANSACTION_ID: papd.Series[padt.String]
    DATE_EMITTED: papd.Series[padt.DateTime] = pa.Field(
        ge=datetime.datetime(2023, 8, 31), le=datetime.datetime(2024, 9, 30)
    )
    AMOUNT: papd.Series[padt.Float] = pa.Field(gt=0)
    TYPE_OF_PAYMENT: papd.Series[padt.String] = pa.Field(nullable=True)
    MERCHANT_NAME: papd.Series[padt.String] = pa.Field(nullable=True)
    DESCRIPTION: papd.Series[padt.String] = pa.Field(nullable=True)
    SIDE: papd.Series[padt.Int] = pa.Field(nullable=False)
    CATEGORY: papd.Series[padt.String]


class FeatInputsSchema(Schema):
    """Schema for the project inputs."""

    AMOUNT: papd.Series[padt.Float] = pa.Field(gt=0)
    TYPE_OF_PAYMENT: papd.Series[padt.String] = pa.Field(nullable=True)
    MERCHANT_NAME: papd.Series[padt.String] = pa.Field(nullable=True)
    DESCRIPTION: papd.Series[padt.String] = pa.Field(nullable=True)


Inputs = papd.DataFrame[InputsSchema]


class TargetsSchema(Schema):
    """Schema for the project targets."""

    CATEGORY: papd.Series[padt.String]


Targets = papd.DataFrame[TargetsSchema]


class OutputsSchema(Schema):
    """Schema for the project outputs."""

    prediction: papd.Series[padt.String]


Outputs = papd.DataFrame[OutputsSchema]
