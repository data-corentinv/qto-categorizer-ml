"""Define a job for generating batch predictions from a registered model."""

# %% IMPORTS

import typing as T

import pandas as pd
import pydantic as pdt
import typing_extensions as TX

from qto_categorizer_ml.core import schemas
from qto_categorizer_ml.io import datasets, registries
from qto_categorizer_ml.jobs import base

# %% JOBS


class InferenceJob(base.Job):
    """Generate batch predictions from a registered model.

    Parameters:
        inputs (datasets.ReaderKind): reader for the inputs data.
        outputs (datasets.WriterKind ): writer for the outputs data.
        version_or_stage (str): version or mlflow stage of the model.
        loader (registries.LoaderKind): loader system for the model.
    """

    KIND: T.Literal["InferenceJob"] = "InferenceJob"

    # Inputs
    inputs: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    # Outputs
    outputs: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    # Model
    version_or_stage: str = "Production"
    # Loader
    loader: registries.LoaderKind = pdt.Field(registries.CustomLoader(), discriminator="KIND")

    @TX.override
    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)
        # - mlflow
        client = self.mlflow_service.client()
        logger.info("With client: {}", client)

        # inputs
        logger.info("Read inputs: {}", self.inputs)
        inputs_ = self.inputs.read()  # unchecked!
        inputs_ = inputs_.dropna()
        inputs = schemas.InputsSchema.check(inputs_)
        logger.debug("- Inputs shape: {}", inputs.shape)

        # model
        logger.info("With model: {}", self.mlflow_service.registry_name)
        model_uri = registries.uri_for_model(
            model_name=self.mlflow_service.registry_name,
            version_or_stage=self.version_or_stage,
        )
        logger.debug("- Model URI: {}", model_uri)

        # loader
        logger.info("Load model: {}", self.loader)
        model = self.loader.load(uri=model_uri)
        logger.debug("- Model: {}", model)

        # version
        # version = client.search_model_versions(
        #     filter_string=f'run_id="{model.model.metadata.run_id}"', max_results=1
        # )[0]

        # outputs
        logger.info("Predict outputs: {}", len(inputs))
        outputs = model.predict(inputs=inputs)  # checked
        logger.debug("- Outputs shape: {}", outputs.shape)

        # write
        logger.info("Write outputs: {}", self.outputs)
        self.outputs.write(pd.DataFrame({"predicted_class": outputs}))

        return locals()
