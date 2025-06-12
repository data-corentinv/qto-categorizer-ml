"""Define a job for transitioning a registered model from a version to a stage."""

# %% IMPORTS

import typing as T

import typing_extensions as TX

from qto_categorizer_ml.jobs import base

# %% JOBS


class TransitionJob(base.Job):
    """Transition a registered model from a version to a stage.

    https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlfloaw.client.MlflowClient.get_latest_versions
    https://mlflow.org/docs/latest/python_api/mlflow.client.html?#mlflow.client.MlflowClient.transition_model_version_stage

    Parameters:
        stage (str): the mlflow stage to transition the model version to.
        version (int | None): the version of the model (use None for latest).
        archive_existing_versions (bool): whether to archive existing versions or not.
        latest_version_stages (list[str] | None): the stages to consider for the latest version.
    """

    KIND: T.Literal["TransitionJob"] = "TransitionJob"

    # Stage
    stage: str = "Production"
    # Version
    version: int | None = None
    archive_existing_versions: bool = True
    latest_version_stages: list[str] | None = None

    @TX.override
    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)
        # - mlflow
        client = self.mlflow_service.client()
        logger.info("With client: {}", client)
        name = self.mlflow_service.registry_name
        # version
        if self.version is None:
            from_version = max(
                client.get_latest_versions(name=name, stages=self.latest_version_stages),
                key=lambda v: int(v.version),
            )
        else:
            from_version = client.get_model_version(name=name, version=self.version)
        logger.info("From version: {}", from_version)
        # transition
        logger.info("Transition model: {}", name)
        to_version = client.transition_model_version_stage(
            name=name,
            stage=self.stage,
            version=from_version.version,
            archive_existing_versions=self.archive_existing_versions,
        )
        logger.debug("- To version: {}", to_version)
        return locals()
