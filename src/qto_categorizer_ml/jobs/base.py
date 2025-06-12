"""Base for high-level project tasks."""

# %% IMPORTS

import abc
import types as TS
import typing as T

import pydantic as pdt
import typing_extensions as TX

from qto_categorizer_ml.io import services

# %% TYPES

# Local job variables
Locals = T.Dict[str, T.Any]

# %% JOBS


class Job(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a job.

    use a job to execute runs in  context.
    e.g., to define common services like logger

    Parameters:
        aws_service (services.AWSService): manage the AWS system.
        logger_service (services.LoggerService): manage the logging system.
        mlflow_service (services.MlflowService): manage the mlflow system.
    """

    KIND: str

    aws_service: services.AWSService = services.AWSService()
    logger_service: services.LoggerService = services.LoggerService()
    mlflow_service: services.MlflowService = services.MlflowService()

    def __enter__(self) -> TX.Self:
        """Enter the job context.

        Returns:
            TX.Self: return the current object.
        """
        self.logger_service.start()
        logger = self.logger_service.logger()
        logger.debug("[START] Logger service: {}", self.logger_service)
        logger.debug("[START] AWS service: {}", self.aws_service)
        self.aws_service.start()
        logger.debug("[START] Mlflow service: {}", self.mlflow_service)
        self.mlflow_service.start()
        return self

    def __exit__(
        self,
        exc_type: T.Type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TS.TracebackType | None,
    ) -> T.Literal[False]:
        """Exit the job context.

        Args:
            exc_type (T.Type[BaseException] | None): ignored.
            exc_value (BaseException | None): ignored.
            exc_traceback (TS.TracebackType | None): ignored.

        Returns:
            T.Literal[False]: always propagate exceptions.
        """
        logger = self.logger_service.logger()
        logger.debug("[STOP] Mlflow service: {}", self.mlflow_service)
        self.mlflow_service.stop()
        logger.debug("[STOP] AWS service: {}", self.aws_service)
        self.aws_service.stop()
        logger.debug("[STOP] Logger service: {}", self.logger_service)
        self.logger_service.stop()
        return False  # re-raise

    @abc.abstractmethod
    def run(self) -> Locals:
        """Run the job in context.

        Returns:
            Locals: local job variables.
        """
