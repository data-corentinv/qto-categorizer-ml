# %% IMPORTS

from qto_categorizer_ml.io import services
from qto_categorizer_ml.jobs import base

# %% JOBS


def test_job(
    aws_service: services.AWSService,
    logger_service: services.LoggerService,
    mlflow_service: services.MlflowService,
) -> None:
    """Test job

    Args:
        aws_service (services.AWSService): _description_
        logger_service (services.LoggerService): _description_
        mlflow_service (services.MlflowService): _description_

    Returns:
        _type_: _description_
    """

    # given
    class MyJob(base.Job):
        """Job example

        Args:
            base (_type_): _description_

        Returns:
            _type_: _description_
        """

        KIND: str = "MyJob"

        def run(self) -> base.Locals:
            a, b = 1, "test"
            return locals()

    job = MyJob(
        aws_service=aws_service, logger_service=logger_service, mlflow_service=mlflow_service
    )
    # when
    with job as runner:
        out = runner.run()
    # then
    # - inputs
    assert hasattr(job, "aws_service"), "Job should have an AWS service!"
    assert hasattr(job, "logger_service"), "Job should have an Logger service!"
    assert hasattr(job, "mlflow_service"), "Job should have an Mlflow service!"
    # - outputs
    assert set(out) == {"self", "a", "b"}, "Run should return the local variables!"
