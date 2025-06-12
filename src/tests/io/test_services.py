# %% IMPORTS


import _pytest.logging as pl
import boto3
import mlflow
from qto_categorizer_ml.io import services

# %% SERVICES


def test_aws_service(aws_service: services.AWSService) -> None:
    # given
    service = aws_service
    # when
    session = service.session()
    # then
    assert boto3.DEFAULT_SESSION is not None, "Boto3 default session should be set!"
    assert session.region_name == service.region_name, "Session region name should be the same!"
    assert (
        boto3.DEFAULT_SESSION.region_name == service.region_name
    ), "Boto3 default session region name should be the same!"


def test_logger_service(
    logger_service: services.LoggerService, logger_caplog: pl.LogCaptureFixture
) -> None:
    # given
    service = logger_service
    logger = service.logger()
    # when
    logger.debug("DEBUG")
    logger.error("ERROR")
    # then
    assert "DEBUG" in logger_caplog.messages, "Debug message should be logged!"
    assert "ERROR" in logger_caplog.messages, "Error message should be logged!"


def test_mlflow_service(mlflow_service: services.MlflowService) -> None:
    # given
    service = mlflow_service
    config = mlflow_service.RunConfig(
        name="testing",
        tags={"service": "mlflow"},
        description="a test run.",
        log_system_metrics=True,
    )
    # when
    client = service.client()
    with service.run_context(config=config) as run:
        pass
    finished = client.get_run(run_id=run.info.run_id)
    # then
    # - run
    assert config.tags is not None, "Run config tags should be set!"
    # - mlflow
    assert service.tracking_uri == mlflow.get_tracking_uri(), "Tracking URI should be the same!"
    assert service.registry_uri == mlflow.get_registry_uri(), "Registry URI should be the same!"
    assert mlflow.get_experiment_by_name(service.experiment_name), "Experiment should be setup!"
    # - client
    assert service.tracking_uri == client.tracking_uri, "Tracking URI should be the same!"
    assert service.registry_uri == client._registry_uri, "Tracking URI should be the same!"
    assert client.get_experiment_by_name(service.experiment_name), "Experiment should be setup!"
    # - run
    assert run.info.run_name == config.name, "Run name should be the same!"
    assert config.description in run.data.tags.values(), "Run desc. should be in tags values!"
    assert (
        run.data.tags.items() > config.tags.items()
    ), "Run tags should be a subset of the given tags!"
    assert run.info.status == "RUNNING", "Run should be running!"
    # - finished
    assert finished.info.status == "FINISHED", "Finished should be finished!"
