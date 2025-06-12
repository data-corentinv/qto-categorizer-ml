"""Configuration for the tests."""

# %% IMPORTS

import os
import typing as T

import mlflow
import moto
import mypy_boto3_s3.service_resource as s3
import omegaconf
import pytest
from _pytest import logging as pl
from qto_categorizer_ml.core import metrics, models, schemas
from qto_categorizer_ml.io import datasets, registries, services
from qto_categorizer_ml.utils import searchers, signers, splitters

# %% FIXTURES

# %% - Paths


@pytest.fixture(scope="session")
def tests_path() -> str:
    """Return the path of the tests folder."""
    file = os.path.abspath(__file__)
    parent = os.path.dirname(file)
    return parent


@pytest.fixture(scope="session")
def data_path(tests_path: str) -> str:
    """Return the path of the data folder."""
    return os.path.join(tests_path, "data")


@pytest.fixture(scope="session")
def conf_path(tests_path: str) -> str:
    """Return the path of the conf folder."""
    return os.path.join(tests_path, "conf")


@pytest.fixture(scope="session")
def inputs_path(data_path: str) -> str:
    """Return the path of the inputs dataset."""
    return os.path.join(data_path, "data-products.csv")


@pytest.fixture(scope="session")
def targets_path(data_path: str) -> str:
    """Return the path of the targets dataset."""
    return os.path.join(data_path, "outputs.csv")


@pytest.fixture(scope="session")
def outputs_path(data_path: str) -> str:
    """Return the path of the outputs dataset."""
    return os.path.join(data_path, "predictions.csv")


@pytest.fixture(scope="function")
def tmp_outputs_path(tmp_path: str) -> str:
    """Return a tmp path for the outputs dataset."""
    return os.path.join(tmp_path, "predictions.csv")


# %% - Configs


@pytest.fixture(scope="session")
def extra_config() -> str:
    """Extra config for scripts."""
    # use OmegaConf resolver: ${tmp_path:}
    config = """
    {
        "job": {
            "mlflow_service": {
                "tracking_uri": "${tmp_path:}/tracking/",
                "registry_uri": "${tmp_path:}/registry/",
            }
        }
    }
    """
    return config


# %% - Datasets


@pytest.fixture(scope="session")
def inputs_reader(inputs_path: str) -> datasets.CSVReader:
    """Return a reader for the inputs dataset."""
    return datasets.CSVReader(path=inputs_path)


@pytest.fixture(scope="session")
def targets_reader(targets_path: str) -> datasets.CSVReader:
    """Return a reader for the targets dataset."""
    return datasets.CSVReader(path=targets_path)


@pytest.fixture(scope="session")
def outputs_reader(outputs_path: str) -> datasets.CSVReader:
    """Return a reader for the outputs dataset."""
    return datasets.CSVReader(path=outputs_path)


@pytest.fixture(scope="function")
def tmp_outputs_writer(tmp_outputs_path: str) -> datasets.CSVWriter:
    """Return a writer for the tmp outputs dataset."""
    return datasets.CSVWriter(path=tmp_outputs_path)


# %% - Dataframes


@pytest.fixture(scope="session")
def inputs(inputs_reader: datasets.CSVReader) -> schemas.Inputs:
    """Return the inputs data."""
    data = inputs_reader.read()
    return schemas.InputsSchema.check(data)


@pytest.fixture(scope="session")
def targets(targets_reader: datasets.CSVReader) -> schemas.Targets:
    """Return the targets data."""
    data = targets_reader.read()
    return schemas.TargetsSchema.check(data)


@pytest.fixture(scope="session")
def outputs(outputs_reader: datasets.CSVReader) -> schemas.Outputs:
    """Return the outputs data."""
    data = outputs_reader.read()
    return schemas.OutputsSchema.check(data)


# %% - Splitters


@pytest.fixture(scope="session")
def splitter() -> splitters.TrainTestSplitter:
    """Return the default splitter object."""
    return splitters.TrainTestSplitter()


# %% - Searchers


@pytest.fixture(scope="session")
def searcher() -> searchers.Searcher:
    """Return the default searcher object."""
    param_grid = {"max_depth": [1, 2], "n_estimators": [3]}
    return searchers.GridCVSearcher(param_grid=param_grid)


# %% - Subsets


@pytest.fixture(scope="session")
def train_test_sets(
    splitter: splitters.Splitter, inputs: schemas.Inputs, targets: schemas.Targets
) -> tuple[schemas.Inputs, schemas.Targets, schemas.Inputs, schemas.Targets]:
    """Return the inputs and targets train and test sets from the splitter."""
    train_index, test_index = next(splitter.split(inputs=inputs, targets=targets))
    inputs_train, inputs_test = inputs.iloc[train_index], inputs.iloc[test_index]
    targets_train, targets_test = targets.iloc[train_index], targets.iloc[test_index]
    return (
        T.cast(schemas.Inputs, inputs_train),
        T.cast(schemas.Targets, targets_train),
        T.cast(schemas.Inputs, inputs_test),
        T.cast(schemas.Targets, targets_test),
    )


# %% - Models


@pytest.fixture(scope="session")
def model(
    train_test_sets: tuple[schemas.Inputs, schemas.Targets, schemas.Inputs, schemas.Targets],
) -> models.BaselineModel:
    """Return a train model for testing."""
    model = models.BaselineModel()
    inputs_train, targets_train, _, _ = train_test_sets
    model.fit(inputs=inputs_train, targets=targets_train)
    return model


# %% - Metrics


@pytest.fixture(scope="session")
def metric() -> metrics.SklearnMetric:
    """Return the default metric."""
    return metrics.SklearnMetric()


# %% - Signers


@pytest.fixture(scope="session")
def signer() -> signers.InferSigner:
    """Return a model signer."""
    return signers.InferSigner()


# %% - Environs


@pytest.fixture(scope="session")
def aws_environ() -> T.Dict[str, str]:
    """Return the AWS environ for testing."""
    os.environ["AWS_ACCESS_KEY_ID"] = "AWS_ACCESS_KEY_ID"
    os.environ["AWS_DEFAULT_REGION"] = "antarctica-99"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "AWS_SECRET_ACCESS_KEY"
    os.environ["AWS_SECURITY_TOKEN"] = "AWS_SECURITY_TOKEN"
    os.environ["AWS_SESSION_TOKEN"] = "AWS_SESSION_TOKEN"
    os.environ["MOTO_ALLOW_NONEXISTENT_REGION"] = "True"
    return dict(os.environ)


# %% - Services


@pytest.fixture(scope="session", autouse=True)
def aws_service(aws_environ: T.Dict[str, str]) -> T.Generator[services.AWSService, None, None]:
    """Return and start the AWS service ."""
    region_name = aws_environ["AWS_DEFAULT_REGION"]
    service = services.AWSService(region_name=region_name)
    service.start()
    yield service
    service.stop()


@pytest.fixture(scope="session", autouse=True)
def logger_service() -> T.Generator[services.LoggerService, None, None]:
    """Return and start the logger service."""
    service = services.LoggerService(colorize=False, diagnose=True)
    service.start()
    yield service
    service.stop()


@pytest.fixture
def logger_caplog(
    caplog: pl.LogCaptureFixture, logger_service: services.LoggerService
) -> T.Generator[pl.LogCaptureFixture, None, None]:
    """Extend pytest caplog fixture with the logger service (loguru)."""
    # https://loguru.readthedocs.io/en/stable/resources/migration.html#replacing-caplog-fixture-from-pytest-library
    logger = logger_service.logger()
    handler_id = logger.add(
        caplog.handler,
        level=0,
        format="{message}",
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(scope="function", autouse=True)
def mlflow_service(tmp_path: str) -> T.Generator[services.MlflowService, None, None]:
    """Return and start the mlflow service."""
    service = services.MlflowService(
        tracking_uri=f"{tmp_path}/tracking/",
        registry_uri=f"{tmp_path}/registry/",
        experiment_name="Experiment-Testing",
        registry_name="Registry-Testing",
    )
    service.start()
    yield service
    service.stop()


# %% - Resources


@pytest.fixture(scope="function")
def s3_resource(
    aws_service: services.AWSService,
) -> T.Generator[s3.S3ServiceResource, None, None]:
    """Return an S3 resource from the AWS Service."""
    with moto.mock_aws():
        session = aws_service.session()
        resource = session.resource("s3", region_name=aws_service.region_name)
        yield resource  # https://docs.getmoto.org/en/latest/docs/getting_started.html#example-on-usage


@pytest.fixture(scope="function")
def s3_bucket(aws_service: services.AWSService, s3_resource: s3.S3ServiceResource) -> s3.Bucket:
    """Return an empty S3 bucket from the S3 resource."""
    # variables
    bucket_name = "my-bucket"
    region_name = aws_service.region_name
    bucket_conf = {"LocationConstraint": region_name}
    # resources
    bucket = s3_resource.Bucket(name=bucket_name)
    bucket.create(CreateBucketConfiguration=bucket_conf)  # type: ignore[arg-type]
    bucket.wait_until_exists()
    return bucket


@pytest.fixture(scope="function")
def s3_conf_object(s3_bucket: s3.Bucket) -> s3.Object:
    """Return an S3 object to an empty config."""
    key = "conf/config.yml"  # empty file
    obj = s3_bucket.Object(key=key)
    return obj


@pytest.fixture(scope="function")
def s3_inputs_object(s3_bucket: s3.Bucket, inputs_path: str) -> s3.Object:
    """Return an S3 object to the inputs dataset."""
    base = os.path.basename(inputs_path)
    key = os.path.join("data", base)
    obj = s3_bucket.Object(key=key)
    obj.upload_file(inputs_path)
    return obj


@pytest.fixture(scope="function")
def s3_targets_object(s3_bucket: s3.Bucket, targets_path: str) -> s3.Object:
    """Return an S3 object to the targets dataset."""
    base = os.path.basename(targets_path)
    key = os.path.join("data", base)
    obj = s3_bucket.Object(key=key)
    obj.upload_file(targets_path)
    return obj


@pytest.fixture(scope="function")
def s3_outputs_object(s3_bucket: s3.Bucket, outputs_path: str) -> s3.Object:
    """Return an S3 object to the outputs dataset."""
    base = os.path.basename(outputs_path)
    key = os.path.join("data", base)
    obj = s3_bucket.Object(key=key)
    obj.upload_file(outputs_path)
    return obj


@pytest.fixture(scope="function")
def s3_tmp_outputs_object(s3_bucket: s3.Bucket, outputs_path: str) -> s3.Object:
    """Return an temporary S3 object to the outputs dataset."""
    base = os.path.basename(outputs_path)
    key = os.path.join("tmp", base)
    obj = s3_bucket.Object(key=key)
    return obj


# %% - Resolvers


@pytest.fixture(scope="session", autouse=True)
def tests_path_resolver(tests_path: str) -> str:
    """Register the tests path resolver with OmegaConf."""

    def resolver() -> str:
        """Get tests path."""
        return tests_path

    omegaconf.OmegaConf.register_new_resolver("tests_path", resolver, use_cache=True, replace=False)
    return tests_path


@pytest.fixture(scope="function", autouse=True)
def tmp_path_resolver(tmp_path: str) -> str:
    """Register the tmp path resolver with OmegaConf."""

    def resolver() -> str:
        """Get tmp data path."""
        return tmp_path

    omegaconf.OmegaConf.register_new_resolver("tmp_path", resolver, use_cache=False, replace=True)
    return tmp_path


# %% - Signatures


@pytest.fixture(scope="session")
def signature(
    signer: signers.Signer, inputs: schemas.Inputs, outputs: schemas.Outputs
) -> signers.Signature:
    """Return the signature for the testing model."""
    return signer.sign(inputs=inputs, outputs=outputs)


# %% - Registries


@pytest.fixture(scope="session")
def saver() -> registries.CustomSaver:
    """Return the default model saver."""
    return registries.CustomSaver(path="custom-model")


@pytest.fixture(scope="session")
def loader() -> registries.CustomLoader:
    """Return the default model loader."""
    return registries.CustomLoader()


@pytest.fixture(scope="session")
def register() -> registries.MlflowRegister:
    """Return the default model register."""
    tags = {"context": "test", "role": "fixture"}
    return registries.MlflowRegister(tags=tags)


@pytest.fixture(scope="function")
def model_run(mlflow_service: services.MlflowService) -> T.Generator[mlflow.ActiveRun, None, None]:
    """Yield an mlflow run context for tracking."""
    run_config = mlflow_service.RunConfig(name="Custom-Run")
    with mlflow_service.run_context(config=run_config) as run:
        yield run


@pytest.fixture(scope="function")
def model_version(
    model: models.Model,
    inputs: schemas.Inputs,
    signature: signers.Signature,
    saver: registries.Saver,
    register: registries.MlflowRegister,
    model_run: mlflow.ActiveRun,
    mlflow_service: services.MlflowService,
) -> registries.Version:
    """Save and register the default model version."""
    info = saver.save(model=model, signature=signature, input_example=inputs)
    version = register.register(name=mlflow_service.registry_name, model_uri=info.model_uri)
    assert version.run_id == model_run.info.run_id
    return version


@pytest.fixture(scope="function")
def model_stage(
    model_version: registries.Version,
    mlflow_service: services.MlflowService,
) -> registries.Version:
    """Transition the default model version to staging."""
    stage = "Staging"
    client = mlflow_service.client()
    model_stage = client.transition_model_version_stage(
        name=mlflow_service.registry_name, version=model_version.version, stage=stage
    )
    return model_stage
