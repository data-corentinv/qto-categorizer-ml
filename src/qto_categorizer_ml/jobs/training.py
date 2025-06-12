# %% IMPORTS

import typing as T

import pydantic as pdt
import typing_extensions as TX

from qto_categorizer_ml.core import metrics, models, schemas
from qto_categorizer_ml.io import datasets, registries, services
from qto_categorizer_ml.jobs import base
from qto_categorizer_ml.utils import signers, splitters

# %% JOBS


class TrainingJob(base.Job):
    """Train and register a single AI/ML model.

    Parameters:
        run_config (services.MlflowService.RunConfig): run config.
        inputs (datasets.ReaderKind): reader for the inputs data.
        model (models.ModelKind): machine learning model to train.
        scorers (metrics.MetricsKind): metrics for the scoring.
        splitter (splitters.SplitterKind): data sets splitter.
        saver (registries.SaverKind): model saver.
        signer (signers.SignerKind): model signer.
        registry (registries.RegisterKind): model register.
    """

    KIND: T.Literal["TrainingJob"] = "TrainingJob"

    # Run
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(name="Training")

    # Data
    inputs: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")

    # Model
    model: models.ModelKind = pdt.Field(..., discriminator="KIND")

    # Metric
    scorers: metrics.MetricsKind = [
        metrics.SklearnMetric(name="accuracy_score", greater_is_better=True)
    ]

    # Splitter
    splitter: splitters.SplitterKind = pdt.Field(
        splitters.TrainTestSplitter(), discriminator="KIND"
    )

    # Saver
    saver: registries.SaverKind = pdt.Field(registries.CustomSaver(), discriminator="KIND")

    # Signer
    signer: signers.SignerKind = pdt.Field(signers.InferSigner(), discriminator="KIND")

    # Registrer
    # - avoid shadowing pydantic `register` pydantic function
    registry: registries.RegisterKind = pdt.Field(registries.MlflowRegister(), discriminator="KIND")

    @TX.override
    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)
        # - mlflow
        client = self.mlflow_service.client()
        logger.info("With client: {}", client)

        with (
            self.mlflow_service.run_context(config=self.run_config) as run,
        ):
            logger.info("With Mlflow Run: {}", run.info)
            # data
            # - read inputs, targets
            logger.info("Read inputs: {}", self.inputs)
            inputs_ = self.inputs.read()
            # - feature selections
            features = ["AMOUNT", "TYPE_OF_PAYMENT", "MERCHANT_NAME", "DESCRIPTION"]
            target = "CATEGORY"
            inputs = inputs_[features + [target]].drop_duplicates()
            targets = inputs.pop(target)
            inputs = schemas.FeatInputsSchema.check(inputs)
            logger.debug("- Inputs shape: {}", inputs.shape)
            # - missing values cleaning
            inputs["MERCHANT_NAME"] = inputs.MERCHANT_NAME.fillna("No marchant")
            inputs["DESCRIPTION"] = inputs.DESCRIPTION.fillna("No Description")
            inputs["TYPE_OF_PAYMENT"] = inputs.TYPE_OF_PAYMENT.fillna("No type payment")

            # splitter
            logger.info("With splitter: {}", self.splitter)
            # - index
            train_index, test_index = next(self.splitter.split(inputs=inputs, targets=targets))
            # - inputs
            inputs_train = T.cast(schemas.FeatInputsSchema, inputs.iloc[train_index][features])
            inputs_test = T.cast(schemas.FeatInputsSchema, inputs.iloc[test_index][features])
            logger.debug("- Inputs train shape: {}", inputs_train.shape)
            logger.debug("- Inputs test shape: {}", inputs_test.shape)
            # - targets
            targets_train = T.cast(schemas.Targets, targets.iloc[train_index])
            targets_test = T.cast(schemas.Targets, targets.iloc[test_index])
            logger.debug("- Targets train shape: {}", targets_train.shape)
            logger.debug("- Targets test shape: {}", targets_test.shape)

            # model
            logger.info("Fit model: {}", self.model)
            self.model.fit(inputs=inputs_train, targets=targets_train)

            # outputs
            logger.info("Predict outputs: {}", len(inputs_test))
            outputs_test = self.model.predict(inputs=inputs_test)
            logger.debug("- Outputs test shape: {}", outputs_test.shape)

            # scorers
            for scorer in self.scorers:
                logger.info("Compute metric: {}", scorer)
                targets_test_encoded = self.model._encode_target(targets_test)
                outputs_test_encoded = self.model._encode_target(outputs_test)
                score = scorer.score(targets=targets_test_encoded, outputs=outputs_test_encoded)
                client.log_metric(run_id=run.info.run_id, key=scorer.name, value=score)
                logger.debug("- Metric score: {}", score)

            # fit models on entire dataset before saving into mlflow model registry
            logger.info("Train model on entire dataset..")
            self.model.fit(inputs=inputs, targets=targets)

            # signer
            logger.info("Sign model: {}", self.signer)
            model_signature = self.signer.sign(
                inputs=inputs_test[features].dropna(), outputs=outputs_test
            )
            logger.debug("- Model signature: {}", model_signature.to_dict())

            # saver
            logger.info("Save model: {}", self.saver)
            model_info = self.saver.save(
                model=self.model, signature=model_signature, input_example=inputs
            )
            logger.debug("- Model URI: {}", model_info.model_uri)

            # register
            logger.info("Register model: {}", self.registry)
            model_version = self.registry.register(
                name=self.mlflow_service.registry_name, model_uri=model_info.model_uri
            )
            logger.debug("- Model version: {}", model_version)
        return locals()
