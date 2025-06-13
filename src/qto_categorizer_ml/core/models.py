"""Define trainable machine learning models."""

# %% IMPORTS

import abc
import typing as T

import numpy as np
import pydantic as pdt
import typing_extensions as TX
from sklearn import ensemble
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from qto_categorizer_ml.core import schemas

# %% TYPES

# Model params
ParamKey = str
ParamValue = T.Any
Params = dict[ParamKey, ParamValue]

# %% MODELS


class Model(abc.ABC, pdt.BaseModel, strict=True, frozen=False, extra="forbid"):
    """Base class for a project model.

    Use a model to adapt AI/ML frameworks.
    e.g., to swap easily one model with another.
    """

    KIND: str

    def get_params(self, deep: bool = True) -> Params:
        """Get the model params.

        Args:
            deep (bool, optional): ignored. Defaults to True.

        Returns:
            Params: internal model parameters.
        """
        params: Params = {}
        for key, value in self.model_dump().items():
            if not key.startswith("_") and not key.isupper():
                params[key] = value
        return params

    def set_params(self, **params: ParamValue) -> TX.Self:
        """Set the model params in place.

        Returns:
            TX.Self: instance of the model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @abc.abstractmethod
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets) -> TX.Self:
        """Fit the model on the given inputs and targets.

        Args:
            inputs (schemas.Inputs): model training inputs.
            targets (schemas.Targets): model training targets.

        Returns:
            TX.Self: instance of the model.
        """

    @abc.abstractmethod
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        """Generate outputs with the model for the given inputs.

        Args:
            inputs (schemas.Inputs): model prediction inputs.

        Returns:
            schemas.Outputs: model prediction outputs.
        """

    def get_internal_model(self) -> T.Any:
        """Return the internal model in the object.

        Raises:
            NotImplementedError: method not implemented.

        Returns:
            T.Any: any internal model (either empty or fitted).
        """
        raise NotImplementedError()


# %%BASELINE


class MostFrequentCategoryByMerchant(BaseEstimator, TransformerMixin):
    """Move to another dir"""

    merchant_to_category_: dict = {}

    def fit(self, X, y):
        """Fit method."""
        df = X.copy()
        df["CATEGORY"] = y

        merchant_label_counts = (
            df.groupby(["MERCHANT_NAME", "CATEGORY"]).size().reset_index(name="count")
        )
        most_frequent = merchant_label_counts.sort_values("count", ascending=False).drop_duplicates(
            "MERCHANT_NAME"
        )

        # Dictionnaire de correspondance MERCHANT_NAME -> catégorie la plus fréquente
        self.merchant_to_category_ = dict(
            zip(most_frequent["MERCHANT_NAME"], most_frequent["CATEGORY"])
        )

        return self

    def transform(self, X):
        """Transform method."""
        X_ = X.copy()
        X_["MOST_FREQUENT_CATEGORY"] = X_["MERCHANT_NAME"].map(self.merchant_to_category_)
        outputs = X_.MOST_FREQUENT_CATEGORY.values
        # return [int(i) for i in np.nan_to_num(outputs, nan=25.0)]
        return np.nan_to_num(outputs, nan=25.0).astype(int)


class BaselineModel(Model):
    """Baseline model: takes to most frequent category of a marchant).
    Accuracy estimated to be ~60%.

    Parameters:
        corespondance_table_path (str): path to save the corespondance table.
    """

    KIND: T.Literal["BaselineModel"] = "BaselineModel"

    # params
    corespondance_table_path: str = "data/corespondance_table.csv"
    pipeline: None = None
    le: None = None

    def fit(self, inputs, targets):
        self.le = LabelEncoder()
        targets_ = self.le.fit_transform(targets)

        self.pipeline = Pipeline([("merchant_category_encoder", MostFrequentCategoryByMerchant())])
        self.pipeline.fit(inputs, targets_)

    def predict(self, inputs):
        outputs = self.pipeline.transform(inputs)
        return self._decode_target(outputs)

    def _encode_target(self, outputs):
        return self.le.transform(outputs)

    def _decode_target(self, outputs_encoded):
        return self.le.inverse_transform(outputs_encoded)


class SKLearnPipelineModel(Model):
    """SKLearn Pipeline model: takes to most frequent category of a marchant).

    Parameters:
        max_features_desc (int): Maximum number of features to keep in the TfidfVectorizer for the DESCRIPTION column.
        max_features_merch (int): Maximum number of features to keep in the TfidfVectorizer for the MERCHANT_NAME column.
        n_components_desc (int): Number of components to retain after dimensionality reduction (e.g., TruncatedSVD) on DESCRIPTION vectors.
        n_components_merch (int): Number of components to retain after dimensionality reduction on MERCHANT_NAME vectors.
        random_state (int): Random seed for reproducibility across training runs.
        n_estimators (int): Number of trees in the RandomForestClassifier.
        max_depth (int): Maximum depth of each tree in the forest.
        n_jobs (int): Number of CPU cores to use for parallel processing. Use -1 to use all available cores.
    """

    KIND: T.Literal["SKLearnPipelineModel"] = "SKLearnPipelineModel"

    # params

    max_features_desc: int = 1000
    n_components_desc: int = 50
    max_features_merch: int = 500
    n_components_merch: int = 30
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int = 30
    n_jobs: int = -1

    pipeline: None = None
    le: None = None

    def _create_pipeline(self) -> Pipeline:
        """Create SKLearn Pipeline."""
        desc_pipe = make_pipeline(
            TfidfVectorizer(max_features=self.max_features_desc),
            TruncatedSVD(n_components=self.n_components_desc, random_state=self.random_state),
        )

        merch_pipe = make_pipeline(
            TfidfVectorizer(max_features=self.max_features_merch),
            TruncatedSVD(n_components=self.n_components_merch, random_state=self.random_state),
        )

        type_pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"))

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["AMOUNT"]),
                ("cat", type_pipe, ["TYPE_OF_PAYMENT"]),
                ("desc", desc_pipe, "DESCRIPTION"),
                ("merchant", merch_pipe, "MERCHANT_NAME"),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    ensemble.RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        return pipeline

    def fit(self, inputs, targets):
        """Fit sklearn pipeline."""
        self.le = LabelEncoder()
        targets_ = self.le.fit_transform(targets)

        self.pipeline = self._create_pipeline()
        self.pipeline.fit(inputs, targets_)

    def predict(self, inputs):
        """Generate prediction from sklearn pipeline."""
        outputs = self.pipeline.predict(inputs)
        return self._decode_target(outputs)

    def _encode_target(self, outputs):
        """Encode target."""
        return self.le.transform(outputs)

    def _decode_target(self, outputs_encoded):
        """Decode target."""
        return self.le.inverse_transform(outputs_encoded)


ModelKind = BaselineModel | SKLearnPipelineModel
