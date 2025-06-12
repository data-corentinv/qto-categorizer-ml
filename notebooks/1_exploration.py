# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC **The goal of this notebook is to explore the problem and come up with a first AI/ML prototype.**
# MAGIC
# MAGIC We use [scikit-learn](https://scikit-learn.org/stable/) to explore the dataset, prepare the features, and train an ML model.
# MAGIC
# MAGIC We then use [MLflow](https://mlflow.org/) to track experiments and register the model in Databricks.
# MAGIC
# MAGIC You can also configure the notebook fields above with Databricks [widgets](https://docs.databricks.com/notebooks/widgets.html).

# COMMAND ----------

# MAGIC %md
# MAGIC # IMPORTS

# COMMAND ----------

import mlflow
import pandas as pd
from sklearn import datasets, ensemble, model_selection, pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC # WIDGETS

# COMMAND ----------

dbutils.widgets.removeAll()
# Spliting
dbutils.widgets.text("test_size", "0.25")
# Modeling
dbutils.widgets.text("max_depth", "5")
dbutils.widgets.text("random_state", "0")
dbutils.widgets.text("n_estimators", "20")

# COMMAND ----------

# MAGIC %md
# MAGIC # CONFIGS

# COMMAND ----------

# Spliting
test_size = float(dbutils.widgets.get("test_size"))
# Modeling
max_depth = int(dbutils.widgets.get("max_depth"))
random_state = int(dbutils.widgets.get("random_state"))
n_estimators = int(dbutils.widgets.get("n_estimators"))

# COMMAND ----------

# MAGIC %md
# MAGIC # SETTINGS

# COMMAND ----------

# MLflow
EXPERIMENT_NAME = "/experiments/qto-catgorizer-catgorizer-ml"
MODEL_NAME = "qto-catgorizer-ml"
RUN_NAME = "Exploration"

# COMMAND ----------

# MLflow
mlflow.autolog()
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC # DATASETS
# MAGIC
# MAGIC We use the [Wine dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/) to showcase this project.
# MAGIC
# MAGIC This dataset is associated with a classification task.

# COMMAND ----------

data, target = datasets.load_wine(return_X_y=True, as_frame=True)
print("Data shape:", data.shape, "; Target shape:", target.shape)

# COMMAND ----------

data.info()
data.head()

# COMMAND ----------

target.info()
target.value_counts().to_frame()

# COMMAND ----------

# MAGIC %md
# MAGIC # ANALYSIS

# COMMAND ----------

dbutils.data.summarize(data)

# COMMAND ----------

# MAGIC %md
# MAGIC # SPLITTING

# COMMAND ----------

data_train, data_test, target_train, target_test = model_selection.train_test_split(
    data,
    target,
    test_size=test_size,
    random_state=random_state,
)
print("Data train:", data_train.shape, "; Data test:", data_test.shape)
print("Target train:", data_train.shape, "; Target test:", data_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC # MODELLING
# MAGIC
# MAGIC We create a [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) from scikit-learn as a simple baseline.
# MAGIC
# MAGIC The model is trained on the train data, and evaluated on the test data.

# COMMAND ----------

classifier = ensemble.RandomForestClassifier(
    max_depth=max_depth,
    n_estimators=n_estimators,
    random_state=random_state,
)
model = pipeline.Pipeline(steps=[("classifier", classifier)])
model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training

# COMMAND ----------

with mlflow.start_run(run_name=RUN_NAME) as run:
    model.fit(data_train, target_train)
run.info

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation

# COMMAND ----------

score = model.score(data_test, target_test)
print("Evaluation score:", score)
