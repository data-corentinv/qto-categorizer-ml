## DATABRICKS

## - SYNC

databricks-sync: isset-EMAIL ## Sync local files with Databricks.
	databricks --profile=$(DATABRICKS_PROFILE) sync --watch src/ /Users/$(EMAIL)/$(GITHUB_REPOSITORY)/src/

# - BUNDLE

databricks-bundle-%: ## Execute a job on Databricks with Bundle.
	databricks --profile=$(DATABRICKS_PROFILE) bundle validate
	databricks --profile=$(DATABRICKS_PROFILE) bundle deploy --var="job_name=$*"
	databricks --profile=$(DATABRICKS_PROFILE) bundle run main

## - CLUSTERS

databricks-cluster-start: isset-DATABRICKS_CLUSTER_ID ## Start the Databricks cluster.
	databricks --profile=$(DATABRICKS_PROFILE) clusters start $(DATABRICKS_CLUSTER_ID)

databricks-cluster-restart: isset-DATABRICKS_CLUSTER_ID ## Restart the Databricks cluster.
	databricks --profile=$(DATABRICKS_PROFILE) clusters restart $(DATABRICKS_CLUSTER_ID)

databricks-cluster-shutdown: isset-DATABRICKS_CLUSTER_ID ## Shutdown the Databricks cluster.
	databricks --profile=$(DATABRICKS_PROFILE) clusters delete $(DATABRICKS_CLUSTER_ID)

## - LIBRARIES

databricks-library-upload: build-wheel ## Upload the library to Databricks.
	databricks --profile=$(DATABRICKS_PROFILE) fs cp --overwrite "$(PACKAGE_WHEEL_PATH)" "$(DATABRICKS_PACKAGE_WHEEL_PATH)"

databricks-library-install: isset-DATABRICKS_CLUSTER_ID ## Install the library on Databricks.
	databricks --profile=$(DATABRICKS_PROFILE) libraries install --json='{"cluster_id": "$(DATABRICKS_CLUSTER_ID)", "libraries": [{"whl": "$(DATABRICKS_PACKAGE_WHEEL_PATH)"}]}'

databricks-library-uninstall: isset-DATABRICKS_CLUSTER_ID ## Uninstall the library on Databricks.
	databricks --profile=$(DATABRICKS_PROFILE) libraries uninstall --json='{"cluster_id": "$(DATABRICKS_CLUSTER_ID)", "libraries": [{"whl": "$(DATABRICKS_PACKAGE_WHEEL_PATH)"}]}'

databricks-library: databricks-library-upload databricks-library-install ## Run all the Databricks library tasks.


## - CREATE REPO

init-databricks-repo: isset-EMAIL ## Initialize the Databricks repository for the user.
	databricks --profile=$(DATABRICKS_PROFILE) repos create --url=$(GITHUB_FULL_URL) --path=/Users/$(EMAIL)/$(GITHUB_REPOSITORY)

init-databricks: init-databricks-repo ## Run all the main Databricks tasks.

## - CREATE MLFLOW EXPERIMENT AND REGISTRY

init-mlflow-experiment: ## Initialize an MLflow Experiment to manage runs.
	databricks --profile=$(DATABRICKS_PROFILE) experiments create-experiment "$(MLFLOW_EXPERIMENT_NAME)" --artifact-location="$(MLFLOW_ARTIFACT_LOCATION)"

init-mlflow-registry: ## Initialize an MLflow Registry to manage versions.
	databricks --profile=$(DATABRICKS_PROFILE) model-registry create-model "$(MLFLOW_MODEL_NAME)"

init-mlflow: init-mlflow-experiment init-mlflow-registry ## Run all the main Databricks MLflow tasks.
