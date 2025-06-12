# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC **The goal of this notebook is to demonstrate how you can synchronize your local code base with Databricks.**
# MAGIC
# MAGIC **Prerequisites**:
# MAGIC - You should install all libraries required by your package before running this notebook
# MAGIC     - e.g., Run this command on Databricks notebook: `%pip install awswrangler pandera`
# MAGIC     - note: you can use `make DATABRICKS_CLUSTER_ID=<your-databricks-cluster-id> databricks-library` on your laptop to automate this process
# MAGIC
# MAGIC **Actions required**:
# MAGIC 1. Enable the [autoreload extension](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html) to automatically update imported modules in notebooks
# MAGIC 2. Import the package module where you added the code to test that the module is accessible
# MAGIC 3. Extend the Python system path to include the source folder of the project repository (i.e., src/)
# MAGIC
# MAGIC You have two options to synchronize the content of your local repository to the Databricks repository:
# MAGIC 1. **Databricks CLI**: A CLI tool developed by Databricks https://docs.databricks.com/en/dev-tools/cli/sync-commands.html
# MAGIC     - Run `make databricks-sync` to automatically sync the repositories on code change
# MAGIC     - Note: this synchronization is uni-directional (from local to Databricks repository)
# MAGIC 2. **Git**: A feature of Databricks Web UI https://docs.databricks.com/repos/sync-remote-repo.html
# MAGIC     - Right-lick on your git repository on Databricks to pull the latest changes from the local repository
# MAGIC     - Note: this option is more complex, time-consumming, and error prone. We recommend using Databricks CLI.
# MAGIC
# MAGIC **We recommend you to commit all notebook changes on Databricks, and all other files on your computer.**

# COMMAND ----------

# MAGIC %md
# MAGIC # EXTENDS

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC # IMPORTS

# COMMAND ----------

import sys

SOURCES = "../src/"
if SOURCES not in sys.path:
    print("Adding src/ to sys.path ...")
    sys.path.insert(0, SOURCES)
    print("DONE")

# COMMAND ----------

import qto_categorizer_ml as src

# COMMAND ----------

# MAGIC %md
# MAGIC # TESTING
# MAGIC
# MAGIC Run `make databricks-sync` on your computer to synchronize the local and remote repositories.
# MAGIC
# MAGIC If you define a new variable in the `__init__` module, you should see it after 1-2 seconds.

# COMMAND ----------

src.__file__

# COMMAND ----------

dir(src)
