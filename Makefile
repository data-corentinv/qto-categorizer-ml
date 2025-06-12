# https://www.gnu.org/software/make/manual/make.html

## ENVS

# Default environment
# - use: make ENV=prod [target]
ENV ?= local

## VARS

$(info [Makefile] Loading common variables from env/base.env ...)
include env/base.env
$(info [Makefile] Loading specific variables from env/$(ENV).env ...)
include env/$(ENV).env

## HELP

.DEFAULT_GOAL:=help
help: ## List all the make tasks.
	@grep --no-filename "##" tasks/*.mk

## TASKS

# - commons
include tasks/helpers.mk

# - project
include tasks/builders.mk
include tasks/bumpers.mk
include tasks/checkers.mk
include tasks/cleaners.mk
include tasks/documenters.mk
include tasks/executers.mk
include tasks/formatters.mk
include tasks/inits.mk
include tasks/installers.mk
include tasks/publishers.mk
include tasks/reporters.mk

# - remote
include tasks/databricks.mk
# include tasks/sagemaker.mk
