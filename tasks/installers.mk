## INSTALLERS

install-dev: ## Install the project in dev mode.
	poetry install --all-extras

install-prod: ## Install the project in prod mode.
	poetry install --only main

installers: install-dev ## Run all the installers.
