## EXECUTERS

execute-%: ## Execute the project task (e.g., make execute-training).
	poetry run $(PACKAGE_NAME) conf/$(ENV)/$*.yml
