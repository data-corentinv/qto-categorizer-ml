## FORMATTERS

format-imports: ## Format the project imports.
	poetry run ruff check --fix --select=I src/qto_categorizer_ml/ src/tests/

format-sources: ## Format the project sources.
	poetry run ruff format src/qto_categorizer_ml/ src/tests/

formatters: format-imports format-sources ## Run all the formatters.
