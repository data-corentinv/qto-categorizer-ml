## CHECKERS

check-types: ## Check the project code types with mypy.
	poetry run mypy src/qto_categorizer_ml/ src/tests/

check-tests: ## Check the project unit tests with pytest.
	poetry run pytest --numprocesses="auto" src/tests/ --disable-pytest-warnings

check-format: ## Check the project source format with ruff.
	poetry run ruff format --check src/qto_categorizer_ml/ src/tests/

check-poetry: ## Check the project pyproject.toml with poetry.
	poetry check --lock

check-quality: ## Check the project code quality with ruff.
	poetry run ruff check src/qto_categorizer_ml/ src/tests/

check-security: ## Check the project code security with bandit.
	poetry run bandit --recursive --configfile=pyproject.toml src/

check-coverage: ## Check the project test coverage with coverage.
	poetry run pytest --cov=src/qto_categorizer_ml/ --cov-fail-under=80 --numprocesses="auto" --disable-pytest-warnings src/tests/

checkers: check-types check-format check-poetry check-quality check-security check-coverage ## Run all the checkers.
