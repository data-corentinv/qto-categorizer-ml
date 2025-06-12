## BUMPERS

## - Python

bump-python-version: isset-PYTHON_VERSION ## Bump the python version installed.
	echo "$(PYTHON_VERSION)" > .python-version

bump-python-project: isset-PYTHON_VERSION ## Bump the python version of the project.
	sed -i.bak 's/^python = .*/python = "^$(PYTHON_VERSION)"/' pyproject.toml && rm pyproject.toml.bak

bump-python-sonar: isset-PYTHON_VERSION ## Bump the python version of Sonar Cloud properties.
	sed -i.bak 's/^sonar.python.version=.*/sonar.python.version=$(PYTHON_VERSION)/' sonar-project.properties && rm sonar-project.properties.bak

## - Package

bump-package-project: isset-PACKAGE_VERSION ## Bump the package version of the project.
	sed -i.bak 's/^version = .*/version = "$(PACKAGE_VERSION)"/' pyproject.toml && rm pyproject.toml.bak

bump-package-sphinxdocs: isset-PACKAGE_VERSION ## Bump the package version of the documentation.
	sed -i.bak 's/^version = .*/version = "$(PACKAGE_VERSION)"/' docs/source/conf.py && rm docs/source/conf.py.bak

bump-package-sonar: isset-PACKAGE_VERSION ## Bump the package version of Sonar Cloud properties.
	sed -i.bak 's/^sonar.projectVersion=.*/sonar.projectVersion=$(PACKAGE_VERSION)/' sonar-project.properties && rm sonar-project.properties.bak

## - Bumpers

bump-python: bump-python-version bump-python-project bump-python-sonar ## Run all the python bumpers.

bump-package: bump-package-project bump-package-sphinxdocs bump-package-sonar ## Run all the package bumpers.
