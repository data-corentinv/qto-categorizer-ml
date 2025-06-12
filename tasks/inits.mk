## INITS

## - Install

init-install: install-dev ## Initialize all the install tasks.

## - Sonar Quality

init-sonar-project: isset-SONAR_TOKEN ## Initialize a Sonar project for reporting code quality.
	@curl -X POST --fail-with-body "https://$(SONAR_TOKEN)@$(SONAR_HOST)/api/projects/create" \
		--data "organization=$(SONAR_ORGANIZATION)&name=$(SONAR_REPOSITORY)&project=$(SONAR_PROJECT)"

init-sonar-branch: isset-SONAR_TOKEN ## Configure the default Sonar branch using GitHub main branch.
	@curl -X POST --fail-with-body "https://$(SONAR_TOKEN)@$(SONAR_HOST)/api/project_branches/rename" \
		--data "project=$(SONAR_PROJECT)&name=$(GITHUB_MAIN_BRANCH)"

init-sonar: init-sonar-project init-sonar-branch ## Run all the main Sonar tasks.
