BOLD := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: generate # Generate the ipynbs
generate:
	@echo "${BOLD}Generating lecture notebooks...${RESET}"
	@go run ./001_features_showcase/main.go
	@go run ./002_matplotlib/main.go
	@go run ./003_sklearn/main.go


.PHONY: notebook # Run the notebook server
notebook: 
	@echo "${BOLD}Running notebook server...${RESET}"
	@uv run --with jupyter jupyter lab

.PHONY: help # Display the help message
help:
	@echo "${BOLD}Available targets:${RESET}"
	@cat Makefile | grep '.PHONY: [a-z]' | sed 's/.PHONY: / /g' | sed 's/ #* / - /g'
