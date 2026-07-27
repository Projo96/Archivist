# Shortcuts for the most frequent commands.
# Run `make` on its own for the list.

.DEFAULT_GOAL := help
.PHONY: help setup models up down logs ingest reset ask health lint

help:  ## Show this list
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Create .env from the template and install Python dependencies
	@test -f .env || cp .env.example .env
	pip install -r requirements.txt

models:  ## Pull the Ollama models named in .env
	bash scripts/setup-models.sh

up:  ## Start the Docker stack (Qdrant + API + Open WebUI)
	docker compose up -d --build

down:  ## Stop the stack
	docker compose down

logs:  ## Follow the API logs
	docker compose logs -f rag-api

ingest:  ## Index the documents in DOCS_PATH
	cd app && python ingest.py

reset:  ## Wipe the collection and reindex everything
	cd app && python ingest.py --reset

ask:  ## Interactive query session in the terminal
	cd app && python query.py

health:  ## API status and indexed vector count
	@curl -s http://127.0.0.1:8000/health | python -m json.tool

lint:  ## Check that the sources compile
	python -m compileall -q app
