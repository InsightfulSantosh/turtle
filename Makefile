PYTHON := .venv/bin/python
BACKEND_PYTHONPATH := backend/src

.PHONY: api-dev backend-test frontend-build frontend-dev frontend-test lint test

api-dev:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m uvicorn analysis_service.api:app --reload --port 8000

backend-test:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m pytest -q

lint:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m ruff check backend/src backend/tests
	npm --prefix frontend run lint

frontend-build:
	npm --prefix frontend run build

frontend-dev:
	npm --prefix frontend run dev

frontend-test:
	npm --prefix frontend run test

test: backend-test frontend-test
