PYTHON := .venv/bin/python
BACKEND_PYTHONPATH := backend/src

.PHONY: backend-api backend-test data frontend-build frontend-dev frontend-test test

backend-api:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m uvicorn --app-dir backend api.main:app --host 0.0.0.0 --port 8080

backend-test:
	cd backend && PYTHONPATH=src ../$(PYTHON) -m pytest -q

data:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m data_pipeline.prepare_real_data

frontend-build:
	npm --prefix frontend run build

frontend-dev:
	npm --prefix frontend run dev

frontend-test:
	npm --prefix frontend run test

test: backend-test frontend-test
