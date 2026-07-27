PYTHON := .venv/bin/python
BACKEND_PYTHONPATH := backend/src

.PHONY: backend-api backend-test data data-vision fashion-evaluate fashion-index fashion-match frontend-build frontend-dev frontend-test qdrant test

backend-api:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m uvicorn --app-dir backend api.main:app --host 0.0.0.0 --port 8080

backend-test:
	cd backend && PYTHONPATH=src ../$(PYTHON) -m pytest -q

data:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m data_pipeline.prepare_real_data

data-vision:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m data_pipeline.prepare_real_data --with-vision

qdrant:
	docker compose -f docker-compose.qdrant.yml up -d

fashion-index:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m fashion_matching.index_catalog --manifest $(MANIFEST) --activate

fashion-match:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m fashion_matching.match --query-manifest $(MANIFEST) --output $(OUTPUT)

fashion-evaluate:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m fashion_matching.evaluation --results $(RESULTS) --labels $(LABELS)

frontend-build:
	npm --prefix frontend run build

frontend-dev:
	npm --prefix frontend run dev

frontend-test:
	npm --prefix frontend run test

test: backend-test frontend-test
