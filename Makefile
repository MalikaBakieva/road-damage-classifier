.PHONY: help install install-dev sample-data prepare train train-multiclass evaluate demo \
        serve ui docker-build docker-up docker-down test lint format mlflow clean

PYTHON  ?= python
CONFIG  ?= configs/binary.yaml
PORT    ?= 8000

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies (CPU torch)
	pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
	pip install -r requirements.txt
	pip install -e .

install-dev:  ## Install dev dependencies
	pip install -r requirements-dev.txt
	pip install -e .

sample-data:  ## Generate synthetic RDD2022-shaped data (no download needed)
	$(PYTHON) scripts/make_sample_data.py --per-country 120

prepare:  ## Build the crop dataset from data/raw
	$(PYTHON) -m rdc.data.prepare --config $(CONFIG)

train:  ## Train the binary model
	$(PYTHON) -m rdc.train --config configs/binary.yaml

train-multiclass:  ## Train the multiclass model
	$(PYTHON) -m rdc.train --config configs/multiclass.yaml

evaluate:  ## Evaluate on the held-out test split
	$(PYTHON) -m rdc.evaluate --config $(CONFIG)

demo:  ## Full pipeline on synthetic data (~3 min, CPU)
	$(PYTHON) scripts/train_demo_model.py

serve:  ## Run the inference API
	uvicorn rdc.api.main:app --host 0.0.0.0 --port $(PORT) --reload

ui:  ## Run the Streamlit demo
	streamlit run app/streamlit_app.py

mlflow:  ## Open the MLflow UI on :5000
	mlflow ui --backend-store-uri file:./mlruns --port 5000

docker-build:  ## Build the Docker images
	docker compose build

docker-up:  ## Start API + UI (http://localhost:8501)
	docker compose up -d

docker-down:  ## Stop the containers
	docker compose down

test:  ## Run the test suite
	pytest -v

test-fast:  ## Run tests, skipping the slow training ones
	pytest -v -m "not slow"

lint:  ## Lint and format check
	ruff check src tests scripts app
	ruff format --check src tests scripts app

format:  ## Auto-format
	ruff check --fix src tests scripts app
	ruff format src tests scripts app

clean:  ## Remove caches and generated artefacts
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
