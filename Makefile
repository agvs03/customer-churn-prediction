.PHONY: install data train api docker test clean mlflow-ui

install:
	pip install -r requirements.txt

data:
	python data/generate_data.py --out data/churn_raw.csv

train:
	python -m src.train

api:
	uvicorn api.main:app --reload --port 8000

test:
	pytest -q

docker:
	docker compose up --build

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache mlruns mlflow.db
