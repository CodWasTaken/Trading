.PHONY: install test lint run dashboard docker

install:
	python -m pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

run:
	uvicorn trading_app.main:app --reload --port 8000

dashboard:
	cd apps/dashboard && npm run dev

docker:
	docker compose up --build
