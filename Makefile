.PHONY: install dev bootstrap test lint eval api ui up up-prod down logs sft clean

install:        ## Install paket + dependensi dev/ui
	pip install -e ".[dev,ui,pdf]"

bootstrap:      ## Data sintetis + index + classifier + dataset SFT
	minesense bootstrap

test:
	pytest -q

lint:
	ruff check src tests ui

eval:           ## Evaluasi + quality gates (exit 1 bila gagal)
	minesense eval

api:
	uvicorn minesense.api.main:app --reload --port 8000

ui:
	MINESENSE_API_URL=http://localhost:8000 streamlit run ui/app.py

up:             ## Jalankan stack lokal (docker)
	docker compose up -d --build

up-prod:        ## Stack produksi + HTTPS (Caddy)
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

sft:            ## Dataset fine-tuning LLM
	minesense sft

clean:
	rm -rf artifacts/index artifacts/models artifacts/reports .pytest_cache
