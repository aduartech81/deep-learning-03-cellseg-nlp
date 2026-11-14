# Comandos únicos de reproducción, pruebas, imagen y servicio.
PROFILE ?= full
.PHONY: reproduce smoke test lint build serve book

reproduce:          ## corre todos los capítulos con los datos reales (requiere GPU y token de Kaggle)
	DL03_PROFILE=$(PROFILE) bash scripts/run_chapters.sh

smoke:              ## prueba de punta a punta con datos sintéticos (CPU, minutos)
	DL03_PROFILE=smoke bash scripts/run_chapters.sh

test:
	DL03_PROFILE=smoke pytest -q tests/

lint:
	ruff check src app tests
	mypy src/dl03/rle.py src/dl03/seg_metrics.py src/dl03/stats.py src/dl03/calibration.py src/dl03/drift.py app

build:
	docker build -t dl03-api .

serve:
	docker run --rm -p 8000:8000 dl03-api

book:
	jupyter-book build .
