# Imagen de inferencia multi-etapa: dependencias bloqueadas, usuario sin privilegios, HEALTHCHECK y sin datos de
# entrenamiento (solo los artefactos de app/models).
FROM python:3.12-slim AS builder
WORKDIR /w
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

FROM python:3.12-slim
RUN useradd -m appuser
COPY --from=builder /install /usr/local
WORKDIR /srv
COPY src/dl03 /srv/src/dl03
COPY configs /srv/configs
COPY app /srv/app
ENV PYTHONPATH=/srv/src:/srv MODELS_DIR=/srv/app/models PORT=8000
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os, urllib.request as u; u.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
