# ============================================================
# API-образ: serving FastAPI-сервиса (/predict, /health, /metrics)
# Состав: core + api extra. БЕЗ mlflow/optuna/dvc/duckdb — инференс
# их не использует (см. ленивую инициализацию ArtifactManager в main.py).
# ============================================================
FROM python:3.10-slim
 
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
 
ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1
 
WORKDIR /app
 
# Кэшируемый слой: зависимости ставятся отдельно от кода,
# чтобы правки в src/ не пересобирали окружение с нуля.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --extra api --no-install-project
 
# Код проекта
COPY src/ src/
COPY sql/ sql/
COPY configs/ configs/
COPY main.py main.py
 
RUN uv sync --no-dev --extra api
 
EXPOSE 8000
 
# Точка входа — src.api.main:app (НЕ api.main:app — модуль живёт под src/,
# и сам src/api/main.py делает `from src.api.schemas import ...`,
# то есть ожидает себя именно по этому пути).
#
# ВАЖНО: `uv sync` (в отличие от `uv pip install`) всегда создаёт .venv,
# даже при UV_SYSTEM_PYTHON=1 — эта переменная влияет только на `uv pip`.
# Поэтому uvicorn физически лежит в /app/.venv/bin/uvicorn, а не в
# системном PATH. `uv run` сам находит и активирует .venv перед запуском.
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]