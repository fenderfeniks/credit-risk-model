
# ============================================================
# Train-образ: batch-режимы корневого main.py (train / tune / evaluate /
# inference). Состав: core + train extra (mlflow, optuna, matplotlib,
# seaborn, dvc, duckdb). Не содержит fastapi-стека — этот образ никогда
# не поднимает HTTP-сервер, его команда подставляется снаружи Airflow'ом
# через DockerOperator (mode=train / mode=evaluate / ...).
# ============================================================
FROM python:3.10-slim
 
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
 
ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1
 
WORKDIR /app
 
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --extra train --no-install-project
 
COPY src/ src/
COPY configs/ configs/
COPY main.py main.py
COPY sql/ sql/
 
RUN uv sync --no-dev --extra train --no-editable
 
# Никакого CMD: это batch-образ. Команда (mode=train / mode=evaluate / ...)
# подставляется DockerOperator'ом в Airflow. ENTRYPOINT фиксирует
# интерпретатор и точку входа, Airflow передаёт только Hydra-overrides.
#
# `uv run` нужен по той же причине, что и в api-образе: `uv sync` всегда
# создаёт .venv, даже при UV_SYSTEM_PYTHON=1 (эта переменная влияет только
# на `uv pip`), поэтому python из .venv не виден напрямую в системном PATH.
ENTRYPOINT ["uv", "run", "python", "-m", "main"]