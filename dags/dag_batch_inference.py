"""
DAG: batch_inference

Назначение:
    Ежедневный прогон новых данных через уже обученную и задеплоенную модель.
    Не обучает и не переобучает модель — только инференс на готовых артефактах.

Архитектура (DockerOperator):
    Инференс не требует mlflow/optuna/matplotlib — это lazy-инициализация
    в main.py (tracker создаётся только для mode in train/tune/evaluate),
    поэтому run_inference использует ЛЁГКИЙ api-образ
    (sber_autopodpiska-api:latest, core + api extra), а не train-образ.

    У api-образа CMD по умолчанию — uvicorn (он же постоянно работающий
    сервис api в docker-compose.yml). Для batch-инференса CMD переопределяется
    через параметр entrypoint DockerOperator на `python -m main`, и в command
    передаются Hydra-overrides (mode=inference и путь к файлу).

Зависимости:
    Файл с новыми данными должен лежать по пути data/raw/new_data.csv
    в корне проекта на хосте — это PythonOperator-сенсор проверяет внутри
    airflow-контейнера (см. bind mount data/raw в docker-compose.yml),
    а DockerOperator пробрасывает ТУ ЖЕ хостовую папку в inference-контейнер
    через переменную окружения HOST_PROJECT_ROOT (см. docker-compose.yml).
"""
from __future__ import annotations

import os
import pendulum

from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.exceptions import AirflowRescheduleException, AirflowTaskTimeout
from airflow.operators.python import PythonOperator
from docker.types import Mount


# HOST_PROJECT_ROOT прокидывается в airflow-scheduler из .env (см. docker-compose.yml).
# Нужен, потому что DockerOperator монтирует bind-путь С ХОСТА в новый контейнер —
# путь внутри самого airflow-контейнера (AIRFLOW_DATA_PATH выше) для этого не подходит.
HOST_PROJECT_ROOT = os.environ.get("HOST_PROJECT_ROOT", "")

API_IMAGE = "credit-risk-model-api:latest"
DOCKER_NETWORK = "deploy_ml_network"  # см. примечание в dag_retrain_pipeline.py
DOCKER_URL = "unix://var/run/docker.sock"

# Путь внутри inference-контейнера, куда монтируется data/raw с хоста
DATA_FILE_NAME = "train_data_0.pq"
AIRFLOW_DATA_PATH = f"/opt/airflow/data/raw/{DATA_FILE_NAME}"


def check_file_readiness(**context):
    ti = context['ti']
    start_time = ti.start_date

    if start_time and (pendulum.now() - start_time).total_seconds() > (60 * 15):
        raise AirflowTaskTimeout("Превышен таймаут ожидания файла (15 минут).")

    if not os.path.exists(AIRFLOW_DATA_PATH):
        print(f"Файл {AIRFLOW_DATA_PATH} еще не появился.")
        print("Режим reschedule: освобождаем воркер на 30 секунд...")
        raise AirflowRescheduleException(pendulum.now().add(seconds=30))

    print(f"Отлично! Файл найден: {AIRFLOW_DATA_PATH}. Пайплайн идет дальше.")


default_args = {
    "owner": "ml-team",
    "retries": 0,
    "retry_delay": pendulum.duration(minutes=5),
}


def notify_completion(**context):
    print(f"Batch inference завершён успешно. DAG run: {context['run_id']}")


def build_inference_mounts() -> list[Mount]:
    if not HOST_PROJECT_ROOT:
        raise RuntimeError(
            "HOST_PROJECT_ROOT не задан. Укажи его в .env рядом с docker-compose.yml "
            "(абсолютный путь к корню проекта на хосте) — без него DockerOperator "
            "не сможет смонтировать data/raw в inference-контейнер."
        )
    return [
        # Монтируем сырые данные
        Mount(
            source=f"{HOST_PROJECT_ROOT}/data/raw",
            target="/app/data/raw",
            type="bind",
        ),
        # ДОБАВЛЯЕМ: Монтируем обработанные данные / кэш витрин (на всякий случай)
        Mount(
            source=f"{HOST_PROJECT_ROOT}/data/processed",
            target="/app/data/processed",
            type="bind",
        ),
        # ИСПРАВЛЯЕМ: Меняем "volume" на надежный "bind" к локальной папке с моделями
        Mount(
            source=f"{HOST_PROJECT_ROOT}/models",
            target="/app/models",
            type="bind",
        ),
    ]


with DAG(
    dag_id="batch_inference",
    description="Ежедневный батч-инференс на новых данных",
    default_args=default_args,
    schedule=None,
    start_date=pendulum.datetime(2026, 6, 1, tz="UTC"),
    catchup=False,
    tags=["ml", "inference", "production", "docker-operator"],
) as dag:

    # 1. Ждем файл на той же файловой системе, что видит сам Airflow
    # (bind mount data/raw в docker-compose.yml у airflow-scheduler).
    check_data_exists = PythonOperator(
        task_id="check_data_exists",
        python_callable=check_file_readiness,
        retries=2,
    )

    # 2. Инференс в изолированном контейнере из api-образа.
    # api-образ лёгкий (core + api extra), без mlflow/optuna — подходит
    # для batch inference, который их не использует (см. main.py).
    run_inference = DockerOperator(
        task_id="run_inference",
        image=API_IMAGE,
        api_version="auto",
        auto_remove="success",
        docker_url=DOCKER_URL,
        network_mode=DOCKER_NETWORK,
        mount_tmp_dir=False,
        mounts=build_inference_mounts(),
        # У api-образа CMD = uv run uvicorn (это сервис, не batch-команда).
        # Для разового инференса переопределяем entrypoint на тот же
        # orchestrator, что использует train-образ. `uv run` обязателен:
        # uv sync всегда ставит зависимости в .venv (даже при
        # UV_SYSTEM_PYTHON=1 — она влияет только на uv pip), поэтому
        # голый "python" не найдётся в системном PATH контейнера.

        entrypoint=["uv", "run", "--no-sync", "python", "-m", "main"],
        command="mode=inference paths.data_file_name=new_data.parquet",
        environment={
            "PYTHONUNBUFFERED": "1"
        },
    )

    notify_done = PythonOperator(
        task_id="notify_done",
        python_callable=notify_completion,
    )

    #run_inference >> notify_done
    check_data_exists >> run_inference >> notify_done