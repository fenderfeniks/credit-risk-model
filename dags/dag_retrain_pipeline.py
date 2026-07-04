"""
DAG: retrain_pipeline

Назначение:
    Полный цикл переобучения модели: train -> evaluate.
    Деплой НЕ автоматический — после evaluate инженер должен изучить
    метрики в MLflow UI и вручную запустить отдельный DAG `deploy_model`
    через Airflow UI ("Trigger DAG"), передав версию модели для деплоя.

    Это намеренное архитектурное решение: на retrain-пайплайнах с
    бизнес-метриками (а не просто accuracy) автоматический деплой по
    порогу — частая причина инцидентов, если метрика на тесте улучшилась
    но модель деградировала на специфичном сегменте. Человек в контуре
    дешевле инцидента в проде.

Архитектура (DockerOperator):
    Airflow здесь — чистый оркестратор. Он не импортирует и не запускает
    ML-код напрямую (airflow-образ намеренно лёгкий, без mlflow/optuna/
    catboost). Каждая ML-таска поднимает ОТДЕЛЬНЫЙ контейнер из образа
    credit-risk-train:latest (core + train extra, см. pyproject.toml)
    через хостовый docker.sock, выполняет в нём `python -m main mode=...`
    и контейнер удаляется после завершения.

    train/evaluate-контейнеры подключены к сети ml_network, чтобы видеть
    mlflow:5000 и postgres:5432 по именам сервисов из docker-compose.yml.
    Volume models_data общий с сервисом api — train пишет туда веса,
    api их читает.

Запуск:
    Вручную через Airflow UI, либо по расписанию (например еженедельно)
    если команда уверена в стабильности пайплайна данных.
"""
from __future__ import annotations

import os
import pendulum

from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from docker.types import Mount

TRAIN_IMAGE = "credit-risk-train:latest"
# Имя сети Docker Compose генерирует как <имя_папки_проекта>_<имя_сети>.
# Если compose-файл лежит в configs/deploy/, имя проекта по умолчанию = "deploy",
# поэтому сеть = "deploy_ml_network". Если запускаешь compose с -p/COMPOSE_PROJECT_NAME,
# или имя папки другое — проверь `docker network ls` и поправь константу ниже.
DOCKER_NETWORK = "deploy_ml_network"
DOCKER_URL = "unix://var/run/docker.sock"

HOST_PROJECT_ROOT = os.environ.get("HOST_PROJECT_ROOT")

# Извлекаем переменные окружения, проброшенные в airflow-scheduler
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME")

default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}


def print_mlflow_reminder(**context):
    print(
        "Train + evaluate завершены.\n"
        "Проверьте метрики в MLflow UI (http://localhost:5000).\n"
        "Если всё устраивает, запустите DAG 'deploy_model' вручную."
    )


def common_docker_kwargs(task_id: str) -> dict:
    """Общие параметры DockerOperator, чтобы не дублировать в каждой таске."""
    return dict(
        task_id=task_id,
        image=TRAIN_IMAGE,
        api_version="auto",
        auto_remove="success",
        docker_url=DOCKER_URL,
        network_mode=DOCKER_NETWORK,
        mount_tmp_dir=False,
        mounts=[
            # Train пишет сюда веса/препроцессор/feature_schema —
            # тот же volume, что смонтирован в сервис api.
            Mount(source="deploy_models_data", target="/app/models", type="volume"),
            Mount(source=f"{HOST_PROJECT_ROOT}/configs", target="/app/configs", type="bind"),
            Mount(
                source=f"{HOST_PROJECT_ROOT}/data", 
                target="/app/data", 
                type="bind"
            ),
        ],
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            # Направляем DockerOperator на твой Windows-хост
            "POSTGRES_HOST": "host.docker.internal", 
            "POSTGRES_PORT": "5432",
            
        },
    )


# ============================================================
# ПАЙПЛАЙН ОБУЧЕНИЯ
# ============================================================
with DAG(
    dag_id="retrain_pipeline",
    description="Пайплайн переобучения: Train -> Evaluate",
    default_args=default_args,
    schedule=None,
    start_date=pendulum.datetime(2026, 6, 1, tz="UTC"),
    catchup=False,
    tags=["ml", "training", "manual-gate", "docker-operator"],
) as dag:

    train_model = DockerOperator(
        **common_docker_kwargs("train_model"),
        command="mode=train",
    )

    evaluate_model = DockerOperator(
        **common_docker_kwargs("evaluate_model"),
        command="mode=evaluate",
    )

    manual_gate_reminder = PythonOperator(
        **{"task_id": "manual_gate_reminder", "python_callable": print_mlflow_reminder}
    )

    train_model >> evaluate_model >> manual_gate_reminder


# ============================================================
# ПАЙПЛАЙН ДЕПЛОЯ
# ============================================================
with DAG(
    dag_id="deploy_model",
    description="Деплой модели в прод. Запускается вручную.",
    default_args=default_args,
    schedule=None,
    start_date=pendulum.datetime(2026, 6, 1, tz="UTC"),
    catchup=False,
    tags=["ml", "deploy", "manual-only"],
) as deploy_dag:

    deploy_artifacts = DockerOperator(
        task_id="deploy_artifacts",
        image="docker:cli",  # Официальный мелкий образ, где есть команда docker
        command="docker restart credit_risk_api",
        api_version="auto",
        auto_remove="success",
        # Используем ту же докер-сеть
        network_mode="deploy_ml_network", 
        # Пробрасываем сокет хоста прямо в этот мини-контейнер
        mounts=[
            Mount(source="/var/run/docker.sock", target="/var/run/docker.sock", type="bind")
        ],
    )

    deploy_artifacts