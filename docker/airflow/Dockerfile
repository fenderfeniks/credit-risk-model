FROM apache/airflow:2.9.2-python3.10
 
USER root
 
# docker-cli нужен только для DockerOperator (общается с хостовым docker.sock,
# который проброшен в docker-compose.yml). Никаких build-essential/git/uv —
# Airflow больше не импортирует и не запускает твой ML-код напрямую,
# он только поднимает контейнеры app-api / app-train через Docker API.
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 
USER airflow
 
# Провайдер для DockerOperator
RUN pip install --no-cache-dir apache-airflow-providers-docker==3.9.0