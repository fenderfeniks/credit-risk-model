from __future__ import annotations

import logging
import mlflow
from typing import Dict, Any
from contextlib import contextmanager
from optuna_integration.mlflow import MLflowCallback

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Единый менеджер для сохранения метрик, параметров и файлов (паттерн Facade)."""

    def __init__(self, cfg, project_root):

        self.PROJECT_ROOT = project_root
        self.cfg = cfg
        # 1. Подключаем базу SQLite для метрик и параметров
        self.tracking_uri = cfg.logging.mlflow.tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # 2. Собираем железный абсолютный URI-путь для папки с тяжелыми моделями
        rel_artifact_path = cfg.logging.mlflow.artifact_uri_rel
        self.artifact_uri = (self.PROJECT_ROOT / rel_artifact_path).as_uri()

    def set_experiment(self, experiment_name: str, artifact_location: str = None):
        """Устанавливает или создает эксперимент с автоматической привязкой к file:/// URI."""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Если путь к артефактам не передан явно в аргументы метода,
            # берем наш железный self.artifact_uri, собранный в __init__ через .as_uri()
            loc = artifact_location if artifact_location else self.artifact_uri
            
            logger.info(f"Создание нового эксперимента '{experiment_name}' с локацией артефактов: {loc}")
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=loc
            )
            
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str = None, nested: bool = False):
        """Контекстный менеджер для управления жизненным циклом запуска с авто-версионированием."""
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            try:
                # 1. Вытаскиваем версии и чейнджлоги из наших новых конфигов
                tabular_cfg = self.cfg.get('data', {}).get('tabular', {})
                model_cfg = self.cfg.get('model', {})


                agg_version = tabular_cfg.get('aggrigation_version', 'v1.0.0')
                prep_version = tabular_cfg.get('preprocessing_version', '0.0.0')
                feat_version = tabular_cfg.get('features_version', '0.0.0')
                model_version = model_cfg.get('model_version', '0.0.0')


                agg_log = tabular_cfg.get('aggrigation_changelog', '')
                prep_log = tabular_cfg.get('preprocessing_changelog', '')
                feat_log = tabular_cfg.get('features_changelog', '')
                model_log = model_cfg.get('model_changelog', '')

                # 2. Логируем версии как параметры (чтобы по ним можно было фильтровать в таблице MLflow)
                mlflow.log_params({
                    "aggrigation_version": agg_version,
                    "version_preprocessing": prep_version,
                    "version_features": feat_version,
                    "version_model": model_version
                })

                # 3. Устанавливаем теги с описанием изменений (changelog улетает в метаданные рана)
                mlflow.set_tags({
                    "aggrigation_changelog": agg_log,
                    "preprocessing_changelog": prep_log,
                    "features_changelog": feat_log,
                    "model_changelog": model_log,
                    "model_architecture": model_cfg.get('name', 'unknown')
                })
                
            except Exception as e:
                logger.warning(f"Не удалось автоматически записать метаданные версий в MLflow: {e}")

            yield run

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Сохраняет словарь метрик."""
        for k, v in metrics.items():
            # Санируем ключ: заменяем двоеточия, знаки равенства и запятые на '_'
            safe_key = (
                k.replace(":", "_")
                 .replace("=", "_")
                 .replace(",", "_")
            )
            mlflow.log_metric(safe_key, v, step=step)

    def log_params(self, params: Dict[str, Any]):
        """Сохраняет гиперпараметры."""
        mlflow.log_params(params)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Загружает локальный файл (веса, препроцессор) в хранилище трекера."""
        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            logger.debug(f"Артефакт {local_path} сохранен в {artifact_path}")
        except Exception as e:
            logger.warning(f"Не удалось отправить артефакт в трекер: {e}")

    def log_dict(self, dictionary: dict, file_name: str, artifact_path: str = None):
        """Сохраняет словарь (например, схему) напрямую как JSON в трекер."""
        full_path = f"{artifact_path}/{file_name}" if artifact_path else file_name
        try:
            mlflow.log_dict(dictionary, full_path)
        except Exception as e:
            logger.warning(f"Не удалось сохранить словарь {file_name} в трекер: {e}")

    def get_optuna_callback(self, metric_name: str, experiment_name: str = "Tuning"):
        """Фабрика коллбэков для Optuna с жесткой привязкой к текущему эксперименту."""
        try:
            return [
                MLflowCallback(
                    tracking_uri=self.tracking_uri, 
                    metric_name=metric_name,
                    mlflow_kwargs={"experiment_name": experiment_name} # <=== Привязывает триалы к твоему эксперименту!
                )
            ]
        except ImportError:
            logger.warning("Пакет optuna_integration не установлен. Логирование Trials отключено.")
            return []
        
    def log_figure(self, figure, file_name: str, artifact_path: str = None):
        """Сохраняет объект matplotlib/seaborn figure напрямую в MLflow."""
        try:
            # mlflow.log_figure умеет принимать объект matplotlib.figure.Figure
            mlflow.log_figure(figure, f"{artifact_path}/{file_name}" if artifact_path else file_name)
            logger.debug(f"График {file_name} успешно залогирован в MLflow.")
        except Exception as e:
            logger.warning(f"Не удалось отправить график {file_name} в трекер: {e}")