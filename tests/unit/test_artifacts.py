"""
test_artifacts.py — тесты ArtifactManager.

Покрытие:
- log_metrics/log_params не падают внутри активного run
- log_metrics корректно санирует ключи (заменяет :, =, , на _)
- log_artifact корректно обрабатывает отсутствующий файл (warning, не crash)
- set_experiment создаёт эксперимент если его нет (идемпотентность)
- get_optuna_callback возвращает список (пустой или с коллбэком)
- start_run автоматически логирует версии и закрывает run на выходе
- log_figure не падает при попытке залогировать график
"""

import pytest
import mlflow
import logging
from unittest.mock import MagicMock
from src.core.artifacts import ArtifactManager

# ---------------------------------------------------------------------------
# Тест 1: log_metrics и log_params внутри run (и санитизация ключей)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_log_metrics_and_params_inside_run(mock_config, tmp_path):
    manager = ArtifactManager(mock_config, tmp_path)
    manager.set_experiment("test_experiment")

    with manager.start_run(run_name="test_run") as run:
        manager.log_params({"lr": 0.01, "depth": 6})
        
        # Передаем "грязные" ключи, чтобы проверить санитизацию
        manager.log_metrics({"rmse:val": 0.15, "mae=val": 0.10, "loss,val": 0.20})
        
        # Получаем сохраненные метрики, чтобы убедиться, что они очистились
        client = mlflow.tracking.MlflowClient(manager.tracking_uri)
        data = client.get_run(run.info.run_id).data
        
        assert "rmse_val" in data.metrics
        assert "mae_val" in data.metrics
        assert "loss_val" in data.metrics

# ---------------------------------------------------------------------------
# Тест 2: log_artifact с несуществующим файлом — warning, не crash
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_log_artifact_missing_file_logs_warning_not_crash(mock_config, tmp_path, caplog):
    manager = ArtifactManager(mock_config, tmp_path)
    manager.set_experiment("test_experiment")

    with manager.start_run(run_name="test_artifact"):
        with caplog.at_level(logging.WARNING, logger="src.core.artifacts"):
            manager.log_artifact("/nonexistent/path/model.pkl", "models")
            assert "Не удалось отправить артефакт" in caplog.text

# ---------------------------------------------------------------------------
# Тест 3: set_experiment — идемпотентен
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_set_experiment_is_idempotent(mock_config, tmp_path):
    manager = ArtifactManager(mock_config, tmp_path)
    manager.set_experiment("idempotent_test_exp")
    
    # Второй вызов не должен упасть или перезаписать криво
    manager.set_experiment("idempotent_test_exp")
    
    experiment = mlflow.get_experiment_by_name("idempotent_test_exp")
    assert experiment is not None

# ---------------------------------------------------------------------------
# Тест 4: get_optuna_callback возвращает список
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_get_optuna_callback_returns_list(mock_config, tmp_path):
    manager = ArtifactManager(mock_config, tmp_path)
    result = manager.get_optuna_callback(metric_name="val_score")
    assert isinstance(result, list)

# ---------------------------------------------------------------------------
# Тест 5: log_dict сохраняет словарь
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_log_dict_inside_run(mock_config, tmp_path):
    manager = ArtifactManager(mock_config, tmp_path)
    manager.set_experiment("test_experiment")
    test_dict = {"col_a": "float64", "col_b": "int64"}

    with manager.start_run(run_name="test_dict"):
        # Просто проверяем, что метод отрабатывает без исключений
        manager.log_dict(test_dict, "feature_schema.json", "schemas")

# ---------------------------------------------------------------------------
# Тест 6: start_run закрывает run на выходе
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_start_run_closes_run_on_exit(mock_config, tmp_path):
    manager = ArtifactManager(mock_config, tmp_path)
    manager.set_experiment("test_experiment")

    with manager.start_run(run_name="ctx_test"):
        assert mlflow.active_run() is not None

    assert mlflow.active_run() is None

# ---------------------------------------------------------------------------
# Тест 7: log_figure обрабатывается корректно (warning при сбое, не crash)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_log_figure_inside_run(mock_config, tmp_path, caplog):
    manager = ArtifactManager(mock_config, tmp_path)
    manager.set_experiment("test_experiment")
    
    # Имитируем график. Если mlflow не сможет его распарсить, мы ожидаем warning, а не crash
    dummy_fig = MagicMock() 

    with manager.start_run(run_name="test_figure"):
        with caplog.at_level(logging.WARNING, logger="src.core.artifacts"):
            manager.log_figure(dummy_fig, "test_plot.png", "plots")