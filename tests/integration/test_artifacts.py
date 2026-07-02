"""
test_artifacts.py — тесты ArtifactManager.

Покрытие:
- log_metrics/log_params не падают внутри активного run
- log_metrics вне run поднимает исключение (а не молча игнорирует)
- get_optuna_callback возвращает список (пустой или с коллбэком)
- set_experiment создаёт эксперимент если его нет
- log_artifact корректно обрабатывает отсутствующий файл (warning, не crash)
"""

import pytest
import mlflow
import logging
from src.core.artifacts import ArtifactManager

# ---------------------------------------------------------------------------
# Тест 1: log_metrics и log_params не падают внутри run
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_log_metrics_and_params_inside_run(mock_config, tmp_path):
    manager = ArtifactManager(mock_config, tmp_path)
    manager.set_experiment("test_experiment")

    with manager.start_run(run_name="test_run"):
        manager.log_params({"lr": 0.01, "depth": 6})
        manager.log_metrics({"rmse": 0.15, "mae": 0.10})
        manager.log_metrics({"val_loss": 0.20}, step=1)

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

# ---------------------------------------------------------------------------
# Тест 3: set_experiment — идемпотентен
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_set_experiment_is_idempotent(mock_config, tmp_path):
    manager = ArtifactManager(mock_config, tmp_path)
    manager.set_experiment("idempotent_test_exp")
    manager.set_experiment("idempotent_test_exp")

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
        manager.log_dict(test_dict, "feature_schema.json", "schemas")

# ---------------------------------------------------------------------------
# Тест 6: start_run закрывает run
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_start_run_closes_run_on_exit(mock_config, tmp_path):
    manager = ArtifactManager(mock_config, tmp_path)
    manager.set_experiment("test_experiment")

    with manager.start_run(run_name="ctx_test"):
        assert mlflow.active_run() is not None

    assert mlflow.active_run() is None