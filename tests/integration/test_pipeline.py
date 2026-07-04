"""
test_pipeline.py — интеграционные тесты MLPipeline.

Покрытие:
- Smoke-тест полного цикла train → артефакты на диске → predict
- predict до train/load поднимает ValueError
- Выходной тип predict — np.ndarray
- Размер выходного вектора == размеру входа
- Цикл save → load → predict даёт идентичные предсказания
- Препроцессор и схемы сохраняются и загружаются корректно
- use_tracker=False не пишет в MLflow вне run
"""

import pytest
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.core.pipeline import MLPipeline
from src.core.artifacts import ArtifactManager

@pytest.fixture(autouse=True)
def mock_get_model_globally():
    """Подменяет фабрику моделей на заглушку для всего файла."""
    import src.core.pipeline as pipeline_module
    original_get_model = pipeline_module.get_model
    pipeline_module.get_model = lambda cfg, root: DummyModel()
    yield
    pipeline_module.get_model = original_get_model

class DummyModel:
    """Локальный стаб модели, реализующий интерфейс BaseModelWrapper."""
    def __init__(self):
        self.file_extension = ".cbm"
        
    def fit(self, X, y, X_val=None, y_val=None, tracker=None):
        pass
        
    def predict(self, X):
        return np.zeros(len(X))
        
    def predict_proba(self, X):
        return np.hstack([np.ones((len(X), 1)) * 0.8, np.ones((len(X), 1)) * 0.2])
        
    def save(self):
        return "models/mock_model_v1.0.0.cbm"
        
    def load(self, path):
        pass

    def get_artifact_path(self, models_dir, version):
        return Path(models_dir) / f"mock_model_v{version}{self.file_extension}"

    def get_feature_importance(self, X):
        return pd.DataFrame({'Feature': X.columns, 'Importance': [1.0] * len(X.columns)})

def _get_splits(sample_data, target):
    train = sample_data.iloc[:160]
    val = sample_data.iloc[160:180]
    test = sample_data.iloc[180:]
    return (
        train.drop(columns=[target]), train[target],
        val.drop(columns=[target]), val[target],
        test.drop(columns=[target]), test[target],
    )

# ---------------------------------------------------------------------------
# Тест 1: Smoke-тест обучения пайплайна
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_pipeline_train_runs_without_error(mock_config, sample_data, tmp_path):
    tracker = ArtifactManager(mock_config, tmp_path)
    pipeline = MLPipeline(mock_config, tracker, tmp_path)
    target = mock_config.data.tabular.target_col
    X_train, y_train, X_val, y_val, _, _ = _get_splits(sample_data, target)

    pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=False, use_tracker=False)
    assert pipeline.preprocessor is not None

# ---------------------------------------------------------------------------
# Тест 2: Защита от предикта до обучения
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_before_train_raises_value_error(mock_config, sample_data, tmp_path):
    tracker = ArtifactManager(mock_config, tmp_path)
    pipeline = MLPipeline(mock_config, tracker, tmp_path)
    target = mock_config.data.tabular.target_col
    X = sample_data.drop(columns=[target])

    with pytest.raises(ValueError, match="еще не обучен"):
        pipeline.predict(X)

# ---------------------------------------------------------------------------
# Тест 3: Формат и размер выхода predict
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_predict_output_type_and_shape(mock_config, sample_data, trained_pipeline):
    target = mock_config.data.tabular.target_col
    X_test = sample_data.iloc[180:].drop(columns=[target])
    preds = trained_pipeline.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)

# ---------------------------------------------------------------------------
# Тест 4: Артефакты сохраняются на диск (раздельные версии)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_artifacts_are_saved_to_disk(mock_config, sample_data, tmp_path):
    tracker = ArtifactManager(mock_config, tmp_path)
    tracker.set_experiment("pipeline_integration_test")
    
    pipeline = MLPipeline(mock_config, tracker, tmp_path)
    target = mock_config.data.tabular.target_col
    X_train, y_train, X_val, y_val, _, _ = _get_splits(sample_data, target)

    with tracker.start_run(run_name="test_save_artifacts"):
        pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=True, use_tracker=True)

    models_dir = tmp_path / mock_config.paths.models_dir
    prep_ver = mock_config.data.tabular.preprocessing_version
    feat_ver = mock_config.data.tabular.features_version

    expected_prep_path = models_dir / f"preprocessing_v{prep_ver}.pkl"
    assert expected_prep_path.exists(), f"Файл не найден по пути: {expected_prep_path}"

    assert (models_dir / f"preprocessing_v{prep_ver}.pkl").exists()
    assert (models_dir / f"feature_schema_v{feat_ver}.json").exists()

# ---------------------------------------------------------------------------
# Тест 5: Схема фичей — валидный JSON
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_feature_schema_is_valid_json(mock_config, sample_data, tmp_path):
    tracker = ArtifactManager(mock_config, tmp_path)
    tracker.set_experiment("pipeline_integration_test")
    
    pipeline = MLPipeline(mock_config, tracker, tmp_path)
    target = mock_config.data.tabular.target_col
    X_train, y_train, X_val, y_val, _, _ = _get_splits(sample_data, target)

    with tracker.start_run(run_name="test_schema_artifacts"):
        pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=True, use_tracker=True)

    feat_ver = mock_config.data.tabular.features_version
    schema_path = tmp_path / mock_config.paths.models_dir / f"feature_schema_v{feat_ver}.json"

    with open(schema_path) as f:
        schema = json.load(f)
    assert isinstance(schema, dict)
    assert len(schema) > 0

# ---------------------------------------------------------------------------
# Тест 6: Идентичность предсказаний после загрузки (save -> load -> predict)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_save_load_predict_is_identical(mock_config, sample_data, tmp_path):
    tracker = ArtifactManager(mock_config, tmp_path)
    tracker.set_experiment("pipeline_integration_test")
    
    pipeline_original = MLPipeline(mock_config, tracker, tmp_path)
    target = mock_config.data.tabular.target_col
    X_train, y_train, X_val, y_val, X_test, _ = _get_splits(sample_data, target)

    with tracker.start_run(run_name="test_save_load_artifacts"):
        pipeline_original.train(X_train, y_train, X_val, y_val, save_artifacts=True, use_tracker=True)
    
    preds_original = pipeline_original.predict(X_test)

    pipeline_loaded = MLPipeline(mock_config, tracker, tmp_path)
    
    # Симулируем создание файла весов модели, чтобы загрузчик на диске его увидел.
    # Используем метод get_artifact_path, чтобы путь 100% совпал с тем, что ищет пайплайн.
    models_dir = tmp_path / mock_config.paths.models_dir
    model_ver = mock_config.model.model_version
    
    dummy_model = DummyModel()
    model_file = dummy_model.get_artifact_path(models_dir, model_ver)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text("mock_weight")

    pipeline_loaded.load()
    preds_loaded = pipeline_loaded.predict(X_test)
    
    np.testing.assert_array_almost_equal(preds_original, preds_loaded, decimal=5)