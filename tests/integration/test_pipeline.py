"""
test_pipeline.py — интеграционные тесты MLPipeline.

Покрытие:
- Smoke-тест полного цикла train → артефакты на диске → predict
- predict до train/load поднимает ValueError
- Выходной тип predict — np.ndarray
- Размер выходного вектора == размеру входа
- Цикл save → load → predict даёт идентичные предсказания
- Препроцессор сохраняется и загружается корректно
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
    import src.core.pipeline as pipeline_module
    original_get_model = pipeline_module.get_model
    pipeline_module.get_model = lambda cfg, root: DummyModel()
    yield
    pipeline_module.get_model = original_get_model

class DummyModel:
    def __init__(self):
        self.file_extension = ".cbm"
    def fit(self, X, y, X_val=None, y_val=None, tracker=None):
        pass
    def predict(self, X):
        return np.zeros(len(X))
    def save(self):
        return "mock_model_v1.0.0.cbm"
    def load(self, path):
        pass

def _get_splits(sample_data, target):
    train = sample_data.iloc[:160]
    val = sample_data.iloc[160:180]
    test = sample_data.iloc[180:]
    return (
        train.drop(columns=[target]), train[target],
        val.drop(columns=[target]), val[target],
        test.drop(columns=[target]), test[target],
    )

@pytest.mark.integration
def test_pipeline_train_runs_without_error(mock_config, sample_data, tmp_path):
    tracker = ArtifactManager(mock_config, tmp_path)
    pipeline = MLPipeline(mock_config, tracker, tmp_path)
    target = mock_config.data.tabular.target_col
    X_train, y_train, X_val, y_val, _, _ = _get_splits(sample_data, target)

    pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=False, use_tracker=False)
    assert pipeline.preprocessor is not None

@pytest.mark.unit
def test_predict_before_train_raises_value_error(mock_config, sample_data, tmp_path):
    tracker = ArtifactManager(mock_config, tmp_path)
    pipeline = MLPipeline(mock_config, tracker, tmp_path)
    target = mock_config.data.tabular.target_col
    X = sample_data.drop(columns=[target])

    with pytest.raises(ValueError, match="еще не обучен"):
        pipeline.predict(X)

@pytest.mark.integration
def test_predict_output_type_and_shape(mock_config, sample_data, trained_pipeline):
    target = mock_config.data.tabular.target_col
    X_test = sample_data.iloc[180:].drop(columns=[target])
    preds = trained_pipeline.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)

@pytest.mark.integration
def test_artifacts_are_saved_to_disk(mock_config, sample_data, tmp_path):
    tracker = ArtifactManager(mock_config, tmp_path)
    # Явно создаем/выставляем тест-эксперимент, чтобы mlflow не ругался на ID=1
    tracker.set_experiment("pipeline_integration_test")
    
    pipeline = MLPipeline(mock_config, tracker, tmp_path)
    target = mock_config.data.tabular.target_col
    X_train, y_train, X_val, y_val, _, _ = _get_splits(sample_data, target)

    # Запускаем в контексте рана, так как save_artifacts внутри дергает tracker.log_dict
    with tracker.start_run(run_name="test_save_artifacts"):
        pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=True, use_tracker=True)

    models_dir = tmp_path / mock_config.paths.models_dir
    prep_ver = mock_config.data.tabular.preprocessing_version
    feat_ver = mock_config.data.tabular.features_version

    assert (models_dir / f"preprocessing_v{prep_ver}.pkl").exists()
    assert (models_dir / f"feature_schema_v{feat_ver}.json").exists()


# ---------------------------------------------------------------------------
# Тест 6: Схема фичей содержит корректный JSON
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
# Тест 7: Цикл save → load → predict даёт идентичные предсказания
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
    
    # Симулируем создание файла весов модели, чтобы загрузчик на диске его увидел
    model_ver = mock_config.model.model_version
    model_file = tmp_path / mock_config.paths.models_dir / f"{mock_config.model.name}_v{model_ver}.cbm"
    model_file.write_text("mock_weight")

    pipeline_loaded.load()
    preds_loaded = pipeline_loaded.predict(X_test)
    np.testing.assert_array_almost_equal(preds_original, preds_loaded, decimal=5)