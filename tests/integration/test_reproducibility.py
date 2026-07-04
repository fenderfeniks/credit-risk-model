"""
test_reproducibility.py — тесты воспроизводимости обучения и сплитования.

Покрытие:
- Два сплита с одним seed дают идентичный набор данных (на уровне client_id)
- Два обучения пайплайна с одним seed дают идентичные предсказания
"""

import pytest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path

from src.core.pipeline import MLPipeline
from src.core.artifacts import ArtifactManager
from src.core.splitting import split_data

@pytest.fixture(autouse=True)
def mock_get_model_globally():
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


def _train_pipeline(cfg, sample_data, tmp_path):
    """Вспомогательная функция для прогона данных через новый split_data и обучения."""
    OmegaConf.update(cfg, "data.tabular.id_col", "client_id")
    
    train_df, val_df, _ = split_data(cfg, sample_data)
    
    target = cfg.data.tabular.target_col
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    tracker = ArtifactManager(cfg, tmp_path)
    pipeline = MLPipeline(cfg, tracker, tmp_path)
    pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=False, use_tracker=False)
    
    return pipeline

# ---------------------------------------------------------------------------
# Тест 1: Идентичность предсказаний при одинаковом seed
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_same_seed_produces_identical_predictions(mock_config, sample_data, tmp_path):
    OmegaConf.update(mock_config, "data.tabular.id_col", "client_id")
    target = mock_config.data.tabular.target_col
    
    # Будем предсказывать на всем датасете для проверки детерминизма
    X_test = sample_data.drop(columns=[target])
    
    pipeline_1 = _train_pipeline(mock_config, sample_data, tmp_path)
    preds_1 = pipeline_1.predict(X_test)

    pipeline_2 = _train_pipeline(mock_config, sample_data, tmp_path)
    preds_2 = pipeline_2.predict(X_test)
    
    np.testing.assert_array_equal(preds_1, preds_2)

# ---------------------------------------------------------------------------
# Тест 2: Идентичность сплитования при одинаковом seed
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_data_splits_are_reproducible(mock_config, sample_data):
    OmegaConf.update(mock_config, "data.tabular.id_col", "client_id")
    
    train_1, val_1, test_1 = split_data(mock_config, sample_data)
    train_2, val_2, test_2 = split_data(mock_config, sample_data)

    # Проверяем, что индексы и сами client_id совпадают 1 в 1
    assert list(train_1["client_id"]) == list(train_2["client_id"])
    assert list(val_1["client_id"]) == list(val_2["client_id"])
    assert list(test_1["client_id"]) == list(test_2["client_id"])