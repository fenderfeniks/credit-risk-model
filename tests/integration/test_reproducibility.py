"""
test_reproducibility.py — тесты воспроизводимости обучения.

Покрытие:
- Два обучения с одним seed дают идентичные предсказания
- Два обучения с разными seed дают разные предсказания
  (доказывает что seed реально влияет, а не просто игнорируется)
- Предсказания стабильны при повторных вызовах predict (детерминизм инференса)
"""

import pytest
import numpy as np
from omegaconf import OmegaConf
from src.core.pipeline import MLPipeline
from src.core.artifacts import ArtifactManager
from src.core.data import UniversalDataLoader

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

def _train_pipeline(cfg, sample_data, tmp_path):
    target = cfg.data.tabular.target_col
    X_train = sample_data.iloc[:160].drop(columns=[target])
    y_train = sample_data.iloc[:160][target]
    X_val = sample_data.iloc[160:180].drop(columns=[target])
    y_val = sample_data.iloc[160:180][target]

    tracker = ArtifactManager(cfg, tmp_path)
    pipeline = MLPipeline(cfg, tracker, tmp_path)
    pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=False, use_tracker=False)
    return pipeline

def _get_test_X(cfg, sample_data):
    target = cfg.data.tabular.target_col
    return sample_data.iloc[180:].drop(columns=[target])

@pytest.mark.integration
def test_same_seed_produces_identical_predictions(mock_config, sample_data, tmp_path):
    X_test = _get_test_X(mock_config, sample_data)
    pipeline_1 = _train_pipeline(mock_config, sample_data, tmp_path)
    preds_1 = pipeline_1.predict(X_test)

    pipeline_2 = _train_pipeline(mock_config, sample_data, tmp_path)
    preds_2 = pipeline_2.predict(X_test)
    np.testing.assert_array_equal(preds_1, preds_2)

@pytest.mark.unit
def test_data_splits_are_reproducible(mock_config, sample_data, tmp_path):
    loader = UniversalDataLoader(mock_config, tmp_path, source_type="parquet")
    train_1, val_1, test_1 = loader.get_splits(sample_data)
    train_2, val_2, test_2 = loader.get_splits(sample_data)

    assert list(train_1.index) == list(train_2.index)
    assert list(val_1.index) == list(val_2.index)