"""
conftest.py — базовые фикстуры для всех тестов.

Ключевые принципы:
- mock_config имеет scope="function": каждый тест получает свежую копию,
  мутации одного теста не влияют на другие.
- sample_ga_data генерируется через np.random.Generator (не global seed),
  что делает генерацию детерминированной и изолированной.
- pipeline_factory — фабрика, возвращающая обученный пайплайн;
  используется в нескольких тестах без дублирования логики обучения.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import mlflow

import hydra
from hydra import compose, initialize

from src.core.artifacts import ArtifactManager
from src.core.pipeline import MLPipeline

TARGET_COL = "is_target_action"

@pytest.fixture(scope="function")
def mock_config(tmp_path):
    """Подгружает реальный конфиг и принудительно патчит его недостающими полями для тестов."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
        
        # 1. Принудительно задаем пути в tmp_path, чтобы тесты не писали в реальные папки
        OmegaConf.update(cfg, "paths.data_dir", str(tmp_path / "data"), force_add=True)
        OmegaConf.update(cfg, "paths.models_dir", str(tmp_path / "models"), force_add=True)
        OmegaConf.update(cfg, "paths.reports_dir", str(tmp_path / "reports"), force_add=True)
        OmegaConf.update(cfg, "paths.logs_dir", str(tmp_path / "logs"), force_add=True)
        
        # 2. Патчим недостающие ключи, на которые ругаются тесты
        if "params" not in cfg.model:
            OmegaConf.update(cfg, "model.params", {"depth": 6}, force_add=True)
        if "ml" not in cfg.training:
            OmegaConf.update(cfg, "training.ml", {"early_stopping_rounds": 10, "verbose": 0}, force_add=True)
        if "mlflow" not in cfg.logging:
            OmegaConf.update(cfg, "logging.mlflow", {"experiments": {"train": "t"}}, force_add=True)
            
        return cfg
    

@pytest.fixture(scope="function")
def sample_data():
    """Генерирует синтетический датасет, содержащий обязательные технические колонки проекта."""
    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame({
        "session_id": [f"sess_{i}" for i in range(n)],
        "client_id": [f"client_{i}" for i in range(n)],
        "visit_date": ["2026-06-01"] * n,
        "visit_time": ["12:00:00"] * n,
        "device_brand": rng.choice(["apple", "samsung", "huawei"], size=n),
        "device_category": rng.choice(["mobile", "desktop", "tablet"], size=n),
        "screen_area": rng.integers(300000, 2000000, size=n).astype(float),
        "total_hits": rng.integers(1, 50, size=n).astype(float),
        "explicit_drop": ["trash"] * n,
        "constant_col": [42.0] * n,
        "flag": rng.choice([0, 1], size=n),
        TARGET_COL: rng.choice([0, 1], size=n, p=[0.8, 0.2]),
    })

    # Добавляем немного NaN
    nan_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[nan_idx, "device_brand"] = np.nan

    return df

class DummyModel:
    """Легковесный стаб модели, имитирующий BaseModelWrapper для тестов инференса и пайплайна."""
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
        """Новый метод, необходимый для MLPipeline.load()"""
        return Path(models_dir) / f"mock_model_v{version}{self.file_extension}"

    def get_feature_importance(self, X):
        """Новый метод, необходимый для MLPipeline.train() при отрисовке отчетов"""
        return pd.DataFrame({'Feature': X.columns, 'Importance': [1.0] * len(X.columns)})

@pytest.fixture(scope="function")
def trained_pipeline(mock_config, sample_data, tmp_path):
    """Возвращает предобученный пайплайн с DummyModel."""
    tracker = ArtifactManager(mock_config, tmp_path)
    pipeline = MLPipeline(mock_config, tracker, tmp_path)
    
    target = mock_config.data.tabular.target_col
    X_train = sample_data.drop(columns=[target])
    y_train = sample_data[target]

    # Подменяем get_model внутренним моком, чтобы не дергать реальный CatBoost
    import src.core.pipeline as pipeline_module
    original_get_model = pipeline_module.get_model
    pipeline_module.get_model = lambda cfg, root: DummyModel()

    pipeline.train(X_train, y_train, save_artifacts=False, use_tracker=False)
    
    yield pipeline
    pipeline_module.get_model = original_get_model

@pytest.fixture(scope="function", autouse=True)
def cleanup_mlflow():
    """Гарантирует закрытие MLflow ранов после каждого теста."""
    yield
    while mlflow.active_run():
        mlflow.end_run()