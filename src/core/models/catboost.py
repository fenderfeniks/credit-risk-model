from __future__ import annotations

import logging
from typing import Optional
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .base import BaseModelWrapper

# Импортируем специфичные библиотеки внутри методов или через try/except,
# чтобы код не падал, если в проекте не установлен CatBoost или PyTorch
try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ============================================================
# 2. РЕАЛИЗАЦИЯ ДЛЯ КЛАССИКИ (Пример: CatBoost)
# ============================================================

class CatBoostWrapper(BaseModelWrapper):

    def __init__(self, config: DictConfig, project_root):
        super().__init__(config, project_root)
        self.ml_cfg = self.cfg.training.ml

        # Интегрируем Loss и Metrics из конфига прямо в параметры CatBoost
        full_params = OmegaConf.to_container(self.model_cfg.params, resolve=True)
        full_params['loss_function'] = self.cfg.loss_function

        # CatBoost принимает метрики в формате списка строк (custom_metric)
        if self.cfg.metrics:
            full_params['custom_metric'] = list(self.cfg.metrics)
        # Полностью запрещаем CatBoost создавать папку catboost_info
        full_params['allow_writing_files'] = False

        # Инициализируем модель опираясь на ГЛОБАЛЬНУЮ задачу
        if self.task_type == 'regression':
            self.model = CatBoostRegressor(**full_params)
        elif self.task_type in ['binary', 'multiclass']:
            self.model = CatBoostClassifier(**full_params)
        else:
            raise ValueError(f"Неизвестный task_type для CatBoost: {self.task_type}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
            tracker=None) -> None:  # <--- Добавлен tracker

        cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        # 1. ЛОГИРОВАНИЕ В TRACKER (Без прямого MLflow)
        if tracker:
            params_to_log = self.model.get_params()
            params_to_log.update({
                "model_name": self.model_cfg.name,
                "model_version": self.model_cfg.model_version,
                "cat_features_count": len(cat_features)
            })
            tracker.log_params(params_to_log)

        # 2. ПОДГОТОВКА ДАННЫХ
        train_pool = Pool(X_train, y_train, cat_features=cat_features)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(X_val, y_val, cat_features=cat_features)

        # 3. ОБУЧЕНИЕ С ПАРАМЕТРАМИ ИЗ КОНФИГА
        logger.info(f"Обучение {self.model_cfg.name} (v{self.model_cfg.model_version})...")

        self.model.fit(
            train_pool,
            eval_set=eval_set,
            early_stopping_rounds=self.ml_cfg.early_stopping_rounds,
            verbose=self.ml_cfg.verbose
        )

        # 4. ЛОГИРОВАНИЕ ЛУЧШЕЙ МЕТРИКИ ЧЕРЕЗ TRACKER
        best_score = self.model.get_best_score()
        if eval_set and 'validation' in best_score and tracker:
            metrics_to_log = {f"best_val_{k}": v for k, v in best_score['validation'].items()}
            tracker.log_metrics(metrics_to_log)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(self.model, CatBoostClassifier):
            return self.model.predict_proba(X)
        raise NotImplementedError("predict_proba доступен только для классификации.")

    def save(self) -> str:
        """Нативное сохранение в формат CatBoost (.cbm)"""
        
        save_path = self.get_artifact_path(self.models_dir, self.model_version)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Нативное сохранение (CatBoost ожидает строку, а не объект Path)
        self.model.save_model(str(save_path))
        logger.info(f"Модель CatBoost нативно сохранена в {save_path}")

        return str(save_path)

    def load(self, load_path: str) -> None:
        """Нативная загрузка весов в существующий объект CatBoost."""
        from pathlib import Path

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Файл модели CatBoost не найден по пути: {load_path}")

        # Метод load_model загружает веса прямо в пустую архитектуру,
        # которую мы заранее инициализировали в __init__
        self.model.load_model(str(load_path))
        logger.info(f"Модель CatBoost успешно загружена из {load_path}")

    @property
    def file_extension(self) -> str:
        return ".cbm"

    def get_best_val_score(self, metric_name: str) -> float:
        scores = self.model.get_best_score()
        if 'validation' not in scores:
            return 0.0
        val_scores = scores['validation']
        # Ищем без учёта регистра
        for key, value in val_scores.items():
            if key.lower() == metric_name.lower():
                return value
        # Fallback: первая доступная метрика
        logger.warning(f"Метрика '{metric_name}' не найдена в {list(val_scores.keys())}. Берём первую.")
        return next(iter(val_scores.values()), 0.0)
    
    def get_feature_importance(self, X: pd.DataFrame = None) -> pd.DataFrame:
        """
        Возвращает DataFrame с колонками ['Feature', 'Importance'], 
        отсортированный по убыванию важности признаков.
        """      
        # Получаем значения важности из нативной модели CatBoost
        importances = self.model.get_feature_importance()
        
        # Так как CatBoost возвращает просто массив чисел, 
        # нам нужны имена колонок, чтобы сопоставить их
        feature_names = X.columns if X is not None else [f"feature_{i}" for i in range(len(importances))]
        
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        return fi_df