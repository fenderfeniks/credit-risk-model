from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .base import BaseModelWrapper

# Импортируем специфичные библиотеки внутри методов или через try/except,
# чтобы код не падал, если в проекте не установлен CatBoost или PyTorch
try:
    import xgboost as xgb

    XGBOOST_INSTALLED = True
except ImportError:
    XGBOOST_INSTALLED = False

logger = logging.getLogger(__name__)


# ============================================================
# 4. XGBoost
# ============================================================

class XGBoostWrapper(BaseModelWrapper):
    def __init__(self, config: DictConfig, project_root):
        super().__init__(config, project_root)
        if not XGBOOST_INSTALLED:
            raise ImportError("Библиотека xgboost не установлена!")

        self.ml_cfg = self.cfg.training.ml
        full_params = OmegaConf.to_container(self.model_cfg.params, resolve=True)
        full_params['objective'] = self.cfg.loss_function

        objective_mapping = {
            'Logloss': 'binary:logistic',
            'CrossEntropy': 'binary:logistic',
            'RMSE': 'reg:squarederror',
            'MAE': 'reg:absoluteerror',
            'MultiClass': 'multi:softmax',
            'MultiClassOneVsAll': 'multi:softprob'
        }
        raw_objective = self.cfg.loss_function
        # Если название есть в словаре, берем значение для XGBoost. Если нет - оставляем как есть.
        full_params['objective'] = objective_mapping.get(raw_objective, raw_objective)

        metric_mapping = {
            'auc': 'auc',
            'f1': 'aucpr',            # Встроенного F1 нет, Precision-Recall AUC — лучшая альтернатива
            'accuracy': 'error',      # XGBoost измеряет ошибку (1 - Accuracy)
            'rocauc': 'auc',
            'roc_auc': 'auc',
            'logloss': 'logloss',
            'rmse': 'rmse',
            'mae': 'mae'
        }

        if self.cfg.metrics:
            # Переводим ListConfig от Hydra/OmegaConf в обычный список
            metrics_list = list(self.cfg.metrics)
            mapped_metrics = []
            
            for m in metrics_list:
                m_lower = str(m).lower()
                # Переводим метрику или оставляем её в нижнем регистре
                mapped_metrics.append(metric_mapping.get(m_lower, m_lower))
                
            # Передаем строку, если метрика одна, или список, если несколько
            full_params['eval_metric'] = mapped_metrics[0] if len(mapped_metrics) == 1 else mapped_metrics

        if self.ml_cfg.early_stopping_rounds > 0:
            full_params['early_stopping_rounds'] = self.ml_cfg.early_stopping_rounds

        # Обязательно для нативной работы с category dtype (без ручного One-Hot/Ordinal)
        full_params['enable_categorical'] = True

        if self.task_type == 'regression':
            self.model = xgb.XGBRegressor(**full_params)
        elif self.task_type in ['binary', 'multiclass']:
            self.model = xgb.XGBClassifier(**full_params)
        else:
            raise ValueError(f"Неизвестный task_type для XGBoost: {self.task_type}")

        self.cat_columns_ = None
        self.cat_categories_ = {}

    def _prepare_categorical(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = X.copy()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if fit:
            self.cat_columns_ = cat_cols
            self.cat_categories_ = {
                col: sorted(X[col].astype(str).unique().tolist()) for col in cat_cols
            }

        if self.cat_columns_ is None:
            raise RuntimeError("Модель ещё не обучена — категориальные колонки неизвестны.")

        for col in self.cat_columns_:
            if col in X.columns:
                X[col] = pd.Categorical(X[col].astype(str), categories=self.cat_categories_[col])

        return X

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
            tracker=None) -> None:

        X_train = self._prepare_categorical(X_train, fit=True)
        X_val_prepared = self._prepare_categorical(X_val) if X_val is not None else None

        if tracker:
            params_to_log = self.model.get_params()
            params_to_log.update({
                "model_name": self.model_cfg.name,
                "model_version": self.model_cfg.model_version,
                "cat_features_count": len(self.cat_columns_),
            })
            tracker.log_params(params_to_log)

        eval_set = [(X_val_prepared, y_val)] if X_val_prepared is not None and y_val is not None else None
        verbose_val = self.ml_cfg.verbose if self.ml_cfg.verbose > 0 else False

        logger.info(f"Обучение XGBoost ({self.model_cfg.name})...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose_val
        )

        if eval_set and hasattr(self.model, 'best_score') and tracker:
            metric_name = self.cfg.metrics[0] if self.cfg.metrics else "score"
            tracker.log_metrics({f"best_val_{metric_name}": self.model.best_score})

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = self._prepare_categorical(X)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            X = self._prepare_categorical(X)
            return self.model.predict_proba(X)
        raise NotImplementedError("predict_proba доступен только для задач классификации.")

    def save(self) -> str:
        """Нативное сохранение XGBoost в универсальный формат UBJSON.

        ВАЖНО: нативный формат XGBoost хранит только саму модель. Метаданные
        cat_columns_/cat_categories_, нужные _prepare_categorical() при инференсе,
        сохраняем отдельным .meta.json файлом рядом с моделью — иначе после load()
        в новой сессии predict() будет падать с RuntimeError
        ("категориальные колонки неизвестны"), т.к. wrapper не хранит их нигде,
        кроме памяти текущего процесса.
        """
        save_path = self.get_artifact_path(self.models_dir, self.model_version)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(save_path))

        meta_path = save_path.with_suffix(save_path.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "cat_columns_": self.cat_columns_,
                "cat_categories_": self.cat_categories_,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Модель XGBoost нативно сохранена в {save_path}")
        logger.info(f"Метаданные категориальных колонок сохранены в {meta_path}")

        return str(save_path)

    def load(self, load_path: str) -> None:
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Файл модели XGBoost не найден: {load_path}")

        self.model.load_model(str(load_path))

        # Метаданные подхватываются автоматически по соглашению об именовании
        # (<model_path>.meta.json), явно вызывать ничего дополнительно не нужно.
        meta_path = load_path.with_suffix(load_path.suffix + ".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.cat_columns_ = meta.get("cat_columns_")
            self.cat_categories_ = meta.get("cat_categories_", {})
        else:
            logger.warning(
                f"Файл метаданных {meta_path} не найден. "
                "predict() упадет на _prepare_categorical() — модель нужно переобучить "
                "и пересохранить через новую версию save()."
            )

        logger.info(f"Модель XGBoost успешно загружена из {load_path}")

    @property
    def file_extension(self) -> str:
        return ".ubj"

    def get_best_val_score(self, metric_name: str) -> float:
        if not hasattr(self.model, 'best_score'):
            return 0.0

        configured_metrics = list(self.cfg.metrics) if self.cfg.metrics else []
        primary_metric = configured_metrics[0] if configured_metrics else None

        if primary_metric and primary_metric.lower() != metric_name.lower():
            logger.warning(
                f"XGBoost хранит только метрику eval_metric[0]='{primary_metric}', "
                f"но запрошена '{metric_name}'. Возвращаю best_score для '{primary_metric}', "
                f"это может быть НЕ та метрика, которую оптимизирует Optuna!"
            )

        return self.model.best_score
    
    def get_feature_importance(self, X: pd.DataFrame = None) -> pd.DataFrame:
        """Возвращает DataFrame важности признаков для XGBoost."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            # Фолбек на случай нативного бустера
            try:
                booster = self.model.get_booster()
                score = booster.get_score(importance_type='weight')
                # get_score возвращает словарь, преобразуем его под колонки X
                importances = [score.get(col, 0) for col in X.columns] if X is not None else list(score.values())
            except Exception:
                logger.warning("Не удалось извлечь важность признаков из XGBoost.")
                importances = np.zeros(len(X.columns) if X is not None else 0)

        feature_names = X.columns if X is not None else [f"feature_{i}" for i in range(len(importances))]

        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        return fi_df