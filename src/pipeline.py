"""
pipeline.py

Пайплайн кредитного скоринга — от сырых данных до предикта.

Использование:

    from pipeline import CreditScoringPipeline

    # Создание и обучение
    pipeline = CreditScoringPipeline(model_params=best_params)
    pipeline.fit(df_raw, y)

    # Предикт
    result = pipeline.predict(df_raw_new)

    # Сохранение (один файл содержит всё)
    import dill
    with open('credit_pipeline.pkl', 'wb') as f:
        dill.dump(pipeline, f)

    # Загрузка
    with open('credit_pipeline.pkl', 'rb') as f:
        loaded = dill.load(f)

    result = loaded.predict(df_raw_new)
"""

import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from aggregator import CreditHistoryAggregatorFast
from feature_engineering import feature_engineering


class CreditScoringPipeline:
    """
    Автоматизированный пайплайн кредитного скоринга.

    Инкапсулирует все шаги обработки данных:
    агрегация → feature engineering → масштабирование → модель.

    Поддерживает два режима:
    - fit() — обучение всех компонентов на сырых данных
    - predict() — применение обученных компонентов к новым данным

    Parameters
    ----------
    model_params : dict
        Гиперпараметры LGBMClassifier.
    rare_threshold : float
        Порог для определения редких бинов в агрегаторе. По умолчанию 0.05.
    """

    def __init__(self, model_params: dict, rare_threshold: float = 0.05):
        self.model_params   = model_params
        self.rare_threshold = rare_threshold

        # Компоненты — заполняются в fit()
        self.aggregator_   = None  # CreditHistoryAggregatorFast
        self.scaler_       = None  # StandardScaler
        self.model_        = None  # LGBMClassifier
        self.feature_cols_ = None  # порядок колонок при обучении

    def fit(self, raw_df: pd.DataFrame, y: pd.Series) -> 'CreditScoringPipeline':
        """
        Обучает все компоненты пайплайна на сырых данных.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Сырые данные — строки кредитов с колонкой 'id'.
        y : pd.Series
            Целевая переменная (1 — дефолт, 0 — нет).

        Returns
        -------
        self
        """
        # 1. Агрегация — обучаем агрегатор на полных данных
        self.aggregator_ = CreditHistoryAggregatorFast(
            rare_threshold=self.rare_threshold
        )
        self.aggregator_.update_global_stats(raw_df)
        self.aggregator_.finalize_global_stats()

        agg_df = self.aggregator_.transform(raw_df)
        X      = agg_df.drop(columns=['id'])

        # 2. Производные признаки
        X = feature_engineering(X)

        # 3. Фиксируем порядок колонок — важно для корректного инференса
        self.feature_cols_ = X.columns.tolist()

        # 4. Масштабирование — fit только на трейне
        self.scaler_ = StandardScaler()
        X_scaled     = pd.DataFrame(
            self.scaler_.fit_transform(X),
            columns=self.feature_cols_
        )

        # 5. Обучение модели
        self.model_ = lgb.LGBMClassifier(**self.model_params)
        self.model_.fit(X_scaled, y)

        print(f"Pipeline fitted. Признаков: {len(self.feature_cols_)}")
        return self

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Предсказывает вероятность дефолта для новых данных.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Сырые данные — строки кредитов с колонкой 'id'.

        Returns
        -------
        pd.DataFrame
            Датафрейм с колонками 'id' и 'flag' (вероятность дефолта 0..1).
        """
        if self.aggregator_ is None:
            raise RuntimeError("Сначала вызови fit().")

        # 1. Агрегация — только transform, не fit
        agg_df = self.aggregator_.transform(raw_df)
        ids    = agg_df['id']
        X      = agg_df.drop(columns=['id'])

        # 2. Производные признаки
        X = feature_engineering(X)

        # 3. Гарантируем тот же порядок колонок, что и при обучении
        X = X[self.feature_cols_]

        # 4. Масштабирование — только transform, не fit
        X_scaled = pd.DataFrame(
            self.scaler_.transform(X),
            columns=self.feature_cols_
        )

        # 5. Предикт — возвращаем вероятности класса 1
        proba = self.model_.predict_proba(X_scaled)[:, 1]

        return pd.DataFrame({'id': ids.values, 'flag': proba})
