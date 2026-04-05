"""
feature_engineering.py

Создание производных признаков из агрегированного датафрейма.

Использование:
    from feature_engineering import feature_engineering
    X = feature_engineering(X_agg)
"""

import numpy as np
import pandas as pd


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт производные признаки из агрегированного датафрейма.

    Все признаки создаются как соотношения или разности уже существующих
    агрегатов — без обращения к сырым данным.

    Parameters
    ----------
    X : pd.DataFrame
        Агрегированный датафрейм (выход CreditHistoryAggregatorFast.transform).
        Не должен содержать колонку 'id'.

    Returns
    -------
    pd.DataFrame
        Датафрейм с добавленными производными признаками.
    """
    X = X.copy()

    # --- Блок 1: Качество просрочек ---

    # Отношение тяжёлых просрочек (90+ дней) к лёгким (до 5 дней).
    # Клиент у которого есть только тяжёлые просрочки значительно
    # рискованнее клиента с лёгкими это сигнал эскалации проблем.
    X['serious_to_mild_ratio'] = (
        X['loans90_overdue_share'] /
        X['loans5_overdue_share'].replace(0, np.nan)
    ).fillna(0)

    # Доля серьёзных просрочек среди всех.
    # Качество просрочек важнее их количества:
    # 1 тяжёлая просрочка хуже, чем 10 лёгких.
    X['serious_overdue_fraction'] = (
        X['count_serious_overdue'] /
        X['count_any_overdue'].replace(0, np.nan)
    ).fillna(0)

    # --- Блок 2: Динамика финансовой нагрузки ---

    # Рост утилизации лимита от первого к последнему кредиту.
    # Растущая утилизация сигнализирует о нарастании финансовой нагрузки.
    X['util_first_vs_last'] = (
        X['pre_util_last'] - X['pre_util_first']
    )

    # Тренд остатка долга.
    # Растущий долг при неизменном лимите признак ухудшения положения.
    X['outstanding_trend'] = (
        X['pre_loans_outstanding_last'] - X['pre_loans_outstanding_first']
    )

    # Изменение кредитного лимита.
    # Снижение лимита банком говорит о пересмотре риска клиента.
    X['credit_limit_trend'] = (
        X['pre_loans_credit_limit_last'] - X['pre_loans_credit_limit_first']
    )

    return X
