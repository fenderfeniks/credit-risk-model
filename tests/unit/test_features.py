"""
test_features.py — юнит-тесты для TabularPreprocessor и FeatureEngineer.

Покрытие:
- TabularPreprocessor:
  - sklearn-контракт: transform до fit бросает NotFittedError
  - Количество строк не изменяется
  - NaN полностью устраняются (imputation)
  - drop_cols из конфига действительно удаляются
  - Константные колонки удаляются автоматически
  - Статистики считаются только по train (защита от Data Leakage)
  - Выбросы обрезаются корректно (zscore)
  - Идемпотентность (fit_transform == fit + transform)
- FeatureEngineer:
  - Не падает, если на вход пришел DF без нужных колонок (безопасный _safe_concat)
  - Успешно генерирует новые фичи, если исходные данные присутствуют
"""

import pytest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.exceptions import NotFittedError

from src.core.features import TabularPreprocessor, FeatureEngineer

@pytest.fixture
def base_tabular_cfg(mock_config):
    """Гарантируем наличие необходимых списков для TabularPreprocessor."""
    OmegaConf.update(mock_config, "data.tabular.exact_excludes", [])
    OmegaConf.update(mock_config, "data.tabular.prefix_excludes", [])
    OmegaConf.update(mock_config, "data.tabular.suffix_excludes", [])
    return mock_config

@pytest.fixture
def X_train_clean(sample_data, base_tabular_cfg):
    target = base_tabular_cfg.data.tabular.target_col
    return sample_data.drop(columns=[target]).copy()

@pytest.fixture
def X_test_clean(sample_data, base_tabular_cfg):
    target = base_tabular_cfg.data.tabular.target_col
    return sample_data.drop(columns=[target]).copy()

# ===========================================================================
# ТЕСТЫ: TabularPreprocessor
# ===========================================================================

@pytest.mark.unit
def test_transform_before_fit_raises_error(base_tabular_cfg, X_train_clean):
    """Трансформация до обучения должна валиться из-за отсутствия обученных параметров."""
    preprocessor = TabularPreprocessor(base_tabular_cfg)
    with pytest.raises(NotFittedError):
        preprocessor.transform(X_train_clean)

@pytest.mark.unit
def test_row_count_preserved_after_transform(base_tabular_cfg, X_train_clean):
    """Препроцессор технически преобразует признаки, не выкидывая строки."""
    preprocessor = TabularPreprocessor(base_tabular_cfg)
    preprocessor.fit(X_train_clean)
    X_out = preprocessor.transform(X_train_clean)
    assert X_out.shape[0] == X_train_clean.shape[0]

@pytest.mark.unit
def test_no_nan_after_transform(base_tabular_cfg, X_train_clean):
    """Все пропуски должны закрываться заглушками fill_values_ или бизнес-правилами."""
    preprocessor = TabularPreprocessor(base_tabular_cfg)
    preprocessor.fit(X_train_clean)
    X_out = preprocessor.transform(X_train_clean)
    assert X_out.isnull().sum().sum() == 0, f"Остались незаполненные NaN: {X_out.isnull().sum().to_dict()}"

@pytest.mark.unit
def test_explicit_drop_cols_are_removed(base_tabular_cfg, X_train_clean):
    """Колонки, переданные в черном списке drop_cols, должны отсутствовать на выходе."""
    preprocessor = TabularPreprocessor(base_tabular_cfg)
    preprocessor.fit(X_train_clean)
    X_out = preprocessor.transform(X_train_clean)
    assert "explicit_drop" not in X_out.columns

@pytest.mark.unit
def test_constant_column_is_auto_dropped(base_tabular_cfg, X_train_clean):
    """Колонки с избыточным процентом констант (max_constant_pct) вычищаются."""
    preprocessor = TabularPreprocessor(base_tabular_cfg)
    preprocessor.fit(X_train_clean)
    X_out = preprocessor.transform(X_train_clean)
    assert "constant_col" not in X_out.columns

@pytest.mark.unit
def test_fit_statistics_use_only_train_data(base_tabular_cfg, X_train_clean, X_test_clean):
    """fill_values_ должны рассчитываться строго по train-выборке (защита от Data Leakage)."""
    OmegaConf.update(base_tabular_cfg, "data.tabular.max_constant_pct", 1.0)
    OmegaConf.update(base_tabular_cfg, "data.tabular.max_missing_pct", 1.0)
    # Принудительно отключаем guard по кардинальности, чтобы 5.0 не стало категорией
    OmegaConf.update(base_tabular_cfg, "data.tabular.max_cardinality_for_categorical", 0)
    
    X_train_mod = X_train_clean.copy()
    X_test_mod = X_test_clean.copy()

    X_train_mod["total_hits"] = 5.0
    X_test_mod["total_hits"] = 999.0
    X_train_mod.loc[0, "total_hits"] = np.nan

    preprocessor = TabularPreprocessor(base_tabular_cfg)
    preprocessor.fit(X_train_mod)

    fill_val = preprocessor.fill_values_.get("total_hits", None)
    
    assert fill_val is not None, "total_hits не попал в словарь импутации fill_values_!"
    assert fill_val == 5.0, f"Утечка данных! Импутация рассчитана не по train: {fill_val}"

@pytest.mark.unit
def test_outlier_clipping_with_zscore(base_tabular_cfg, X_train_clean):
    """При активации zscore экстремальные выбросы жестко обрезаются по рассчитанным границам."""
    OmegaConf.update(base_tabular_cfg, "data.tabular.outlier_method", "zscore")
    OmegaConf.update(base_tabular_cfg, "data.tabular.outlier_threshold", 2.0)
    OmegaConf.update(base_tabular_cfg, "data.tabular.max_cardinality_for_categorical", 0)

    X_with_outliers = X_train_clean.copy()
    X_with_outliers.loc[0, "total_hits"] = 999999.0

    preprocessor = TabularPreprocessor(base_tabular_cfg)
    preprocessor.fit(X_with_outliers)
    X_out = preprocessor.transform(X_with_outliers)

    lower, upper = preprocessor.outlier_bounds_["total_hits"]
    assert X_out["total_hits"].max() <= upper + 1e-5

@pytest.mark.unit
def test_fit_transform_equals_fit_then_transform(base_tabular_cfg, X_train_clean):
    """Результаты сквозного fit_transform и раздельного вызова методов обязаны совпадать."""
    prep_1 = TabularPreprocessor(base_tabular_cfg)
    out_1 = prep_1.fit_transform(X_train_clean)

    prep_2 = TabularPreprocessor(base_tabular_cfg)
    prep_2.fit(X_train_clean)
    out_2 = prep_2.transform(X_train_clean)

    pd.testing.assert_frame_equal(out_1, out_2)

# ===========================================================================
# ТЕСТЫ: FeatureEngineer
# ===========================================================================

@pytest.mark.unit
def test_feature_engineer_safe_execution_on_missing_cols(X_train_clean):
    """FeatureEngineer не должен падать, если в датафрейме нет специфичных банковских колонок."""
    engineer = FeatureEngineer(config={})
    engineer.fit(X_train_clean)
    
    try:
        X_out = engineer.transform(X_train_clean)
        # Убеждаемся, что исходные колонки на месте
        assert "device_brand" in X_out.columns
    except Exception as e:
        pytest.fail(f"FeatureEngineer упал на датасете без целевых колонок: {e}")

@pytest.mark.unit
def test_feature_engineer_generates_features():
    """Если необходимые колонки есть, FeatureEngineer должен успешно собрать новые признаки."""
    df = pd.DataFrame({
        "share_any_overdue": [0.0, 0.5, 0.1],
        "share_serious_overdue": [0.0, 0.1, 0.0],
        "pre_util_last": [1, 2, np.nan],
        "paym_row_share_code_0_mean": [0.99, 0.50, 0.90]
    })
    
    engineer = FeatureEngineer(config={})
    engineer.fit(df)
    X_out = engineer.transform(df)
    
    # 1. Проверяем флаг абсолютно хорошего плательщика
    assert "is_absolutely_perfect_payer" in X_out.columns
    assert list(X_out["is_absolutely_perfect_payer"]) == [1, 0, 0]
    
    # 2. Проверяем флаг хронического должника
    assert "is_chronic_defaulter" in X_out.columns
    assert list(X_out["is_chronic_defaulter"]) == [0, 1, 0]
    
    # 3. Проверяем генерацию категориальных склеек
    assert "cat_util_and_history" in X_out.columns
    # np.nan в pre_util_last должно заполниться -999 по логике инженера
    assert "-999_Mild" in list(X_out["cat_util_and_history"])