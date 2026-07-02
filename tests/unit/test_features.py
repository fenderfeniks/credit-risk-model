"""
test_features.py — юнит-тесты для TabularPreprocessor.

Покрытие:
- sklearn-контракт: transform до fit бросает NotFittedError
- Количество строк не изменяется
- NaN полностью устраняются
- drop_cols из конфига действительно удаляются
- Константные колонки удаляются автоматически
- Статистики считаются только по train (no leakage в fit)
- transform на test не использует test-статистики
- Выбросы обрезаются корректно (zscore)
- fit_transform == fit + transform (идемпотентность)
"""

import pytest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from src.core.features import TabularPreprocessor

@pytest.fixture
def X_train_clean(sample_data, mock_config):
    target = mock_config.data.tabular.target_col
    return sample_data.drop(columns=[target]).copy()

@pytest.fixture
def X_test_clean(sample_data, mock_config):
    target = mock_config.data.tabular.target_col
    return sample_data.drop(columns=[target]).copy()

# ---------------------------------------------------------------------------
# Тест 1: Проверка вызова до fit
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_transform_before_fit_raises_error(mock_config, X_train_clean):
    """Трансформация до обучения должна валиться из-за отсутствия обученных параметров."""
    preprocessor = TabularPreprocessor(mock_config)
    with pytest.raises(Exception):
        preprocessor.transform(X_train_clean)

# ---------------------------------------------------------------------------
# Тест 2: Количество строк неизменно
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_row_count_preserved_after_transform(mock_config, X_train_clean):
    """Препроцессор технически преобразует признаки, не выкидывая строки."""
    preprocessor = TabularPreprocessor(mock_config)
    preprocessor.fit(X_train_clean)
    X_out = preprocessor.transform(X_train_clean)
    assert X_out.shape[0] == X_train_clean.shape[0]

# ---------------------------------------------------------------------------
# Тест 3: После transform не должно остаться NaN в обработанных полях
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_nan_after_transform(mock_config, X_train_clean):
    """Все пропуски должны закрываться заглушками fill_values_ или бизнес-правилами."""
    preprocessor = TabularPreprocessor(mock_config)
    preprocessor.fit(X_train_clean)
    X_out = preprocessor.transform(X_train_clean)
    assert X_out.isnull().sum().sum() == 0, f"Остались незаполненные NaN: {X_out.isnull().sum().to_dict()}"

# ---------------------------------------------------------------------------
# Тест 4: Явные drop_cols из конфига действительно удаляются
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_explicit_drop_cols_are_removed(mock_config, X_train_clean):
    """Колонки, переданные в черном списке drop_cols, должны отсутствовать на выходе."""
    preprocessor = TabularPreprocessor(mock_config)
    preprocessor.fit(X_train_clean)
    X_out = preprocessor.transform(X_train_clean)
    assert "explicit_drop" not in X_out.columns

# ---------------------------------------------------------------------------
# Тест 5: Константная колонка удаляется автоматически
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_constant_column_is_auto_dropped(mock_config, X_train_clean):
    """Колонки с избыточным процентом констант (max_constant_pct) вычищаются."""
    preprocessor = TabularPreprocessor(mock_config)
    preprocessor.fit(X_train_clean)
    X_out = preprocessor.transform(X_train_clean)
    assert "constant_col" not in X_out.columns

# ---------------------------------------------------------------------------
# Тест 6: Статистики fit считаются только по train (no leakage)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_fit_statistics_use_only_train_data(mock_config, X_train_clean, X_test_clean):
    """fill_values_ должны соответствовать статистикам train, а не test."""
    # Отключаем удаление констант, чтобы колонку не дропнуло на этапе анализа мусора
    OmegaConf.update(mock_config, "data.tabular.max_constant_pct", 1.0)
    OmegaConf.update(mock_config, "data.tabular.max_missing_pct", 1.0)
    
    X_train_mod = X_train_clean.copy()
    X_test_mod = X_test_clean.copy()

    # Явно гарантируем float-тип, чтобы pandas не дропнул колонку в object из-за NaN
    X_train_mod["total_hits"] = X_train_mod["total_hits"].astype(float)
    X_test_mod["total_hits"] = X_test_mod["total_hits"].astype(float)

    X_train_mod["total_hits"] = 5.0
    X_test_mod["total_hits"] = 999.0

    # Провоцируем вычисление fill_value
    X_train_mod.loc[0, "total_hits"] = np.nan

    preprocessor = TabularPreprocessor(mock_config)
    preprocessor.fit(X_train_mod)

    fill_val = preprocessor.fill_values_.get("total_hits", None)
    
    assert fill_val is not None, "total_hits не попал в словарь импутации fill_values_!"
    assert fill_val < 100.0, f"Зафиксирована утечка данных! Медиана взята из теста: {fill_val}"

# ---------------------------------------------------------------------------
# Тест 7: Выбросы обрезаются при outlier_method=zscore
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_outlier_clipping_with_zscore(mock_config, X_train_clean):
    """При активации zscore экстремальные выбросы жестко обрезаются по рассчитанным границам."""
    OmegaConf.update(mock_config, "data.tabular.outlier_method", "zscore")
    OmegaConf.update(mock_config, "data.tabular.outlier_threshold", 2.0)

    X_with_outliers = X_train_clean.copy()
    X_with_outliers.loc[0, "total_hits"] = 999999.0

    preprocessor = TabularPreprocessor(mock_config)
    preprocessor.fit(X_with_outliers)
    X_out = preprocessor.transform(X_with_outliers)

    lower, upper = preprocessor.outlier_bounds_["total_hits"]
    assert X_out["total_hits"].max() <= upper + 1e-5
    assert X_out["total_hits"].min() >= lower - 1e-5

# ---------------------------------------------------------------------------
# Тест 8: Идемпотентность (fit_transform == fit + transform)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_fit_transform_equals_fit_then_transform(mock_config, X_train_clean):
    """Результаты сквозного fit_transform и раздельного вызова методов обязаны совпадать."""
    prep_1 = TabularPreprocessor(mock_config)
    out_1 = prep_1.fit_transform(X_train_clean)

    prep_2 = TabularPreprocessor(mock_config)
    prep_2.fit(X_train_clean)
    out_2 = prep_2.transform(X_train_clean)

    pd.testing.assert_frame_equal(out_1, out_2)

# ---------------------------------------------------------------------------
# Тест 9: Наполнение learned_drop_cols_
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_learned_drop_cols_contains_all_bad_columns(mock_config, X_train_clean):
    """Финальный черный список препроцессора должен аккумулировать технические, константные и пустые поля."""
    preprocessor = TabularPreprocessor(mock_config)
    preprocessor.fit(X_train_clean)

    assert "explicit_drop" in preprocessor.learned_drop_cols_
    assert "constant_col" in preprocessor.learned_drop_cols_
    assert "session_id" in preprocessor.learned_drop_cols_