"""
test_model_behavior.py — тесты поведения модели (behavioral testing).

Покрытие:
- Robustness: модель не падает и не возвращает NaN при экстремальных числовых значениях
- Robustness: модель не падает при NaN на инференсе
- Robustness: модель не падает при полностью пустой строке (все NaN)
- Predict_proba: вероятности лежат в математическом диапазоне [0, 1]
- Predict_proba: вероятности бинарных классов суммируются в 1
- Batch-инвариантность: predict(batch) == [predict(row) for row in batch]
"""

import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Тест 1: Экстремальные числовые значения не вызывают NaN или исключение
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_extreme_numeric_values_do_not_cause_nan(mock_config, sample_data, trained_pipeline):
    """Пайплайн должен выдерживать экстремальные выбросы (сработает clip), не отдавая NaN."""
    target = mock_config.data.tabular.target_col
    user = sample_data.iloc[[0]].drop(columns=[target]).copy()
    
    # Имитируем жесткий выброс
    user["total_hits"] = 999_999_999.0

    pred = trained_pipeline.predict(user)
    assert len(pred) == 1
    assert not np.isnan(float(pred[0]))

# ---------------------------------------------------------------------------
# Тест 2: NaN на инференсе — модель не падает
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_nan_input_at_inference_does_not_crash(mock_config, sample_data, trained_pipeline):
    """Случайные пропуски в ключевых полях на проде должны закрываться заглушками."""
    target = mock_config.data.tabular.target_col
    user = sample_data.iloc[[0]].drop(columns=[target]).copy()
    
    # Ломаем пару колонок
    user["device_brand"] = np.nan
    user["screen_area"] = np.nan

    pred = trained_pipeline.predict(user)
    assert len(pred) == 1
    assert not np.isnan(float(pred[0]))

# ---------------------------------------------------------------------------
# Тест 3: Полностью пустая строка (все NaN) — не падает
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_all_nan_row_does_not_crash(mock_config, sample_data, trained_pipeline):
    """Предельный кейс: если на вход пришла полностью пустая сырая строка, пайплайн обязан отдать скор."""
    target = mock_config.data.tabular.target_col
    user = sample_data.iloc[[0]].drop(columns=[target]).copy()

    # Затираем всё в NaN
    for col in user.columns:
        user.loc[:, col] = np.nan

    try:
        pred = trained_pipeline.predict(user)
        assert len(pred) == 1
        assert not np.isnan(float(pred[0]))
    except Exception as e:
        pytest.fail(f"Пайплайн упал при полностью пустом вводе: {e}")

# ---------------------------------------------------------------------------
# Тест 4: predict_proba суммируется в 1
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_predict_proba_sums_to_one(mock_config, sample_data, trained_pipeline):
    """Сумма вероятностей бинарных классов для каждой строки должна быть строго равна 1.0."""
    target = mock_config.data.tabular.target_col
    X_test = sample_data.iloc[180:].drop(columns=[target])

    # Вытаскиваем очищенные данные и прогоняем через внутреннюю модель
    X_clean = trained_pipeline.preprocessor.transform(X_test)
    proba = trained_pipeline.model.predict_proba(X_clean)

    proba_sums = proba.sum(axis=1)
    np.testing.assert_allclose(proba_sums, np.ones(len(proba_sums)), atol=1e-5)

# ---------------------------------------------------------------------------
# Тест 5: predict_proba в диапазоне [0, 1]
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_predict_proba_in_valid_range(mock_config, sample_data, trained_pipeline):
    """Вероятности классов не могут вылетать за математический базис [0, 1]."""
    target = mock_config.data.tabular.target_col
    X_test = sample_data.iloc[180:].drop(columns=[target])

    X_clean = trained_pipeline.preprocessor.transform(X_test)
    proba = trained_pipeline.model.predict_proba(X_clean)

    assert proba.min() >= 0.0
    assert proba.max() <= 1.0

# ---------------------------------------------------------------------------
# Тест 6: Batch-инвариантность predict
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_batch_prediction_equals_individual_predictions(mock_config, sample_data, trained_pipeline):
    """Результат прогноза для пачки строк должен до бита сходиться с итерируемыми поштучно запросами.
    Это гарантирует, что FeatureEngineer и Preprocessor не делают кросс-строковых вычислений на инференсе."""
    target = mock_config.data.tabular.target_col
    X_batch = sample_data.iloc[180:185].drop(columns=[target])

    # Предикт пачкой
    batch_preds = trained_pipeline.predict(X_batch)
    
    # Предикт по одной строке
    individual_preds = np.array([trained_pipeline.predict(X_batch.iloc[[i]])[0] for i in range(len(X_batch))])

    np.testing.assert_array_almost_equal(
        batch_preds, individual_preds, decimal=5,
        err_msg="Батчевые предсказания расходятся с поэлементными. Найдена зависимость от размера пачки!"
    )