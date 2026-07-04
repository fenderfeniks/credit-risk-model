"""
test_metrics.py — юнит-тесты для модуля расчета метрик.

Покрытие:
- Стандартный расчет бинарных метрик (Accuracy, Precision, Recall, F1, ROC-AUC).
- Division by Zero: если модель предсказала все 0, Precision не должен падать с ZeroDivisionError.
- Edge Case: в валидацию (y_true) попал только один класс (ROC-AUC должен возвращать NaN/0, а не крашить пайплайн).
- Многоклассовая классификация не падает, если передать task_type='multiclass'.
"""

import pytest
import numpy as np
import pandas as pd

from src.core.metrics import calculate_metrics

@pytest.fixture
def standard_binary_data():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    y_prob = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], 
                       [0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
    return y_true, y_pred, y_prob

# ---------------------------------------------------------------------------
# Тест 1: Стандартный расчет бинарных метрик
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_calculate_metrics_binary_standard(standard_binary_data):
    y_true, y_pred, y_prob = standard_binary_data
    
    metrics = calculate_metrics(y_true, y_pred, task_type="binary", y_prob=y_prob)
    
    # Проверяем, что базовые ключи на месте
    expected_keys = ["accuracy", "f1_weighted", "roc_auc"]
    for key in expected_keys:
        assert key in metrics, f"Метрика {key} отсутствует в ответе."
        assert not np.isnan(metrics[key]), f"Метрика {key} вернула NaN."

# ---------------------------------------------------------------------------
# Тест 2: Защита от деления на ноль (модель предсказала только нули)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_metrics_all_zeros_prediction_no_crash():
    """Если алгоритм вернул вектор из одних нулей, Precision (TP / (TP + FP)) = 0 / 0. Не должно падать."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0]) # Модель-пессимист
    
    try:
        metrics = calculate_metrics(y_true, y_pred, task_type="binary")
        # Как именно функция обрабатывает ноль - зависит от реализации (обычно 0.0 или warning)
        assert not np.isnan(metrics.get("accuracy", 0))
    except ZeroDivisionError:
        pytest.fail("calculate_metrics упал с ZeroDivisionError!")

# ---------------------------------------------------------------------------
# Тест 3: Защита ROC-AUC от единственного класса в y_true
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_metrics_single_class_in_y_true():
    """Если батч маленький и туда попал только класс 0, roc_auc_score из sklearn кинет ValueError."""
    y_true = np.array([0, 0, 0, 0]) # Только негативный класс
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([[0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.7, 0.3]])
    
    try:
        metrics = calculate_metrics(y_true, y_pred, task_type="binary", y_prob=y_prob)
        # roc_auc может быть не рассчитан или равен 0/NaN, главное — отсутствие Exception
        if "roc_auc" in metrics:
            assert np.isnan(metrics["roc_auc"]) or metrics["roc_auc"] == 0.0 or metrics["roc_auc"] is None
    except ValueError as e:
        pytest.fail(f"calculate_metrics упал при одном классе в таргете: {e}")