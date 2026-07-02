"""
test_data.py — юнит-тесты для UniversalDataLoader.

Покрытие:
- Сохранение полного объёма строк при сплите
- Корректность пропорций (с допуском ±1 на округление sklearn)
- Отсутствие data leakage между выборками
- Корректная стратификация по таргету
- Поведение при граничных значениях (test_size=0, val_size=0)
- Сброс индексов после сплита
"""

import pytest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from src.core.data import UniversalDataLoader

def _make_loader(cfg, tmp_path, test_size: float, val_size: float) -> UniversalDataLoader:
    """Создаёт загрузчик с заданными размерами сплитов."""
    cfg.data.test_size = test_size
    cfg.data.val_size = val_size
    return UniversalDataLoader(cfg, tmp_path, source_type="parquet")

# ---------------------------------------------------------------------------
# Тест 1: Сохранение полного объёма данных
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_splits_preserve_total_row_count(mock_config, sample_data, tmp_path):
    """Ни одна строка не должна потеряться или задублироваться при сплите."""
    loader = _make_loader(mock_config, tmp_path, test_size=0.2, val_size=0.2)
    train, val, test = loader.get_splits(sample_data)

    total = len(train) + len(val) + len(test)
    assert total == len(sample_data), (
        f"Потеря данных при сплите: исходных строк {len(sample_data)}, после сплита {total}."
    )

# ---------------------------------------------------------------------------
# Тест 2: Пропорции сплитов (с допуском ±1 на округление)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("test_size,val_size", [
    (0.2, 0.2),
    (0.1, 0.1),
    (0.15, 0.15),
])
def test_splits_proportions_are_correct(mock_config, sample_data, tmp_path, test_size, val_size):
    """Размеры test и val должны соответствовать заданным пропорциям с допуском ±1 строка."""
    loader = _make_loader(mock_config, tmp_path, test_size=test_size, val_size=val_size)
    train, val, test = loader.get_splits(sample_data)

    n = len(sample_data)
    expected_test = int(n * test_size)
    expected_val = int(n * val_size)

    assert abs(len(test) - expected_test) <= 1, f"Ожидалось ~{expected_test} строк в test, получено {len(test)}."
    assert abs(len(val) - expected_val) <= 1, f"Ожидалось ~{expected_val} строк в val, получено {len(val)}."

# ---------------------------------------------------------------------------
# Тест 3: Отсутствие data leakage — через пересечение множеств ID
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_data_leakage_between_splits(mock_config, sample_data, tmp_path):
    """Каждая строка должна попасть строго в одну выборку (строгий контроль утечек)."""
    loader = _make_loader(mock_config, tmp_path, test_size=0.2, val_size=0.2)
    train, val, test = loader.get_splits(sample_data)

    id_col = "session_id"
    train_ids = set(train[id_col])
    val_ids = set(val[id_col])
    test_ids = set(test[id_col])

    assert train_ids.isdisjoint(val_ids), f"Leakage Train <-> Val: {len(train_ids & val_ids)} общих строк."
    assert train_ids.isdisjoint(test_ids), f"Leakage Train <-> Test: {len(train_ids & test_ids)} общих строк."
    assert val_ids.isdisjoint(test_ids), f"Leakage Val <-> Test: {len(val_ids & test_ids)} общих строк."

    all_ids = train_ids | val_ids | test_ids
    assert all_ids == set(sample_data[id_col]), "Часть строк потерялась при сплите!"

# ---------------------------------------------------------------------------
# Тест 4: Стратификация по таргету для классификации
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_stratification_preserves_class_distribution(mock_config, sample_data, tmp_path):
    """Доля положительного класса в сплитах должна быть репрезентативна (отклонение < 10%)."""
    OmegaConf.update(mock_config, "task_type", "binary")
    loader = _make_loader(mock_config, tmp_path, test_size=0.2, val_size=0.2)
    train, val, test = loader.get_splits(sample_data)

    target = mock_config.data.tabular.target_col
    original_ratio = sample_data[target].mean()
    test_ratio = test[target].mean()

    assert abs(original_ratio - test_ratio) < 0.10, (
        f"Стратификация нарушена: исходная доля = {original_ratio:.2f}, в test = {test_ratio:.2f}."
    )

# ---------------------------------------------------------------------------
# Тест 5: Граничный случай — val_size=0
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_val_size_zero_returns_empty_val(mock_config, sample_data, tmp_path):
    """При val_size=0 валидационная выборка должна быть пустым DataFrame с колонками."""
    loader = _make_loader(mock_config, tmp_path, test_size=0.2, val_size=0.0)
    train, val, test = loader.get_splits(sample_data)

    assert len(val) == 0, f"Ожидался пустой val, получено {len(val)} строк."
    assert list(val.columns) == list(sample_data.columns), "Колонки пустого val должны совпадать с исходными."
    assert len(train) + len(test) == len(sample_data)

# ---------------------------------------------------------------------------
# Тест 6: Граничный случай — test_size=0
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_test_size_zero_returns_empty_test(mock_config, sample_data, tmp_path):
    """При test_size=0 тестовая выборка должна быть пустым DataFrame."""
    loader = _make_loader(mock_config, tmp_path, test_size=0.0, val_size=0.2)
    train, val, test = loader.get_splits(sample_data)

    assert len(test) == 0, f"Ожидался пустой test, получено {len(test)} строк."
    assert len(train) + len(val) == len(sample_data)

# ---------------------------------------------------------------------------
# Тест 7: Индексы сброшены после сплита
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_splits_have_reset_index(mock_config, sample_data, tmp_path):
    """Индексы выборок должны быть непрерывными и начинаться с 0 для стабильного .iloc."""
    loader = _make_loader(mock_config, tmp_path, test_size=0.2, val_size=0.2)
    train, val, test = loader.get_splits(sample_data)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        if len(df) == 0: continue
        assert list(df.index) == list(range(len(df))), f"Индекс {name} не сброшен в 0 непрерывную последовательность."

# ---------------------------------------------------------------------------
# Тест 8: Некорректные пропорции выбрасывают ValueError
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_invalid_split_sizes_raise_error(mock_config, sample_data, tmp_path):
    """Если сумма долей >= 1.0,get_splits должен падать с ValueError."""
    loader = _make_loader(mock_config, tmp_path, test_size=0.6, val_size=0.5)
    with pytest.raises(ValueError, match="Сумма test_size и val_size"):
        loader.get_splits(sample_data)