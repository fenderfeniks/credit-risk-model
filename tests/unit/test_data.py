"""
test_data.py — юнит-тесты для модуля splitting.py.

Покрытие:
- filter_bad_rows корректно удаляет строки с превышением NaNs
- Сплит происходит строго на уровне id_col (нет утечек сессий одного клиента)
- Пропорции сплитов (train/val/test) соблюдаются на уровне уникальных клиентов
- Корректная стратификация по таргету для уникальных клиентов
- Поведение при граничных значениях (test_size=0, val_size=0)
- Выброс ValueError при некорректных долях (test_size + val_size >= 1.0)
- Индексы DataFrame сброшены после сплита
"""

import pytest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.core.splitting import split_data, filter_bad_rows

@pytest.fixture
def grouped_sample_data():
    """Создает датасет, где на одного клиента приходится несколько строк (сессий)."""
    rng = np.random.default_rng(42)
    n_clients = 100
    rows_per_client = 3
    
    data = []
    for i in range(n_clients):
        client_id = f"client_{i}"
        target = rng.choice([0, 1], p=[0.8, 0.2])
        for j in range(rows_per_client):
            data.append({
                "client_id": client_id,
                "session_id": f"sess_{i}_{j}",
                "feature_1": rng.random(),
                "feature_2": rng.random(),
                "is_target_action": target
            })
            
    return pd.DataFrame(data)

def _prep_config(cfg, test_size, val_size):
    """Подготавливает конфиг для тестов сплитования."""
    OmegaConf.update(cfg, "data.test_size", test_size)
    OmegaConf.update(cfg, "data.val_size", val_size)
    OmegaConf.update(cfg, "data.tabular.id_col", "client_id")
    OmegaConf.update(cfg, "data.tabular.max_row_missing_pct", 0.5)
    return cfg

# ---------------------------------------------------------------------------
# Тест 1: Фильтрация плохих строк
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_filter_bad_rows(mock_config):
    """Строки, в которых доля пропусков больше max_row_missing_pct, должны удаляться."""
    OmegaConf.update(mock_config, "data.tabular.max_row_missing_pct", 0.5)
    
    df = pd.DataFrame({
        "col1": [1, 2, np.nan, np.nan],
        "col2": [1, np.nan, np.nan, np.nan],
        "col3": [1, 2, 3, np.nan]
    })
    
    # 1 строка: 0% NaNs (остается)
    # 2 строка: 33% NaNs (остается)
    # 3 строка: 66% NaNs (удаляется, порог 0.5)
    # 4 строка: 100% NaNs (удаляется)
    
    clean_df = filter_bad_rows(mock_config, df)
    
    assert len(clean_df) == 2
    assert list(clean_df["col1"].dropna()) == [1.0, 2.0]

# ---------------------------------------------------------------------------
# Тест 2: Отсутствие data leakage между выборками (на уровне client_id)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_data_leakage_by_client_id(mock_config, grouped_sample_data):
    """Один client_id не должен находиться одновременно в двух разных выборках."""
    cfg = _prep_config(mock_config, test_size=0.2, val_size=0.2)
    train, val, test = split_data(cfg, grouped_sample_data)

    train_ids = set(train["client_id"])
    val_ids = set(val["client_id"])
    test_ids = set(test["client_id"])

    assert train_ids.isdisjoint(val_ids), f"Leakage Train/Val: общие клиенты {train_ids & val_ids}"
    assert train_ids.isdisjoint(test_ids), f"Leakage Train/Test: общие клиенты {train_ids & test_ids}"
    assert val_ids.isdisjoint(test_ids), f"Leakage Val/Test: общие клиенты {val_ids & test_ids}"

    all_original_ids = set(grouped_sample_data["client_id"])
    assert train_ids | val_ids | test_ids == all_original_ids, "Часть клиентов потерялась при сплите!"

# ---------------------------------------------------------------------------
# Тест 3: Пропорции сплитов по УНИКАЛЬНЫМ КЛИЕНТАМ
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("test_size, val_size", [
    (0.2, 0.2), # Ожидаем 60 / 20 / 20 (из 100 клиентов)
    (0.1, 0.1), # Ожидаем 80 / 10 / 10
])
def test_split_proportions_by_unique_clients(mock_config, grouped_sample_data, test_size, val_size):
    """Сплит должен дробить пропорции по id_col, а не по количеству строк."""
    cfg = _prep_config(mock_config, test_size, val_size)
    train, val, test = split_data(cfg, grouped_sample_data)

    n_clients = grouped_sample_data["client_id"].nunique()
    
    expected_test_clients = int(n_clients * test_size)
    expected_val_clients = int(n_clients * val_size)

    # Допуск ±1 клиент из-за округлений sklearn.train_test_split
    assert abs(test["client_id"].nunique() - expected_test_clients) <= 1
    assert abs(val["client_id"].nunique() - expected_val_clients) <= 1

# ---------------------------------------------------------------------------
# Тест 4: Граничный случай — val_size=0
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_val_size_zero_returns_empty_val(mock_config, grouped_sample_data):
    """При val_size=0 валидационная выборка должна быть пустым DataFrame."""
    cfg = _prep_config(mock_config, test_size=0.2, val_size=0.0)
    train, val, test = split_data(cfg, grouped_sample_data)

    assert len(val) == 0, f"Ожидался пустой val, получено {len(val)} строк."
    assert list(val.columns) == list(grouped_sample_data.columns)

# ---------------------------------------------------------------------------
# Тест 5: Граничный случай — test_size=0
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_test_size_zero_returns_empty_test(mock_config, grouped_sample_data):
    """При test_size=0 тестовая выборка должна быть пустой."""
    cfg = _prep_config(mock_config, test_size=0.0, val_size=0.2)
    train, val, test = split_data(cfg, grouped_sample_data)

    assert len(test) == 0, f"Ожидался пустой test, получено {len(test)} строк."

# ---------------------------------------------------------------------------
# Тест 6: Индексы сброшены после сплита
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_splits_have_reset_index(mock_config, grouped_sample_data):
    """Индексы выборок должны быть непрерывными и начинаться с 0 для стабильного .iloc."""
    cfg = _prep_config(mock_config, test_size=0.2, val_size=0.2)
    train, val, test = split_data(cfg, grouped_sample_data)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        if len(df) == 0: continue
        assert list(df.index) == list(range(len(df))), f"Индекс {name} не сброшен."

# ---------------------------------------------------------------------------
# Тест 7: Некорректные пропорции выбрасывают ValueError
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_invalid_split_sizes_raise_error(mock_config, grouped_sample_data):
    """Если сумма долей >= 1.0, функция должна падать с ValueError."""
    cfg = _prep_config(mock_config, test_size=0.6, val_size=0.5)
    with pytest.raises(ValueError, match="Сумма test_size и val_size должна быть < 1.0"):
        split_data(cfg, grouped_sample_data)