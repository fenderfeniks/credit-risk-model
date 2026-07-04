"""
test_latency.py — тесты производительности инференса.

Покрытие:
- Медианная латентность одного запроса укладывается в SLA (для predict и predict_proba)
- P95 латентность (хвост распределения) не превышает 2x SLA
- Латентность батча из N строк масштабируется линейно (не экспоненциально)
- Повторные вызовы не деградируют (нет утечек памяти / state накопления)
"""

import pytest
import time
import numpy as np
import pandas as pd

SINGLE_REQUEST_SLA_SEC = 0.050    # 50 мс — медианный SLA
P95_MULTIPLIER = 2.0              # P95 не должен превышать 2x SLA
WARMUP_RUNS = 5                   # прогрев кэшей
MEASURE_RUNS = 30                 # количество замеров для статистики
BATCH_SIZES = [1, 10, 50]         # размеры батчей для теста масштабируемости

def _measure_latencies(pipeline, X: pd.DataFrame, n_runs: int, use_proba: bool = False) -> np.ndarray:
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        if use_proba:
            # Имитируем логику инференса вероятностей из main.py
            X_clean = pipeline.preprocessor.transform(X)
            pipeline.model.predict_proba(X_clean)
        else:
            pipeline.predict(X)
        times.append(time.perf_counter() - start)
    return np.array(times)

# ---------------------------------------------------------------------------
# Тест 1: Медианная латентность одного запроса <= SLA
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("use_proba", [False, True])
def test_single_request_median_latency_within_sla(mock_config, sample_data, trained_pipeline, use_proba):
    """Медианное время ответа (с расчетом вероятностей и без) должно укладываться в SLA."""
    target = mock_config.data.tabular.target_col
    single_row = sample_data.iloc[[0]].drop(columns=[target])

    for _ in range(WARMUP_RUNS):
        _measure_latencies(trained_pipeline, single_row, 1, use_proba)

    latencies = _measure_latencies(trained_pipeline, single_row, MEASURE_RUNS, use_proba)
    median_latency = np.median(latencies)

    assert median_latency < SINGLE_REQUEST_SLA_SEC, (
        f"Медианная латентность {'(proba)' if use_proba else ''} {median_latency * 1000:.2f} мс "
        f"превысила SLA {SINGLE_REQUEST_SLA_SEC * 1000} мс."
    )

# ---------------------------------------------------------------------------
# Тест 2: P95 латентности не превышает 2x SLA
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
def test_single_request_p95_latency_within_limit(mock_config, sample_data, trained_pipeline):
    """Контроль хвоста распределения времени ответов (P95) критичен для пользовательского опыта."""
    target = mock_config.data.tabular.target_col
    single_row = sample_data.iloc[[0]].drop(columns=[target])

    for _ in range(WARMUP_RUNS):
        trained_pipeline.predict(single_row)

    latencies = _measure_latencies(trained_pipeline, single_row, MEASURE_RUNS)
    p95_latency = np.percentile(latencies, 95)
    p95_limit = SINGLE_REQUEST_SLA_SEC * P95_MULTIPLIER

    assert p95_latency < p95_limit, (
        f"P95 латентность {p95_latency * 1000:.2f} мс превысила лимит {p95_limit * 1000} мс."
    )

# ---------------------------------------------------------------------------
# Тест 3: Латентность батча масштабируется линейно (субэкспоненциально)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
def test_batch_latency_scales_subexponentially(mock_config, sample_data, trained_pipeline):
    """Время работы пакетного инференса не должно расти как O(N^2) из-за скрытых циклов."""
    target = mock_config.data.tabular.target_col
    results = {}

    for batch_size in BATCH_SIZES:
        X_batch = sample_data.iloc[:batch_size].drop(columns=[target])
        for _ in range(WARMUP_RUNS):
            trained_pipeline.predict(X_batch)
        latencies = _measure_latencies(trained_pipeline, X_batch, 10)
        results[batch_size] = np.median(latencies)

    single_latency = results[1]
    for batch_size in BATCH_SIZES[1:]:
        batch_latency = results[batch_size]
        linear_limit = single_latency * batch_size
        assert batch_latency < linear_limit, (
            f"Пакет {batch_size} строк отработал за {batch_latency*1000:.2f} мс, "
            f"что медленнее линейного предела {linear_limit*1000:.2f} мс. Обнаружена O(N^2) операция!"
        )

# ---------------------------------------------------------------------------
# Тест 4: Повторные вызовы не деградируют (нет накопления состояния)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
def test_latency_does_not_degrade_over_time(mock_config, sample_data, trained_pipeline):
    """100-й predict не должен проседать по скорости по сравнению с первыми запусками."""
    target = mock_config.data.tabular.target_col
    single_row = sample_data.iloc[[0]].drop(columns=[target])

    for _ in range(WARMUP_RUNS):
        trained_pipeline.predict(single_row)

    all_latencies = _measure_latencies(trained_pipeline, single_row, 100)
    early_median = np.median(all_latencies[:10])
    late_median = np.median(all_latencies[-10:])

    degradation_ratio = late_median / (early_median + 1e-9)
    assert degradation_ratio < 3.0, f"Латентность деградировала во времени в {degradation_ratio:.2f} раз."