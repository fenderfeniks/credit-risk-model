from __future__ import annotations

import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def filter_bad_rows(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Удаление строк, у которых процент пропусков превышает допустимый порог."""
    max_row_missing = getattr(cfg.data.tabular, 'max_row_missing_pct', 0.50)

    row_missing_frac = df.isnull().mean(axis=1)
    good_rows_mask = row_missing_frac <= max_row_missing

    removed_count = len(df) - good_rows_mask.sum()
    if removed_count > 0:
        logger.info(
            f"[DATA CLEANING] Удалено {removed_count} строк, содержащих > {max_row_missing * 100}% пропусков."
        )
        df = df[good_rows_mask].reset_index(drop=True)

    return df


def _split_client_ids(
    id_col: pd.Series,
    fraction: float,
    seed: int,
    stratify_labels: pd.Series | None = None,
) -> Tuple[set, set]:
    """
    Делит уникальных клиентов на две группы в заданной пропорции.
    Стратификация (если задана) применяется на уровне клиента, не строки.
    """
    unique_ids = id_col.drop_duplicates()

    strat = None
    if stratify_labels is not None:
        # Один лейбл на клиента: берем таргет по последней встреченной строке клиента.
        # Если у клиента таргет мог отличаться между строками — это отдельный вопрос
        # к бизнес-логике таргета, но для сплита нужен ровно один лейбл на группу.
        per_client_label = (
            pd.DataFrame({"id_col": id_col, "label": stratify_labels})
            .drop_duplicates(subset="id_col", keep="last")
            .set_index("id_col")["label"]
        )
        strat = per_client_label.loc[unique_ids].values

    try:
        ids_a, ids_b = train_test_split(
            unique_ids, test_size=fraction, random_state=seed, stratify=strat
        )
    except ValueError as e:
        logger.warning(f"Ошибка стратификации при делении id_col: {e}. Случайный сплит.")
        ids_a, ids_b = train_test_split(
            unique_ids, test_size=fraction, random_state=seed, stratify=None
        )

    return set(ids_a), set(ids_b)


def split_data(cfg: DictConfig, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разбивает датафрейм на Train, Val, Test на уровне client_id (group split),
    чтобы строки одного пользователя не попадали в разные выборки.
    """
    df = filter_bad_rows(cfg, df)

    target_col = cfg.data.tabular.target_col if cfg.data.tabular else None
    id_col = cfg.data.tabular.id_col  # например "client_id"
    task_type = cfg.task_type

    test_size = cfg.data.test_size
    val_size = cfg.data.val_size

    if test_size + val_size >= 1.0:
        raise ValueError("Сумма test_size и val_size должна быть < 1.0!")

    if id_col not in df.columns:
        raise ValueError(f"Колонка группировки '{id_col}' отсутствует в данных!")

    logger.info(f"Разбиение данных по client_id (Задача: {task_type})...")

    use_stratify = (
        target_col and target_col in df.columns
        and task_type in ["binary", "multiclass", "sequence_classification"]
    )
    stratify_labels = df[target_col] if use_stratify else None

    # --- TEST SPLIT (на уровне client_id) ---
    if test_size > 0:
        train_val_ids, test_ids = _split_client_ids(
            df[id_col], fraction=test_size, seed=cfg.seed, stratify_labels=stratify_labels
        )
    else:
        train_val_ids, test_ids = set(df[id_col]), set()

    train_val_df = df[df[id_col].isin(train_val_ids)]
    test_df = df[df[id_col].isin(test_ids)]

    # --- VAL SPLIT (тоже на уровне client_id, внутри train_val) ---
    if val_size > 0:
        val_fraction = val_size / (1.0 - test_size)
        strat_val = train_val_df[target_col] if use_stratify else None

        train_ids, val_ids = _split_client_ids(
            train_val_df[id_col], fraction=val_fraction, seed=cfg.seed, stratify_labels=strat_val
        )
    else:
        train_ids, val_ids = set(train_val_df[id_col]), set()

    train_df = train_val_df[train_val_df[id_col].isin(train_ids)]
    val_df = train_val_df[train_val_df[id_col].isin(val_ids)]

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Sanity-check: пересечений client_id между сплитами быть не должно
    assert not (train_ids & val_ids), "Leakage: пересечение client_id между train и val!"
    assert not ((train_ids | val_ids) & test_ids), "Leakage: пересечение client_id между train/val и test!"

    logger.info(
        f"Размеры выборок (строки): Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}. "
        f"Клиентов: Train {len(train_ids)} | Val {len(val_ids)} | Test {len(test_ids)}"
    )
    return train_df, val_df, test_df