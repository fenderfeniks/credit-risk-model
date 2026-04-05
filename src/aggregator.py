"""
aggregator.py

Агрегация сырых данных кредитной истории до уровня клиента (id).

Использование:
    agg = CreditHistoryAggregatorFast()

    # PASS 1 — собираем глобальную статистику по всем файлам
    for fp in files:
        df_chunk = pd.read_parquet(fp)
        agg.update_global_stats(df_chunk)
    agg.finalize_global_stats()

    # PASS 2 — трансформируем данные
    df_agg = agg.transform(df)
"""

import numpy as np
import pandas as pd
from collections import Counter


class CreditHistoryAggregatorFast:
    """
    Агрегатор кредитной истории клиентов.

    Работает в два прохода:
    1. update_global_stats() — итеративно собирает статистику по всему датасету
    2. finalize_global_stats() — фиксирует глобальные метрики (мода, редкие бины)
    3. transform() — применяет агрегацию к батчу данных

    Parameters
    ----------
    rare_threshold : float
        Порог доли для определения редкого бина. По умолчанию 0.05 (5%).
    """

    def __init__(self, rare_threshold=0.05):
        self.rare_threshold = rare_threshold

        # Колонки флагов просрочек
        self.overdue_zero_cols = [
            'is_zero_loans5', 'is_zero_loans530',
            'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90'
        ]

        # Колонки флагов утилизации
        self.util_zero_cols = [
            'is_zero_util', 'is_zero_over2limit', 'is_zero_maxover2limit'
        ]

        # Категориальные колонки кредитов
        # enc_loans_account_cur исключён — нулевая предсказательная сила
        # (~99.7% кредитов в одной валюте)
        self.loan_cat_cols = [
            'enc_loans_account_holder_type',
            'enc_loans_credit_status',
            'enc_loans_credit_type'
        ]

        # Бинаризованные числовые колонки для агрегации
        self.selected_pre_cols = [
            'pre_since_opened',
            'pre_since_confirmed',
            'pre_loans_credit_limit',
            'pre_loans_next_pay_summ',
            'pre_loans_outstanding',
            'pre_loans_max_overdue_sum',
            'pre_loans_credit_cost_rate',
            'pre_util',
            'pre_over2limit',
            'pre_maxover2limit',
        ]

        # Внутренние состояния — заполняются в процессе обучения
        self.value_counters_              = {col: Counter() for col in self.selected_pre_cols}
        self.paym_codes_                  = set()
        self.global_mode_map_             = {}
        self.rare_bins_map_               = {}
        self.paym_cols_                   = []
        self.available_pre_cols_          = []
        self.available_loan_cat_cols_     = []
        self.available_overdue_zero_cols_ = []
        self.available_util_zero_cols_    = []
        self._finalized                   = False

    def update_global_stats(self, df):
        """
        PASS 1: итеративно обновляет глобальную статистику по батчу данных.
        Вызывать для каждого файла/батча до finalize_global_stats().

        Parameters
        ----------
        df : pd.DataFrame
            Сырые данные одного батча (строки кредитов).
        """
        if not self.paym_cols_:
            self.paym_cols_ = sorted(
                [c for c in df.columns if c.startswith('enc_paym_')],
                key=lambda x: int(x.split('_')[-1])
            )

        if not self.available_pre_cols_:
            self.available_pre_cols_          = [c for c in self.selected_pre_cols if c in df.columns]
            self.available_loan_cat_cols_     = [c for c in self.loan_cat_cols if c in df.columns]
            self.available_overdue_zero_cols_ = [c for c in self.overdue_zero_cols if c in df.columns]
            self.available_util_zero_cols_    = [c for c in self.util_zero_cols if c in df.columns]

        # Накапливаем счётчики значений для бинаризованных колонок
        for col in self.available_pre_cols_:
            vc = df[col].value_counts(dropna=False)
            self.value_counters_[col].update(vc.to_dict())

        # Собираем уникальные коды платежей
        for col in self.paym_cols_:
            self.paym_codes_.update(pd.unique(df[col].dropna()))

    def finalize_global_stats(self):
        """
        PASS 1 завершение: вычисляет глобальную моду и редкие бины
        на основе накопленной статистики. Вызывать один раз после всех update_global_stats().
        """
        for col in self.available_pre_cols_:
            counter = self.value_counters_[col]
            total   = sum(counter.values())

            if total == 0:
                self.global_mode_map_[col] = np.nan
                self.rare_bins_map_[col]   = set()
                continue

            sorted_items           = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            self.global_mode_map_[col] = sorted_items[0][0]
            self.rare_bins_map_[col]   = {
                k for k, v in counter.items()
                if (v / total) < self.rare_threshold
            }

        self.paym_codes_ = sorted(self.paym_codes_)
        self._finalized  = True

    @staticmethod
    def _change_count_fast(df, col):
        """
        Считает количество смен значения колонки col в рамках каждого id.
        Статический метод — не зависит от состояния объекта.

        Parameters
        ----------
        df  : pd.DataFrame — данные с колонками id и col
        col : str          — имя колонки

        Returns
        -------
        pd.Series — количество смен значения для каждого id
        """
        x       = df[['id', col]].copy()
        prev    = x.groupby('id')[col].shift(1)
        changed = (x[col] != prev).astype(np.int8)
        out     = changed.groupby(x['id']).sum() - 1
        out[out < 0] = 0
        return out

    def transform(self, df):
        """
        PASS 2: агрегирует сырые данные до уровня клиента (id).
        Требует предварительного вызова finalize_global_stats().

        Parameters
        ----------
        df : pd.DataFrame
            Сырые данные (строки кредитов).

        Returns
        -------
        pd.DataFrame
            Агрегированный датафрейм — одна строка на клиента.
        """
        if not self._finalized:
            raise RuntimeError(
                "Сначала вызови update_global_stats() по всем файлам "
                "и finalize_global_stats()."
            )

        df    = df.sort_values(['id', 'rn']).reset_index(drop=True)
        parts = [self._build_base_features(df)]

        if self.available_overdue_zero_cols_:
            parts.append(self._build_overdue_features(df))
        if self.available_util_zero_cols_:
            parts.append(self._build_util_features(df))
        if self.available_loan_cat_cols_:
            parts.append(self._build_encoded_loan_features(df))
        if self.paym_cols_:
            parts.append(self._build_payment_features(df))
        if self.available_pre_cols_:
            parts.append(self._build_binned_pre_features(df))

        result = parts[0]
        for part in parts[1:]:
            result = result.merge(part, on='id', how='left')

        return result

    # ------------------------------------------------------------------
    # Приватные методы построения признаков
    # ------------------------------------------------------------------

    def _build_base_features(self, df):
        """
        Базовые характеристики кредитной истории клиента:
        количество кредитов, длина истории, доля незакрытых.
        """
        g   = df.groupby('id', sort=False)
        out = g.agg(
            num_credits=('rn', 'size'),
            max_rn     =('rn', 'max'),
        ).reset_index()

        if 'pclose_flag' in df.columns:
            tmp = g['pclose_flag'].agg(['mean']).reset_index()
            tmp.columns = ['id', 'pclose_flag_share']
            out = out.merge(tmp, on='id', how='left')

        if 'fclose_flag' in df.columns:
            tmp = g['fclose_flag'].agg(['mean']).reset_index()
            tmp.columns = ['id', 'fclose_flag_share']
            out = out.merge(tmp, on='id', how='left')

        return out

    def _build_overdue_features(self, df):
        """
        Признаки просрочек по пяти категориям тяжести:
        до 5 дней, 5-30, 30-60, 60-90, 90+ дней.
        Вычисляет доли и счётчики просрочек, а также агрегаты
        по серьёзным (60-90+) и любым просрочкам.
        """
        g          = df.groupby('id', sort=False)
        cols       = self.available_overdue_zero_cols_
        rename_map = {
            'is_zero_loans5'   : 'loans5',
            'is_zero_loans530' : 'loans530',
            'is_zero_loans3060': 'loans3060',
            'is_zero_loans6090': 'loans6090',
            'is_zero_loans90'  : 'loans90',
        }

        # Доля кредитов с просрочкой (1 - доля без просрочки)
        overdue_share = 1 - g[cols].mean()
        overdue_share.columns = [f'{rename_map[c]}_overdue_share' for c in cols]

        # Абсолютное число просрочек
        overdue_count         = g[cols].sum()
        overdue_count         = g.size().to_frame('num_tmp').join(overdue_count)
        for c in cols:
            overdue_count[c]  = overdue_count['num_tmp'] - overdue_count[c]
        overdue_count         = overdue_count[cols]
        overdue_count.columns = [f'{rename_map[c]}_overdue_count' for c in cols]

        result = pd.concat([overdue_share, overdue_count], axis=1)

        # Серьёзные просрочки (60-90+ дней) — наиболее критичный сигнал дефолта
        if {'is_zero_loans6090', 'is_zero_loans90'}.issubset(df.columns):
            serious_flag      = (
                (1 - df['is_zero_loans6090']) | (1 - df['is_zero_loans90'])
            ).astype(np.int8)
            serious           = serious_flag.groupby(df['id']).agg(['mean', 'sum'])
            serious.columns   = ['share_serious_overdue', 'count_serious_overdue']
            result            = pd.concat([result, serious], axis=1)

        # Любые просрочки — интегральный показатель платёжной дисциплины
        any_overdue             = (1 - df[cols]).max(axis=1)
        any_overdue_agg         = any_overdue.groupby(df['id']).agg(['mean', 'max', 'sum'])
        any_overdue_agg.columns = ['share_any_overdue', 'has_any_overdue', 'count_any_overdue']
        result                  = pd.concat([result, any_overdue_agg], axis=1)

        return result.reset_index()

    def _build_util_features(self, df):
        """
        Признаки утилизации кредитного лимита и превышений.
        Высокая утилизация и превышения — индикаторы финансового стресса.
        """
        g          = df.groupby('id', sort=False)
        cols       = self.available_util_zero_cols_
        rename_map = {
            'is_zero_util'         : 'util',
            'is_zero_over2limit'   : 'over2limit',
            'is_zero_maxover2limit': 'maxover2limit',
        }

        # Доля кредитов с ненулевой утилизацией
        nonzero_share         = 1 - g[cols].mean()
        nonzero_share.columns = [f'{rename_map[c]}_nonzero_share' for c in cols]

        # Абсолютное число кредитов с ненулевой утилизацией
        nonzero_count         = g[cols].sum()
        nonzero_count         = g.size().to_frame('num_tmp').join(nonzero_count)
        for c in cols:
            nonzero_count[c]  = nonzero_count['num_tmp'] - nonzero_count[c]
        nonzero_count         = nonzero_count[cols]
        nonzero_count.columns = [f'{rename_map[c]}_nonzero_count' for c in cols]

        return pd.concat([nonzero_share, nonzero_count], axis=1).reset_index()

    def _build_encoded_loan_features(self, df):
        """
        Признаки категориальных характеристик кредитов:
        тип держателя, статус, тип кредита.
        Для каждой колонки — nunique, first, last, change_count.
        """
        out   = []
        g_all = df.groupby('id', sort=False)

        for col in self.available_loan_cat_cols_:
            tmp = g_all[col].agg(['nunique', 'first', 'last']).reset_index()
            tmp.columns = ['id', f'{col}_nunique', f'{col}_first', f'{col}_last']

            change_count = self._change_count_fast(df, col).rename(
                f'{col}_change_count'
            ).reset_index()
            tmp = tmp.merge(change_count, on='id', how='left')

            # changed_ever только для типа кредита — остальные имели низкую важность
            if col == 'enc_loans_credit_type':
                tmp[f'{col}_changed_ever'] = (
                    tmp[f'{col}_change_count'] > 0
                ).astype(np.int8)

            # first и last для типа держателя имели нулевую важность
            if col == 'enc_loans_account_holder_type':
                tmp = tmp.drop(
                    columns=[f'{col}_first', f'{col}_last'],
                    errors='ignore'
                )

            out.append(tmp)

        result = out[0]
        for part in out[1:]:
            result = result.merge(part, on='id', how='left')

        return result

    def _build_payment_features(self, df):
        """
        Признаки платёжного поведения на основе enc_paym_* колонок.
        Агрегирует разнообразие статусов, частоту смен и доли кодов платежей
        по строкам, затем по клиентам.
        """
        paym = df[['id', 'rn'] + self.paym_cols_].copy()
        vals = paym[self.paym_cols_]

        # Признаки на уровне строки (один кредит в один момент)
        paym['paym_row_nunique']      = vals.nunique(axis=1)
        paym['paym_row_change_count'] = (
            vals.iloc[:, 1:].values != vals.iloc[:, :-1].values
        ).sum(axis=1).astype(np.int16)
        paym['paym_row_first'] = vals.iloc[:, 0]
        paym['paym_row_last']  = vals.iloc[:, -1]

        # Доли наиболее частых кодов (0=норма, 1=закрыт, 3=просрочка, 4=дефолт)
        top_codes = [c for c in [0, 1, 3, 4] if c in self.paym_codes_]
        for code in top_codes:
            paym[f'paym_row_share_code_{code}'] = (
                vals == code
            ).mean(axis=1).astype(np.float32)

        row_cols = [
            'paym_row_nunique',
            'paym_row_change_count',
            'paym_row_first',
            'paym_row_last',
        ] + [f'paym_row_share_code_{c}' for c in top_codes]

        agg         = paym.groupby('id', sort=False)[row_cols].agg(['mean', 'max'])
        agg.columns = ['_'.join(col) for col in agg.columns]
        agg         = agg.reset_index().fillna(0)

        # paym_row_last_max имел нулевую важность
        agg = agg.drop(columns=['paym_row_last_max'], errors='ignore')

        return agg

    def _build_binned_pre_features(self, df):
        """
        Признаки бинаризованных числовых колонок (pre_*).
        Для каждой колонки: nunique, first, last, change_count,
        share_global_mode (типичность клиента), share_rare_bins (аномальность).
        """
        out   = []
        g_all = df.groupby('id', sort=False)

        # Для этих колонок share_rare_bins оказался константным — не создаём
        cols_without_rare_bins = {
            'pre_since_confirmed',
            'pre_loans_max_overdue_sum',
            'pre_since_opened',
            'pre_till_pclose',
            'pre_loans_next_pay_summ',
            'pre_loans_outstanding',
        }

        for col in self.available_pre_cols_:
            tmp = g_all[col].agg(['nunique', 'first', 'last']).reset_index()
            tmp.columns = ['id', f'{col}_nunique', f'{col}_first', f'{col}_last']

            change_count = self._change_count_fast(df, col).rename(
                f'{col}_change_count'
            ).reset_index()
            tmp = tmp.merge(change_count, on='id', how='left')

            # changed_ever имел низкую важность для всех pre_* колонок
            tmp[f'{col}_changed_ever'] = (
                tmp[f'{col}_change_count'] > 0
            ).astype(np.int8)

            # Доля записей в доминирующем бине — насколько типичен клиент
            global_mode       = self.global_mode_map_[col]
            share_global_mode = (
                (df[col] == global_mode)
                .groupby(df['id'])
                .mean()
                .rename(f'{col}_share_global_mode')
                .reset_index()
            )
            tmp = tmp.merge(share_global_mode, on='id', how='left')

            # Доля записей в редких бинах — аномальное поведение клиента
            if col not in cols_without_rare_bins:
                rare_bins       = self.rare_bins_map_[col]
                share_rare_bins = (
                    df[col].isin(rare_bins)
                    .groupby(df['id'])
                    .mean()
                    .rename(f'{col}_share_rare_bins')
                    .reset_index()
                )
                tmp = tmp.merge(share_rare_bins, on='id', how='left')

            out.append(tmp)

        result = out[0]
        for part in out[1:]:
            result = result.merge(part, on='id', how='left')

        return result
