from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from omegaconf import DictConfig, OmegaConf

from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


# ============================================================
# 1. ПРЕПРОЦЕССИНГ (Очистка, пропуски, выбросы)
# ============================================================

 
class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    Класс технической предобработки.
    Отвечает за умную импутацию пропусков по бизнес-правилам,
    схлопывание редких категорий и фильтрацию констант.
 
    Логика:
      1. На fit фиксируются списки колонок (categorical / numeric),
         дальше все шаги (fit и transform) работают только с ними —
         никаких повторных select_dtypes / hasattr(..., 'cat') проверок.
      2. Категориальными считаются все колонки, кроме перечисленных
         в exact_excludes / prefix_excludes и таргета.
    """
 
    def __init__(self, config: DictConfig):
        self.full_cfg = config
        self.cfg = config.data.tabular
        self.target_col = self.cfg.target_col
 
        self.num_strategy = getattr(self.cfg, 'num_fill_strategy', 'median')
        self.cat_strategy = getattr(self.cfg, 'cat_fill_strategy', 'unknown')
        self.exact_excludes = set(self.cfg.exact_excludes)
        self.prefix_excludes = list(self.cfg.prefix_excludes)
 
        self.drop_cols = list(self.cfg.drop_cols) if self.cfg.drop_cols else []
        self.skip_imputation = set(getattr(self.cfg, 'skip_imputation_cols', []))
        self.top_n_categories = getattr(self.cfg, 'top_n_categories', 20)
 
        # Автоматический guard: если у "потенциально категориальной" колонки
        # уникальных значений больше этого порога - считаем её числовой,
        # а не бизнес-категорией (иначе получим ID-подобный мусор в category).
        self.max_cardinality_for_categorical = getattr(
            self.cfg, 'max_cardinality_for_categorical', 100
        )
 
        # Зафиксированные на fit списки колонок (заполняются в fit)
        self.cols_to_cast_categorical_ = []
        self.numeric_cols_ = []
        self.categorical_cols_ = []
 
    @staticmethod
    def _safe_cast_to_category(series: pd.Series) -> pd.Series:
        """
        Приводит серию к category через строковое представление, но
        НЕ трогает настоящие NaN (иначе astype(str) превратит их в
        буквальную строку 'nan', и fillna перестанет их находить).
        Нужно, чтобы CatBoost не падал на float-категориях
        (0.0, 1.0, ...) — он требует int или str внутри cat_features.
        """
        mask_na = series.isna()
        result = series.astype(str)
        result = result.mask(mask_na, np.nan)
        return result.astype('category')
 
    def fit(self, X: pd.DataFrame, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        all_columns = X.columns.tolist()
 
        # ==========================================================
        # ШАГ 1: ОПРЕДЕЛЯЕМ КОЛОНКИ ДЛЯ СМЕНЫ ТИПА (по бизнес-правилам конфига)
        # ==========================================================
        all_excludes = self.exact_excludes | {self.target_col}
        candidate_categorical = [
            col for col in all_columns
            if col not in all_excludes
            and not any(col.startswith(pref) for pref in self.prefix_excludes)
        ]
 
        # Guard по кардинальности: для колонок, которые ИЗНАЧАЛЬНО числовые
        # (не object/category), проверяем nunique. Если уникальных значений
        # больше порога - это, скорее всего, непрерывный числовой признак
        # или ID-подобное поле, а не бизнес-категория, поэтому оставляем
        # её числовой. Уже текстовые (object) колонки проверке не подлежат -
        # они по определению категориальные вне зависимости от кардинальности.
        self.high_cardinality_numeric_cols_ = []
        self.cols_to_cast_categorical_ = []
        for col in candidate_categorical:
            if col not in X.columns:
                continue
            is_originally_numeric = pd.api.types.is_numeric_dtype(X[col])
            if is_originally_numeric:
                nunique = X[col].nunique(dropna=True)
                if nunique > self.max_cardinality_for_categorical:
                    self.high_cardinality_numeric_cols_.append(col)
                    continue
            self.cols_to_cast_categorical_.append(col)
 
        self.fill_values_ = {}
        self.outlier_bounds_ = {}
        self.top_categories_map_ = {}
 
        X_fit = X.copy()
 
        # Смена типов данных — строго на первом месте.
        # ВАЖНО: если колонка изначально числовая (float/int), просто
        # .astype('category') оставит значения-float ВНУТРИ категорий
        # (0.0, 1.0, ...) — CatBoost такие категории не принимает
        # ("cat_features must be integer or string"). Поэтому сначала
        # приводим значение к строке, потом уже кастуем в category.
        for col in self.cols_to_cast_categorical_:
            if col in X_fit.columns:
                X_fit[col] = self._safe_cast_to_category(X_fit[col])
 
        # Исключаем технические колонки из анализа пропусков и констант
        technical_cols = self.drop_cols + []
        X_fit = X_fit.drop(columns=technical_cols, errors='ignore')
 
        # ==========================================================
        # ШАГ 2: АНАЛИЗ МУСОРНЫХ КОЛОНОК (Пропуски и Константы)
        # ==========================================================
        missing_frac = X_fit.isnull().mean()
        max_missing = getattr(self.cfg, 'max_missing_pct', 0.90)
        cols_to_drop_missing = missing_frac[missing_frac > max_missing].index.tolist()
 
        cols_to_drop_const = []
        max_const = getattr(self.cfg, 'max_constant_pct', 0.999)
        for col in X_fit.columns:
            val_counts = X_fit[col].value_counts(normalize=True, dropna=False)
            if not val_counts.empty and val_counts.iloc[0] > max_const:
                cols_to_drop_const.append(col)
 
        # Формируем финальный черный список колонок
        self.learned_drop_cols_ = list(set(technical_cols + cols_to_drop_missing + cols_to_drop_const))
        X_fit = X_fit.drop(columns=self.learned_drop_cols_, errors='ignore')
 
        # ==========================================================
        # ЗАКРЕПЛЯЕМ ФИНАЛЬНЫЕ СПИСКИ num/cat КОЛОНОК
        # (считаем один раз, после всех дропов — дальше используем только их)
        # ==========================================================
        self.numeric_cols_ = [
            col for col in X_fit.columns
            if col not in self.cols_to_cast_categorical_
        ]
        self.categorical_cols_ = [
            col for col in X_fit.columns
            if col in self.cols_to_cast_categorical_
        ]
 
        # ==========================================================
        # ШАГ 3: КАРТИРОВАНИЕ ТОП-КАТЕГОРИЙ (Схлопывание хвостов)
        # ==========================================================
        for col in self.categorical_cols_:
            top_vals = X_fit[col].value_counts().index[:self.top_n_categories].tolist()
            self.top_categories_map_[col] = top_vals
 
        # ==========================================================
        # ШАГ 4: РАСЧЕТ ДЕФОЛТНЫХ ЗАГЛУШЕК ДЛЯ ОСТАВШИХСЯ НАХОДОК
        # ==========================================================
        for col in X_fit.columns:
            if col in self.skip_imputation:
                continue
            if col in self.numeric_cols_:
                val = X_fit[col].median() if self.num_strategy == 'median' else X_fit[col].mean()
                self.fill_values_[col] = 0 if pd.isna(val) else val
            else:
                mode = X_fit[col].mode()
                self.fill_values_[col] = mode[0] if self.cat_strategy == 'mode' and not mode.empty else 'Unknown'
 
        # ==========================================================
        # ШАГ 5: РАСЧЕТ ГРАНИЦ ДЛЯ ВЫБРОСОВ (Строго по numeric_cols_)
        # ==========================================================
        outlier_method = getattr(self.cfg, 'outlier_method', 'none')
 
        if outlier_method == 'zscore':
            thresh = getattr(self.cfg, 'outlier_threshold', 3.0)
            for col in self.numeric_cols_:
                mean, std = X_fit[col].mean(), X_fit[col].std()
                if std > 0:
                    self.outlier_bounds_[col] = (mean - thresh * std, mean + thresh * std)
        elif outlier_method == 'iqr':
            thresh = getattr(self.cfg, 'outlier_threshold', 1.5)
            for col in self.numeric_cols_:
                q1 = X_fit[col].quantile(0.25)
                q3 = X_fit[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    self.outlier_bounds_[col] = (q1 - thresh * iqr, q3 + thresh * iqr)
 
        return self
 
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ['fill_values_', 'learned_drop_cols_', 'top_categories_map_'])
        X_transformed = X.copy()
 
        # ==========================================================
        # 1. ТЕХНИЧЕСКАЯ ОЧИСТКА: смена типов — по тому же списку, что и на fit
        #    (str перед category — см. комментарий в fit)
        # ==========================================================
        for col in self.cols_to_cast_categorical_:
            if col in X_transformed.columns:
                X_transformed[col] = self._safe_cast_to_category(X_transformed[col])
 
        # ==========================================================
        # 2. СХЛОПЫВАНИЕ РЕДКИХ КАТЕГОРИЙ — только по self.categorical_cols_
        # ==========================================================
        for col in self.categorical_cols_:
            if col not in X_transformed.columns:
                continue
            top_vals = self.top_categories_map_[col]
            if 'other_collapsed' not in X_transformed[col].cat.categories:
                X_transformed[col] = X_transformed[col].cat.add_categories('other_collapsed')
            X_transformed[col] = X_transformed[col].where(
                X_transformed[col].isin(top_vals), 'other_collapsed'
            )
 
        # ==========================================================
        # 3. ДРОП НЕНУЖНЫХ, ПУСТЫХ И КОНСТАНТНЫХ КОЛОНОК
        # ==========================================================
        X_transformed = X_transformed.drop(columns=self.learned_drop_cols_, errors='ignore')
 
        # ==========================================================
        # 4. ФИНАЛЬНОЕ ЗАПОЛНЕНИЕ NaN
        #    Категориальным колонкам заранее регистрируем fill-значение
        #    как разрешенную категорию, чтобы fillna не падал.
        # ==========================================================
        for col in self.categorical_cols_:
            if col not in X_transformed.columns:
                continue
            fill_val = self.fill_values_.get(col)
            if fill_val is not None and fill_val not in X_transformed[col].cat.categories:
                X_transformed[col] = X_transformed[col].cat.add_categories([fill_val])
 
        X_transformed = X_transformed.fillna(self.fill_values_)
 
        # ==========================================================
        # 5. ОБРЕЗКА ЭКСТРЕМАЛЬНЫХ ВЫБРОСОВ (Clipping) — только по numeric_cols_
        # ==========================================================
        for col, (lower, upper) in self.outlier_bounds_.items():
            if col in X_transformed.columns and col in self.numeric_cols_:
                X_transformed[col] = X_transformed[col].clip(lower, upper)
 
        return X_transformed
 
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, ['learned_drop_cols_'])
 
        if input_features is not None:
            feature_names = list(input_features)
        elif hasattr(self, 'feature_names_in_'):
            feature_names = list(self.feature_names_in_)
        else:
            raise ValueError(
                "Трансформер не имеет сохраненных feature_names_in_, "
                "и параметр input_features не был передан."
            )
 
        final_features = [
            col for col in feature_names
            if col not in self.learned_drop_cols_
        ]
 
        return np.array(final_features, dtype=object)

# ============================================================
# 2. ИНЖЕНЕРИЯ ПРИЗНАКОВ (Генерация новых фичей)
# ============================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        self.cfg = config or {}
        
        # Реестр всех генерируемых фичей
        self.new_features_ = [
            'overall_typicalness_score',
            'overall_anomaly_score',
            'is_absolutely_perfect_payer',
            'is_chronic_defaulter',        # НОВОЕ
            'cat_last_util_and_time',
            'cat_last_type_and_limit',
            'cat_limit_transition',
            'cat_type_transition',
            'cat_status_transition',       # НОВОЕ
            'cat_util_and_history',        # НОВОЕ
            'cat_diversity_profile'        # НОВОЕ
        ]

    def fit(self, X: pd.DataFrame, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def _safe_concat(self, df: pd.DataFrame, col1: str, col2: str, out_name: str, sep: str = "_"):
        """Безопасная строковая склейка."""
        if col1 in df.columns and col2 in df.columns:
            c1_str = df[col1].fillna(-999).astype(int).astype(str)
            c2_str = df[col2].fillna(-999).astype(int).astype(str)
            df[out_name] = c1_str + sep + c2_str

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)        
        X_transformed = X.copy()

        # =====================================================================
        # 1. МАКРО-ИНДЕКСЫ ТИПИЧНОСТИ И АНОМАЛИЙ
        # =====================================================================
        global_mode_cols = [c for c in X_transformed.columns if c.endswith('_share_global_mode')]
        if global_mode_cols:
            X_transformed['overall_typicalness_score'] = X_transformed[global_mode_cols].mean(axis=1)

        rare_bins_cols = [c for c in X_transformed.columns if c.endswith('_share_rare_bins')]
        if rare_bins_cols:
            X_transformed['overall_anomaly_score'] = X_transformed[rare_bins_cols].mean(axis=1)

        # =====================================================================
        # 2. ЖЕСТКИЕ ФЛАГИ ЭКСТРЕМУМОВ
        # =====================================================================
        if 'paym_row_share_code_0_mean' in X_transformed.columns:
            X_transformed['is_absolutely_perfect_payer'] = (
                X_transformed['paym_row_share_code_0_mean'] >= 0.95
            ).astype(np.int8)

        # НОВОЕ: Хронический должник (быстрая отсечка для класса 1)
        if 'share_any_overdue' in X_transformed.columns and 'share_serious_overdue' in X_transformed.columns:
            X_transformed['is_chronic_defaulter'] = (
                (X_transformed['share_any_overdue'] > 0.3) | 
                (X_transformed['share_serious_overdue'] > 0.0)
            ).astype(np.int8)

        # =====================================================================
        # 3. КАТЕГОРИАЛЬНЫЕ ЭВОЛЮЦИИ И СКЛЕЙКИ
        # =====================================================================
        self._safe_concat(X_transformed, 'pre_util_last', 'pre_since_opened_last', 'cat_last_util_and_time')
        self._safe_concat(X_transformed, 'enc_loans_credit_type_last', 'pre_loans_credit_limit_last', 'cat_last_type_and_limit')
        self._safe_concat(X_transformed, 'pre_loans_credit_limit_first', 'pre_loans_credit_limit_last', 'cat_limit_transition')
        self._safe_concat(X_transformed, 'enc_loans_credit_type_first', 'enc_loans_credit_type_last', 'cat_type_transition')
        
        # НОВОЕ: Эволюция статуса кредита
        self._safe_concat(X_transformed, 'first_enc_loans_credit_status', 'last_enc_loans_credit_status', 'cat_status_transition')

        # НОВОЕ: Прошлое (Доля просрочек) + Настоящее (Утилизация)
        if 'share_any_overdue' in X_transformed.columns and 'pre_util_last' in X_transformed.columns:
            # Превращаем непрерывную долю просрочек в понятные текстовые бакеты
            history_bins = pd.cut(
                X_transformed['share_any_overdue'], 
                bins=[-np.inf, 0.0, 0.2, np.inf], 
                labels=["Clean", "Mild", "Frequent"]
            ).astype(str)
            
            util_str = X_transformed['pre_util_last'].fillna(-999).astype(int).astype(str)
            X_transformed['cat_util_and_history'] = util_str + "_" + history_bins

        # НОВОЕ: Матрица кредитного опыта (Разнообразие типов + Общее число)
        if 'distinct_enc_loans_credit_type_count' in X_transformed.columns and 'num_credits' in X_transformed.columns:
            types_str = X_transformed['distinct_enc_loans_credit_type_count'].fillna(0).astype(int).astype(str)
            loans_str = X_transformed['num_credits'].fillna(0).astype(int).astype(str)
            X_transformed['cat_diversity_profile'] = types_str + "types_" + loans_str + "loans"

        return X_transformed

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            features = list(input_features)
        elif hasattr(self, 'feature_names_in_'):
            features = list(self.feature_names_in_)
        else:
            features = []

        actual_new_features = [f for f in self.new_features_ if f in self.new_features_]
        for f in actual_new_features:
            if f not in features:
                features.append(f)

        return np.array(features, dtype=object)