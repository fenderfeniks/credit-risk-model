-- ============================================================================
-- АГРЕГАЦИЯ КРЕДИТНОЙ ИСТОРИИ (PySpark SQL Version v3.0 - Precomputed Stats)
-- Изменения для Spark:
--  1. ::INT заменено на CAST(... AS INT)
--  2. FIRST(col) заменено на быстрые функции min_by() и max_by()
--  3. EXCLUDE заменено на EXCEPT
--  4. Оконные функции для мод и редких бинов удалены. Используем инъекции из GlobalStatCompiler.
-- ============================================================================

WITH raw_features AS (
    SELECT * FROM {features_source}
    WHERE {id_col} IS NOT NULL
),

-- ============================================================================
-- БЛОК 1: БАЗОВЫЕ СЧЕТЧИКИ ПО КЛИЕНТУ
-- ============================================================================
base_features AS (
    SELECT
        {id_col},
        COUNT(*) AS num_credits,
        MAX(rn)  AS max_rn,
        MIN(rn)  AS min_rn,
        (MAX(rn) - MIN(rn)) AS rn_span,
        CAST((COUNT(*) = 1) AS INT) AS is_single_credit,
        (COUNT(*) * 1.0 / NULLIF(MAX(rn), 0)) AS rn_density,
        AVG(CAST(pclose_flag AS INT)) AS pclose_flag_share,
        MAX(CAST(pclose_flag AS INT)) AS pclose_flag_any,
        AVG(CAST(fclose_flag AS INT)) AS fclose_flag_share,
        MAX(CAST(fclose_flag AS INT)) AS fclose_flag_any
    FROM raw_features
    GROUP BY {id_col}
),

-- ============================================================================
-- БЛОК 2 И 3: ПРОСРОЧКИ И УТИЛИЗАЦИЯ
-- ============================================================================
overdue_util_agg AS (
    SELECT
        {id_col},
        
        AVG(CAST(is_zero_loans5 AS INT))    AS loans5_zero_share,
        AVG(CAST(is_zero_loans530 AS INT))  AS loans530_zero_share,
        AVG(CAST(is_zero_loans3060 AS INT)) AS loans3060_zero_share,
        AVG(CAST(is_zero_loans90 AS INT))   AS loans90_zero_share,
        
        MAX(CAST((is_zero_loans5=0) AS INT))    AS loans5_has_overdue,
        MAX(CAST((is_zero_loans530=0) AS INT))  AS loans530_has_overdue,
        MAX(CAST((is_zero_loans3060=0) AS INT)) AS loans3060_has_overdue,
        MAX(CAST((is_zero_loans90=0) AS INT))   AS loans90_has_overdue,
        
        AVG(CAST(is_zero_util AS INT))          AS util_zero_share,
        MAX(CAST((is_zero_util=0) AS INT))      AS util_has_nonzero,
        SUM(CAST((is_zero_util=0) AS INT))      AS util_nonzero_count,
        
        AVG(CAST(is_zero_over2limit AS INT))    AS over2limit_zero_share,
        MAX(CAST((is_zero_over2limit=0) AS INT)) AS over2limit_has_nonzero,
        
        AVG(CAST(is_zero_maxover2limit AS INT)) AS maxover2limit_zero_share,
        MAX(CAST((is_zero_maxover2limit=0) AS INT)) AS maxover2limit_has_nonzero,
        
        MAX(CASE WHEN is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS has_serious_overdue,
        SUM(CASE WHEN is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS count_serious_overdue,
        AVG(CASE WHEN is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS share_serious_overdue,
        
        MAX(CASE WHEN is_zero_loans5=0 OR is_zero_loans530=0 OR is_zero_loans3060=0 OR is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS has_any_overdue,
        SUM(CASE WHEN is_zero_loans5=0 OR is_zero_loans530=0 OR is_zero_loans3060=0 OR is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS count_any_overdue,
        AVG(CASE WHEN is_zero_loans5=0 OR is_zero_loans530=0 OR is_zero_loans3060=0 OR is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS share_any_overdue,
        
        SUM(CASE WHEN is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) * 1.0 
            / NULLIF(SUM(CAST((is_zero_loans5=0) AS INT)), 0) AS serious_to_mild_ratio

    FROM raw_features
    GROUP BY {id_col}
),

-- ============================================================================
-- БЛОК 5: ПЛАТЕЖИ (Внутристрочная логика)
-- ============================================================================
paym_long AS (
    SELECT {id_col}, rn,
           CAST(regexp_extract(month_col, '[0-9]+', 0) AS INT) AS paym_month,
           paym_status
    FROM raw_features
    UNPIVOT (
        paym_status FOR month_col IN (
            enc_paym_0, enc_paym_1, enc_paym_2, enc_paym_3, enc_paym_4,
            enc_paym_5, enc_paym_6, enc_paym_7, enc_paym_8, enc_paym_9,
            enc_paym_10, enc_paym_11, enc_paym_12, enc_paym_13, enc_paym_14,
            enc_paym_15, enc_paym_16, enc_paym_17, enc_paym_18, enc_paym_19,
            enc_paym_20
        )
    )
),

paym_lagged AS (
    SELECT 
        {id_col}, rn, paym_month, paym_status,
        LAG(paym_status) OVER (PARTITION BY {id_col}, rn ORDER BY paym_month) AS prev_paym_status
    FROM paym_long
    WHERE paym_status IS NOT NULL
),

paym_row_stats AS (
    SELECT
        {id_col},
        rn,
        COUNT(DISTINCT paym_status) AS paym_row_nunique,
        SUM(CAST((paym_status != prev_paym_status) AS INT)) AS paym_row_change_count,
        min_by(paym_status, paym_month) AS paym_row_first,
        max_by(paym_status, paym_month) AS paym_row_last,
        AVG(CAST((paym_status = '0') AS INT)) AS paym_row_share_code_0,
        AVG(CAST((paym_status = '1') AS INT)) AS paym_row_share_code_1,
        AVG(CAST((paym_status = '3') AS INT)) AS paym_row_share_code_3,
        AVG(CAST((paym_status = '4') AS INT)) AS paym_row_share_code_4
    FROM paym_lagged
    GROUP BY {id_col}, rn
),

paym_agg AS (
    SELECT
        {id_col},
        AVG(paym_row_nunique) AS paym_row_nunique_mean,
        MAX(paym_row_nunique) AS paym_row_nunique_max,
        AVG(paym_row_change_count) AS paym_row_change_count_mean,
        MAX(paym_row_change_count) AS paym_row_change_count_max,
        AVG(CAST(paym_row_first AS FLOAT)) AS paym_row_first_mean,
        AVG(CAST(paym_row_last AS FLOAT)) AS paym_row_last_mean,
        AVG(paym_row_share_code_0) AS paym_row_share_code_0_mean,
        AVG(paym_row_share_code_1) AS paym_row_share_code_1_mean,
        AVG(paym_row_share_code_3) AS paym_row_share_code_3_mean,
        AVG(paym_row_share_code_4) AS paym_row_share_code_4_mean
    FROM paym_row_stats
    GROUP BY {id_col}
),

-- ============================================================================
-- БЛОК 4 И 6: КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ 
-- Подставляем посчитанные моды и редкие бины напрямую из Питона через format()
-- ============================================================================
cat_features_prep AS (
    SELECT
        r.{id_col}, r.rn,
        
        r.pre_since_opened,
        CAST((r.pre_since_opened != LAG(r.pre_since_opened) OVER(PARTITION BY r.{id_col} ORDER BY r.rn)) AS INT) AS pre_since_opened_chg,
        CAST(COALESCE(r.pre_since_opened = {m_pre_since_opened}, FALSE) AS INT) AS pre_since_opened_is_mode,
        CAST(COALESCE(r.pre_since_opened IN ({rare_pre_since_opened}), FALSE) AS INT) AS pre_since_opened_is_rare,

        r.pre_since_confirmed,
        CAST((r.pre_since_confirmed != LAG(r.pre_since_confirmed) OVER(PARTITION BY r.{id_col} ORDER BY r.rn)) AS INT) AS pre_since_confirmed_chg,
        CAST(COALESCE(r.pre_since_confirmed = {m_pre_since_confirmed}, FALSE) AS INT) AS pre_since_confirmed_is_mode,

        r.pre_loans_credit_limit,
        CAST((r.pre_loans_credit_limit != LAG(r.pre_loans_credit_limit) OVER(PARTITION BY r.{id_col} ORDER BY r.rn)) AS INT) AS pre_loans_credit_limit_chg,
        CAST(COALESCE(r.pre_loans_credit_limit = {m_pre_loans_credit_limit}, FALSE) AS INT) AS pre_loans_credit_limit_is_mode,
        CAST(COALESCE(r.pre_loans_credit_limit IN ({rare_pre_loans_credit_limit}), FALSE) AS INT) AS pre_loans_credit_limit_is_rare,

        r.pre_loans_outstanding,
        CAST((r.pre_loans_outstanding != LAG(r.pre_loans_outstanding) OVER(PARTITION BY r.{id_col} ORDER BY r.rn)) AS INT) AS pre_loans_outstanding_chg,
        CAST(COALESCE(r.pre_loans_outstanding = {m_pre_loans_outstanding}, FALSE) AS INT) AS pre_loans_outstanding_is_mode,
        CAST(COALESCE(r.pre_loans_outstanding IN ({rare_pre_loans_outstanding}), FALSE) AS INT) AS pre_loans_outstanding_is_rare,

        r.pre_loans_credit_cost_rate,
        CAST((r.pre_loans_credit_cost_rate != LAG(r.pre_loans_credit_cost_rate) OVER(PARTITION BY r.{id_col} ORDER BY r.rn)) AS INT) AS pre_loans_credit_cost_rate_chg,
        CAST(COALESCE(r.pre_loans_credit_cost_rate = {m_pre_loans_credit_cost_rate}, FALSE) AS INT) AS pre_loans_credit_cost_rate_is_mode,
        CAST(COALESCE(r.pre_loans_credit_cost_rate IN ({rare_pre_loans_credit_cost_rate}), FALSE) AS INT) AS pre_loans_credit_cost_rate_is_rare,

        r.pre_util,
        CAST((r.pre_util != LAG(r.pre_util) OVER(PARTITION BY r.{id_col} ORDER BY r.rn)) AS INT) AS pre_util_chg,
        CAST(COALESCE(r.pre_util = {m_pre_util}, FALSE) AS INT) AS pre_util_is_mode,
        CAST(COALESCE(r.pre_util IN ({rare_pre_util}), FALSE) AS INT) AS pre_util_is_rare,

        r.pre_loans530,
        CAST((r.pre_loans530 != LAG(r.pre_loans530) OVER(PARTITION BY r.{id_col} ORDER BY r.rn)) AS INT) AS pre_loans530_chg,
        CAST(COALESCE(r.pre_loans530 = {m_pre_loans530}, FALSE) AS INT) AS pre_loans530_is_mode,
        CAST(COALESCE(r.pre_loans530 IN ({rare_pre_loans530}), FALSE) AS INT) AS pre_loans530_is_rare,

        r.pre_loans3060,
        CAST((r.pre_loans3060 != LAG(r.pre_loans3060) OVER(PARTITION BY r.{id_col} ORDER BY r.rn)) AS INT) AS pre_loans3060_chg,
        CAST(COALESCE(r.pre_loans3060 = {m_pre_loans3060}, FALSE) AS INT) AS pre_loans3060_is_mode,

        r.pre_loans90,
        CAST((r.pre_loans90 != LAG(r.pre_loans90) OVER(PARTITION BY r.{id_col} ORDER BY r.rn)) AS INT) AS pre_loans90_chg,
        CAST(COALESCE(r.pre_loans90 = {m_pre_loans90}, FALSE) AS INT) AS pre_loans90_is_mode,

        r.enc_loans_credit_type,
        CAST((r.enc_loans_credit_type != LAG(r.enc_loans_credit_type) OVER(PARTITION BY r.{id_col} ORDER BY r.rn)) AS INT) AS enc_loans_credit_type_chg,
        CAST(COALESCE(r.enc_loans_credit_type = {m_enc_loans_credit_type}, FALSE) AS INT) AS enc_loans_credit_type_is_mode,
        CAST(COALESCE(r.enc_loans_credit_type IN ({rare_enc_loans_credit_type}), FALSE) AS INT) AS enc_loans_credit_type_is_rare
    FROM raw_features r
),

cat_features_agg AS (
    SELECT
        {id_col},

        COUNT(DISTINCT pre_since_opened) AS pre_since_opened_nunique,
        min_by(pre_since_opened, rn) AS pre_since_opened_first,
        max_by(pre_since_opened, rn) AS pre_since_opened_last,
        SUM(pre_since_opened_chg) AS pre_since_opened_change_count,
        AVG(pre_since_opened_is_mode) AS pre_since_opened_share_global_mode,
        AVG(pre_since_opened_is_rare) AS pre_since_opened_share_rare_bins,

        COUNT(DISTINCT pre_since_confirmed) AS pre_since_confirmed_nunique,
        min_by(pre_since_confirmed, rn) AS pre_since_confirmed_first,
        max_by(pre_since_confirmed, rn) AS pre_since_confirmed_last,
        AVG(pre_since_confirmed_is_mode) AS pre_since_confirmed_share_global_mode,

        COUNT(DISTINCT pre_loans_credit_limit) AS pre_loans_credit_limit_nunique,
        min_by(pre_loans_credit_limit, rn) AS pre_loans_credit_limit_first,
        max_by(pre_loans_credit_limit, rn) AS pre_loans_credit_limit_last,
        SUM(pre_loans_credit_limit_chg) AS pre_loans_credit_limit_change_count,
        AVG(pre_loans_credit_limit_is_mode) AS pre_loans_credit_limit_share_global_mode,
        AVG(pre_loans_credit_limit_is_rare) AS pre_loans_credit_limit_share_rare_bins,

        COUNT(DISTINCT pre_util) AS pre_util_nunique,
        min_by(pre_util, rn) AS pre_util_first,
        max_by(pre_util, rn) AS pre_util_last,
        SUM(pre_util_chg) AS pre_util_change_count,
        AVG(pre_util_is_mode) AS pre_util_share_global_mode,
        AVG(pre_util_is_rare) AS pre_util_share_rare_bins,

        COUNT(DISTINCT pre_loans_outstanding) AS pre_loans_outstanding_nunique,
        max_by(pre_loans_outstanding, rn) AS pre_loans_outstanding_last,
        AVG(pre_loans_outstanding_is_mode) AS pre_loans_outstanding_share_global_mode,

        COUNT(DISTINCT pre_loans_credit_cost_rate) AS pre_loans_credit_cost_rate_nunique,
        max_by(pre_loans_credit_cost_rate, rn) AS pre_loans_credit_cost_rate_last,
        AVG(pre_loans_credit_cost_rate_is_mode) AS pre_loans_credit_cost_rate_share_global_mode,

        COUNT(DISTINCT pre_loans530) AS pre_loans530_nunique,
        MAX(pre_loans530_chg) AS pre_loans530_changed_ever,

        COUNT(DISTINCT pre_loans3060) AS pre_loans3060_nunique,
        max_by(pre_loans3060, rn) AS pre_loans3060_last,
        MAX(pre_loans3060_chg) AS pre_loans3060_changed_ever,

        COUNT(DISTINCT pre_loans90) AS pre_loans90_nunique,
        AVG(pre_loans90_is_mode) AS pre_loans90_share_global_mode,

        COUNT(DISTINCT enc_loans_credit_type) AS enc_loans_credit_type_nunique,
        min_by(enc_loans_credit_type, rn) AS enc_loans_credit_type_first,
        max_by(enc_loans_credit_type, rn) AS enc_loans_credit_type_last,
        SUM(enc_loans_credit_type_chg) AS enc_loans_credit_type_change_count

    FROM cat_features_prep
    GROUP BY {id_col}
),

-- ============================================================================
-- СБОРКА В ОДНУ ВИТРИНУ
-- ============================================================================
aggregated_features AS (
    SELECT
        bf.*,
        ou.* EXCEPT ({id_col}),
        pa.* EXCEPT ({id_col}),
        cf.* EXCEPT ({id_col})
    FROM base_features bf
    LEFT JOIN overdue_util_agg ou ON bf.{id_col} = ou.{id_col}
    LEFT JOIN paym_agg         pa ON bf.{id_col} = pa.{id_col}
    LEFT JOIN cat_features_agg cf ON bf.{id_col} = cf.{id_col}
)

SELECT
    f.*,
    t.* EXCEPT ({id_col})
FROM aggregated_features f
LEFT JOIN {target_source} t
  ON f.{id_col} = t.{id_col}