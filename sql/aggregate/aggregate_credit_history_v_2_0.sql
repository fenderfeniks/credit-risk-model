-- ============================================================================
-- АГРЕГАЦИЯ КРЕДИТНОЙ ИСТОРИИ (v2.0)
-- Диалект: DuckDB
-- Изменения: 
--  1. Убраны нулевые фичи (enc_loans_account_cur, pre_loans6090_first/last и др.)
--  2. Добавлены внутристрочные агрегаты для платежей (enc_paym_*)
--  3. Добавлены фичи share_global_mode и share_rare_bins (частота < 0.05)
--  4. Добавлены соотношения просрочек (serious vs mild)
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
        (COUNT(*) = 1)::INT AS is_single_credit,
        (COUNT(*) * 1.0 / NULLIF(MAX(rn), 0)) AS rn_density,
        AVG(pclose_flag::INT) AS pclose_flag_share,
        MAX(pclose_flag::INT) AS pclose_flag_any,
        AVG(fclose_flag::INT) AS fclose_flag_share,
        MAX(fclose_flag::INT) AS fclose_flag_any
    FROM raw_features
    GROUP BY {id_col}
),

-- ============================================================================
-- БЛОК 2 И 3: ПРОСРОЧКИ И УТИЛИЗАЦИЯ (счетчики + серьезные нарушения)
-- ============================================================================
overdue_util_agg AS (
    SELECT
        {id_col},
        
        -- Просрочки: доли
        AVG(is_zero_loans5::INT)    AS loans5_zero_share,
        AVG(is_zero_loans530::INT)  AS loans530_zero_share,
        AVG(is_zero_loans3060::INT) AS loans3060_zero_share,
        AVG(is_zero_loans90::INT)   AS loans90_zero_share,
        
        -- Просрочки: наличие хотя бы одной
        MAX((is_zero_loans5=0)::INT)    AS loans5_has_overdue,
        MAX((is_zero_loans530=0)::INT)  AS loans530_has_overdue,
        MAX((is_zero_loans3060=0)::INT) AS loans3060_has_overdue,
        MAX((is_zero_loans90=0)::INT)   AS loans90_has_overdue,
        
        -- Утилизация
        AVG(is_zero_util::INT)          AS util_zero_share,
        MAX((is_zero_util=0)::INT)      AS util_has_nonzero,
        SUM((is_zero_util=0)::INT)      AS util_nonzero_count,
        
        AVG(is_zero_over2limit::INT)    AS over2limit_zero_share,
        MAX((is_zero_over2limit=0)::INT) AS over2limit_has_nonzero,
        
        AVG(is_zero_maxover2limit::INT) AS maxover2limit_zero_share,
        MAX((is_zero_maxover2limit=0)::INT) AS maxover2limit_has_nonzero,
        
        -- Интегральные просрочки
        MAX(CASE WHEN is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS has_serious_overdue,
        SUM(CASE WHEN is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS count_serious_overdue,
        AVG(CASE WHEN is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS share_serious_overdue,
        
        MAX(CASE WHEN is_zero_loans5=0 OR is_zero_loans530=0 OR is_zero_loans3060=0 OR is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS has_any_overdue,
        SUM(CASE WHEN is_zero_loans5=0 OR is_zero_loans530=0 OR is_zero_loans3060=0 OR is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS count_any_overdue,
        AVG(CASE WHEN is_zero_loans5=0 OR is_zero_loans530=0 OR is_zero_loans3060=0 OR is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) AS share_any_overdue,
        
        -- Отношение серьезных (90+) к легким (до 5 дней)
        SUM(CASE WHEN is_zero_loans6090=0 OR is_zero_loans90=0 THEN 1 ELSE 0 END) * 1.0 
            / NULLIF(SUM((is_zero_loans5=0)::INT), 0) AS serious_to_mild_ratio

    FROM raw_features
    GROUP BY {id_col}
),

-- ============================================================================
-- БЛОК 5: ПЛАТЕЖИ (Внутристрочная логика)
-- Разворачиваем, считаем статистики внутри ОДНОГО кредита (rn), затем агрегируем
-- ============================================================================
paym_long AS (
    SELECT {id_col}, rn,
           CAST(regexp_extract(month_col, '[0-9]+') AS INTEGER) AS paym_month,
           paym_status
    FROM (
        UNPIVOT raw_features
        ON COLUMNS('^enc_paym_[0-9]+$')
        INTO NAME month_col VALUE paym_status
    )
),
-- ДОБАВЛЕННЫЙ CTE: Сначала считаем оконную функцию LAG отдельно
paym_lagged AS (
    SELECT 
        {id_col}, rn, paym_month, paym_status,
        LAG(paym_status) OVER (PARTITION BY {id_col}, rn ORDER BY paym_month) AS prev_paym_status
    FROM paym_long
    WHERE paym_status IS NOT NULL
),
-- ИСПРАВЛЕННЫЙ CTE: Теперь агрегируем уже посчитанную колонку prev_paym_status
paym_row_stats AS (
    SELECT
        {id_col},
        rn,
        COUNT(DISTINCT paym_status) AS paym_row_nunique,
        SUM((paym_status != prev_paym_status)::INT) AS paym_row_change_count,
        FIRST(paym_status ORDER BY paym_month ASC) AS paym_row_first,
        FIRST(paym_status ORDER BY paym_month DESC) AS paym_row_last,
        AVG((paym_status = '0')::INT) AS paym_row_share_code_0,
        AVG((paym_status = '1')::INT) AS paym_row_share_code_1,
        AVG((paym_status = '3')::INT) AS paym_row_share_code_3,
        AVG((paym_status = '4')::INT) AS paym_row_share_code_4
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
-- БЛОКИ 4 И 6: КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ (Global Mode, Rare Bins, Changes)
-- ============================================================================
global_modes AS (
    SELECT
        MODE(pre_since_opened) AS m_pre_since_opened,
        MODE(pre_since_confirmed) AS m_pre_since_confirmed,
        MODE(pre_loans_credit_limit) AS m_pre_loans_credit_limit,
        MODE(pre_loans_next_pay_summ) AS m_pre_loans_next_pay_summ,
        MODE(pre_loans_outstanding) AS m_pre_loans_outstanding,
        MODE(pre_loans_max_overdue_sum) AS m_pre_loans_max_overdue_sum,
        MODE(pre_loans_credit_cost_rate) AS m_pre_loans_credit_cost_rate,
        MODE(pre_util) AS m_pre_util,
        MODE(pre_over2limit) AS m_pre_over2limit,
        MODE(pre_maxover2limit) AS m_pre_maxover2limit,
        MODE(pre_loans5) AS m_pre_loans5,
        MODE(pre_loans530) AS m_pre_loans530,
        MODE(pre_loans3060) AS m_pre_loans3060,
        MODE(pre_loans90) AS m_pre_loans90,
        MODE(enc_loans_account_holder_type) AS m_enc_loans_account_holder_type,
        MODE(enc_loans_credit_status) AS m_enc_loans_credit_status,
        MODE(enc_loans_credit_type) AS m_enc_loans_credit_type,
        (SELECT COUNT(*) FROM raw_features) AS total_rows
    FROM raw_features
),
cat_features_prep AS (
    SELECT
        r.{id_col}, r.rn,
        
        -- Макрос для каждой колонки: сырое значение, флаг смены, флаг моды, флаг редкого бина (<5%)
        r.pre_since_opened,
        (r.pre_since_opened != LAG(r.pre_since_opened) OVER(PARTITION BY r.{id_col} ORDER BY r.rn))::INT AS pre_since_opened_chg,
        (r.pre_since_opened = gm.m_pre_since_opened)::INT AS pre_since_opened_is_mode,
        ((COUNT(*) OVER(PARTITION BY r.pre_since_opened) * 1.0 / gm.total_rows) < 0.05)::INT AS pre_since_opened_is_rare,

        r.pre_since_confirmed,
        (r.pre_since_confirmed != LAG(r.pre_since_confirmed) OVER(PARTITION BY r.{id_col} ORDER BY r.rn))::INT AS pre_since_confirmed_chg,
        (r.pre_since_confirmed = gm.m_pre_since_confirmed)::INT AS pre_since_confirmed_is_mode,

        r.pre_loans_credit_limit,
        (r.pre_loans_credit_limit != LAG(r.pre_loans_credit_limit) OVER(PARTITION BY r.{id_col} ORDER BY r.rn))::INT AS pre_loans_credit_limit_chg,
        (r.pre_loans_credit_limit = gm.m_pre_loans_credit_limit)::INT AS pre_loans_credit_limit_is_mode,
        ((COUNT(*) OVER(PARTITION BY r.pre_loans_credit_limit) * 1.0 / gm.total_rows) < 0.05)::INT AS pre_loans_credit_limit_is_rare,

        r.pre_loans_outstanding,
        (r.pre_loans_outstanding != LAG(r.pre_loans_outstanding) OVER(PARTITION BY r.{id_col} ORDER BY r.rn))::INT AS pre_loans_outstanding_chg,
        (r.pre_loans_outstanding = gm.m_pre_loans_outstanding)::INT AS pre_loans_outstanding_is_mode,
        ((COUNT(*) OVER(PARTITION BY r.pre_loans_outstanding) * 1.0 / gm.total_rows) < 0.05)::INT AS pre_loans_outstanding_is_rare,

        r.pre_loans_credit_cost_rate,
        (r.pre_loans_credit_cost_rate != LAG(r.pre_loans_credit_cost_rate) OVER(PARTITION BY r.{id_col} ORDER BY r.rn))::INT AS pre_loans_credit_cost_rate_chg,
        (r.pre_loans_credit_cost_rate = gm.m_pre_loans_credit_cost_rate)::INT AS pre_loans_credit_cost_rate_is_mode,
        ((COUNT(*) OVER(PARTITION BY r.pre_loans_credit_cost_rate) * 1.0 / gm.total_rows) < 0.05)::INT AS pre_loans_credit_cost_rate_is_rare,

        r.pre_util,
        (r.pre_util != LAG(r.pre_util) OVER(PARTITION BY r.{id_col} ORDER BY r.rn))::INT AS pre_util_chg,
        (r.pre_util = gm.m_pre_util)::INT AS pre_util_is_mode,
        ((COUNT(*) OVER(PARTITION BY r.pre_util) * 1.0 / gm.total_rows) < 0.05)::INT AS pre_util_is_rare,

        r.pre_loans530,
        (r.pre_loans530 != LAG(r.pre_loans530) OVER(PARTITION BY r.{id_col} ORDER BY r.rn))::INT AS pre_loans530_chg,
        (r.pre_loans530 = gm.m_pre_loans530)::INT AS pre_loans530_is_mode,
        ((COUNT(*) OVER(PARTITION BY r.pre_loans530) * 1.0 / gm.total_rows) < 0.05)::INT AS pre_loans530_is_rare,

        r.pre_loans3060,
        (r.pre_loans3060 != LAG(r.pre_loans3060) OVER(PARTITION BY r.{id_col} ORDER BY r.rn))::INT AS pre_loans3060_chg,
        (r.pre_loans3060 = gm.m_pre_loans3060)::INT AS pre_loans3060_is_mode,

        r.pre_loans90,
        (r.pre_loans90 != LAG(r.pre_loans90) OVER(PARTITION BY r.{id_col} ORDER BY r.rn))::INT AS pre_loans90_chg,
        (r.pre_loans90 = gm.m_pre_loans90)::INT AS pre_loans90_is_mode,

        r.enc_loans_credit_type,
        (r.enc_loans_credit_type != LAG(r.enc_loans_credit_type) OVER(PARTITION BY r.{id_col} ORDER BY r.rn))::INT AS enc_loans_credit_type_chg,
        (r.enc_loans_credit_type = gm.m_enc_loans_credit_type)::INT AS enc_loans_credit_type_is_mode,
        ((COUNT(*) OVER(PARTITION BY r.enc_loans_credit_type) * 1.0 / gm.total_rows) < 0.05)::INT AS enc_loans_credit_type_is_rare

    FROM raw_features r
    CROSS JOIN global_modes gm
),
cat_features_agg AS (
    SELECT
        {id_col},
        
        -- pre_since_opened
        COUNT(DISTINCT pre_since_opened) AS pre_since_opened_nunique,
        FIRST(pre_since_opened ORDER BY rn ASC) AS pre_since_opened_first,
        FIRST(pre_since_opened ORDER BY rn DESC) AS pre_since_opened_last,
        SUM(pre_since_opened_chg) AS pre_since_opened_change_count,
        AVG(pre_since_opened_is_mode) AS pre_since_opened_share_global_mode,
        AVG(pre_since_opened_is_rare) AS pre_since_opened_share_rare_bins,

        -- pre_since_confirmed (исключен share_rare_bins)
        COUNT(DISTINCT pre_since_confirmed) AS pre_since_confirmed_nunique,
        FIRST(pre_since_confirmed ORDER BY rn ASC) AS pre_since_confirmed_first,
        FIRST(pre_since_confirmed ORDER BY rn DESC) AS pre_since_confirmed_last,
        AVG(pre_since_confirmed_is_mode) AS pre_since_confirmed_share_global_mode,

        -- pre_loans_credit_limit
        COUNT(DISTINCT pre_loans_credit_limit) AS pre_loans_credit_limit_nunique,
        FIRST(pre_loans_credit_limit ORDER BY rn ASC) AS pre_loans_credit_limit_first,
        FIRST(pre_loans_credit_limit ORDER BY rn DESC) AS pre_loans_credit_limit_last,
        SUM(pre_loans_credit_limit_chg) AS pre_loans_credit_limit_change_count,
        AVG(pre_loans_credit_limit_is_mode) AS pre_loans_credit_limit_share_global_mode,
        AVG(pre_loans_credit_limit_is_rare) AS pre_loans_credit_limit_share_rare_bins,

        -- pre_util
        COUNT(DISTINCT pre_util) AS pre_util_nunique,
        FIRST(pre_util ORDER BY rn ASC) AS pre_util_first,
        FIRST(pre_util ORDER BY rn DESC) AS pre_util_last,
        SUM(pre_util_chg) AS pre_util_change_count,
        AVG(pre_util_is_mode) AS pre_util_share_global_mode,
        AVG(pre_util_is_rare) AS pre_util_share_rare_bins,

        -- pre_loans_outstanding
        COUNT(DISTINCT pre_loans_outstanding) AS pre_loans_outstanding_nunique,
        FIRST(pre_loans_outstanding ORDER BY rn DESC) AS pre_loans_outstanding_last,
        AVG(pre_loans_outstanding_is_mode) AS pre_loans_outstanding_share_global_mode,

        -- pre_loans_credit_cost_rate
        COUNT(DISTINCT pre_loans_credit_cost_rate) AS pre_loans_credit_cost_rate_nunique,
        FIRST(pre_loans_credit_cost_rate ORDER BY rn DESC) AS pre_loans_credit_cost_rate_last,
        AVG(pre_loans_credit_cost_rate_is_mode) AS pre_loans_credit_cost_rate_share_global_mode,

        -- pre_loans530
        COUNT(DISTINCT pre_loans530) AS pre_loans530_nunique,
        MAX(pre_loans530_chg) AS pre_loans530_changed_ever,

        -- pre_loans3060 (исключен first по логике питона)
        COUNT(DISTINCT pre_loans3060) AS pre_loans3060_nunique,
        FIRST(pre_loans3060 ORDER BY rn DESC) AS pre_loans3060_last,
        MAX(pre_loans3060_chg) AS pre_loans3060_changed_ever,

        -- pre_loans90 (оставлен только нужный минимум)
        COUNT(DISTINCT pre_loans90) AS pre_loans90_nunique,
        AVG(pre_loans90_is_mode) AS pre_loans90_share_global_mode,

        -- enc_loans_credit_type
        COUNT(DISTINCT enc_loans_credit_type) AS enc_loans_credit_type_nunique,
        FIRST(enc_loans_credit_type ORDER BY rn ASC) AS enc_loans_credit_type_first,
        FIRST(enc_loans_credit_type ORDER BY rn DESC) AS enc_loans_credit_type_last,
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
        ou.* EXCLUDE ({id_col}),
        pa.* EXCLUDE ({id_col}),
        cf.* EXCLUDE ({id_col})
    FROM base_features bf
    LEFT JOIN overdue_util_agg ou ON bf.{id_col} = ou.{id_col}
    LEFT JOIN paym_agg         pa ON bf.{id_col} = pa.{id_col}
    LEFT JOIN cat_features_agg cf ON bf.{id_col} = cf.{id_col}
)

SELECT
    f.*,
    t.* EXCLUDE ({id_col})
FROM aggregated_features f
LEFT JOIN {target_source} t
  ON f.{id_col} = t.{id_col}