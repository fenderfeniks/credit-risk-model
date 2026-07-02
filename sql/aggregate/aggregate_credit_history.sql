-- ============================================================================
-- АГРЕГАЦИЯ КРЕДИТНОЙ ИСТОРИИ: N записей (rn) на 1 id -> 1 строка на id
-- Диалект: DuckDB (EXCLUDE, UNPIVOT ... ON COLUMNS(regex), QUALIFY, LIST/STRING_AGG)
--
-- ВАЖНОЕ ДОПУЩЕНИЕ, УЧТЕННОЕ ВЕЗДЕ В ЗАПРОСЕ:
--   Номера бинов НЕ ординальны -- они присваиваются случайно при биннинге
--   (см. описание поля: "каждому промежутку случайным образом назначается
--   уникальный номер от 0 до N-1"). Поэтому:
--     - НИГДЕ не используются AVG/MAX/MIN/сравнения "больше/меньше" по номеру бина
--     - Единственное, что имеет смысл: (а) совпадает ли бин с "типичным" значением
--       (модой), (б) менялся ли бин вообще внутри истории клиента,
--       (в) разнообразие/энтропия значений.
--
-- ЛОГИКА ПО ГРУППАМ ПРИЗНАКОВ:
--   A. Служебные счетчики (сколько кредитов, глубина истории)
--   B. Снэпшот первого и последнего кредита -- сырые бины как категории
--   C. Булевы флаги -- доля True / ever True
--   D. НЕ-ДОМИНАНТНЫЙ БИН для ВСЕХ бинированных признаков, в 2 вариантах:
--        - global: бин клиента отличается от моды по ВСЕЙ выборке
--        - local:  бин клиента отличается от ЕГО СОБСТВЕННОЙ моды (своя норма)
--      Считаем count + share для каждого варианта
--   E. РАЗНООБРАЗИЕ ЗНАЧЕНИЙ (distinct-count) -- применяем ко ВСЕМ бинированным
--      и категориальным колонкам, не только к enc_loans_credit_type
--   F. "МЕНЯЛСЯ ЛИ БИН ВООБЩЕ" -- булев флаг + строковая история бинов
--      по каждой отслеживаемой фиче (например pre_loans_credit_limit: "54321")
--      + флаг "первый бин == последний бин"
--   G. enc_paym_* -- временной ряд статусов платежей.
--      Бины НЕ ординальны, поэтому вместо MAX/AVG считаем:
--        - энтропию распределения бинов по месяцам (кроме 11 и 24)
--        - долю месяцев (кроме 11 и 24), где бин НЕ входит в (0, 3)
--          (0 и 3 -- твои наблюдаемые "нормальные" бины на большинстве месяцев)
--        - булево: были ли месяцы (кроме 11, 24) с попаданием в бин 2
--        - для месяцев 11 и 24 отдельно: НЕ входит ли бин в (4, 1) (там
--          доминантные 4 и 1, худший -- 3)
-- ============================================================================

WITH raw_features AS (
    SELECT * FROM {features_source}
    WHERE {id_col} IS NOT NULL
),

-- ============================================================================
-- A. СЧЕТЧИКИ ПО ID
-- ============================================================================
loan_counts AS (
    SELECT
        {id_col},
        COUNT(*) AS loans_total_count,
        MAX(rn)  AS loans_max_rn
    FROM raw_features
    GROUP BY {id_col}
),

-- ============================================================================
-- B. СНЭПШОТЫ ПЕРВОГО И ПОСЛЕДНЕГО КРЕДИТА
--    Сырые бины оставляем как есть -- они пойдут в модель как категории
-- ============================================================================
last_loan AS (
    SELECT
        {id_col},
        pre_since_opened               AS last_pre_since_opened,
        pre_since_confirmed            AS last_pre_since_confirmed,
        pre_pterm                      AS last_pre_pterm,
        pre_fterm                      AS last_pre_fterm,
        pre_till_pclose                AS last_pre_till_pclose,
        pre_till_fclose                AS last_pre_till_fclose,
        pre_loans_credit_limit         AS last_pre_loans_credit_limit,
        pre_loans_credit_cost_rate     AS last_pre_loans_credit_cost_rate,
        enc_loans_account_holder_type  AS last_enc_loans_account_holder_type,
        enc_loans_credit_status        AS last_enc_loans_credit_status,
        enc_loans_account_cur          AS last_enc_loans_account_cur,
        enc_loans_credit_type          AS last_enc_loans_credit_type,
        pclose_flag                    AS last_pclose_flag,
        fclose_flag                    AS last_fclose_flag
    FROM raw_features
    QUALIFY ROW_NUMBER() OVER (PARTITION BY {id_col} ORDER BY rn DESC) = 1
),
first_loan AS (
    SELECT
        {id_col},
        pre_loans_credit_limit AS first_pre_loans_credit_limit,
        enc_loans_credit_type  AS first_enc_loans_credit_type,
        enc_loans_credit_status AS first_enc_loans_credit_status
    FROM raw_features
    QUALIFY ROW_NUMBER() OVER (PARTITION BY {id_col} ORDER BY rn ASC) = 1
),

-- ============================================================================
-- C. БУЛЕВЫ ФЛАГИ -- доля True и "хоть раз True" по всей истории кредитов
-- ============================================================================
flags_agg AS (
    SELECT
        {id_col},
        AVG(is_zero_loans5::INT)        AS is_zero_loans5_share,
        AVG(is_zero_loans530::INT)      AS is_zero_loans530_share,
        AVG(is_zero_loans3060::INT)     AS is_zero_loans3060_share,
        MAX((NOT is_zero_loans3060)::INT) AS ever_had_loans3060,
        AVG(is_zero_loans6090::INT)     AS is_zero_loans6090_share,
        MAX((NOT is_zero_loans6090)::INT) AS ever_had_loans6090,
        AVG(is_zero_loans90::INT)       AS is_zero_loans90_share,
        MAX((NOT is_zero_loans90)::INT)   AS ever_had_loans90,
        AVG(is_zero_util::INT)          AS is_zero_util_share,
        AVG(is_zero_over2limit::INT)    AS is_zero_over2limit_share,
        AVG(is_zero_maxover2limit::INT) AS is_zero_maxover2limit_share,
        AVG(pclose_flag::INT)           AS pclose_flag_share,
        AVG(fclose_flag::INT)           AS fclose_flag_share
    FROM raw_features
    GROUP BY {id_col}
),

-- ============================================================================
-- D. GLOBAL MODE ПО ВСЕЙ ВЫБОРКЕ (одно число на колонку, для всех клиентов)
--    Список колонок = ВСЕ бинированные признаки (расширено по твоей просьбе)
-- ============================================================================
global_modes AS (
    SELECT
        MODE(pre_since_opened)           AS m_pre_since_opened,
        MODE(pre_since_confirmed)        AS m_pre_since_confirmed,
        MODE(pre_pterm)                  AS m_pre_pterm,
        MODE(pre_fterm)                  AS m_pre_fterm,
        MODE(pre_till_pclose)            AS m_pre_till_pclose,
        MODE(pre_till_fclose)            AS m_pre_till_fclose,
        MODE(pre_loans_credit_limit)     AS m_pre_loans_credit_limit,
        MODE(pre_loans_next_pay_summ)    AS m_pre_loans_next_pay_summ,
        MODE(pre_loans_outstanding)      AS m_pre_loans_outstanding,
        MODE(pre_loans_total_overdue)    AS m_pre_loans_total_overdue,
        MODE(pre_loans_max_overdue_sum)  AS m_pre_loans_max_overdue_sum,
        MODE(pre_loans_credit_cost_rate) AS m_pre_loans_credit_cost_rate,
        MODE(pre_loans5)                 AS m_pre_loans5,
        MODE(pre_loans530)               AS m_pre_loans530,
        MODE(pre_loans3060)              AS m_pre_loans3060,
        MODE(pre_loans6090)              AS m_pre_loans6090,
        MODE(pre_loans90)                AS m_pre_loans90,
        MODE(pre_util)                   AS m_pre_util,
        MODE(pre_over2limit)             AS m_pre_over2limit,
        MODE(pre_maxover2limit)          AS m_pre_maxover2limit,
        MODE(enc_loans_account_holder_type) AS m_enc_loans_account_holder_type,
        MODE(enc_loans_credit_status)       AS m_enc_loans_credit_status,
        MODE(enc_loans_account_cur)         AS m_enc_loans_account_cur,
        MODE(enc_loans_credit_type)         AS m_enc_loans_credit_type
    FROM raw_features
),

-- ============================================================================
-- D2. LOCAL MODE (своя, для каждого id -- типичное значение самого клиента)
-- ============================================================================
local_modes AS (
    SELECT
        {id_col},
        MODE(pre_since_opened)           AS lm_pre_since_opened,
        MODE(pre_since_confirmed)        AS lm_pre_since_confirmed,
        MODE(pre_pterm)                  AS lm_pre_pterm,
        MODE(pre_fterm)                  AS lm_pre_fterm,
        MODE(pre_till_pclose)            AS lm_pre_till_pclose,
        MODE(pre_till_fclose)            AS lm_pre_till_fclose,
        MODE(pre_loans_credit_limit)     AS lm_pre_loans_credit_limit,
        MODE(pre_loans_next_pay_summ)    AS lm_pre_loans_next_pay_summ,
        MODE(pre_loans_outstanding)      AS lm_pre_loans_outstanding,
        MODE(pre_loans_total_overdue)    AS lm_pre_loans_total_overdue,
        MODE(pre_loans_max_overdue_sum)  AS lm_pre_loans_max_overdue_sum,
        MODE(pre_loans_credit_cost_rate) AS lm_pre_loans_credit_cost_rate,
        MODE(pre_loans5)                 AS lm_pre_loans5,
        MODE(pre_loans530)               AS lm_pre_loans530,
        MODE(pre_loans3060)              AS lm_pre_loans3060,
        MODE(pre_loans6090)              AS lm_pre_loans6090,
        MODE(pre_loans90)                AS lm_pre_loans90,
        MODE(pre_util)                   AS lm_pre_util,
        MODE(pre_over2limit)             AS lm_pre_over2limit,
        MODE(pre_maxover2limit)          AS lm_pre_maxover2limit,
        MODE(enc_loans_account_holder_type) AS lm_enc_loans_account_holder_type,
        MODE(enc_loans_credit_status)       AS lm_enc_loans_credit_status,
        MODE(enc_loans_account_cur)         AS lm_enc_loans_account_cur,
        MODE(enc_loans_credit_type)         AS lm_enc_loans_credit_type
    FROM raw_features
    GROUP BY {id_col}
),

-- ============================================================================
-- D3. GLOBAL non-dominant: count/share кредитов клиента, где бин != глобальной моде
--     (кросс джойн с global_modes -- это одна строка, не размножает данные)
-- ============================================================================
global_non_dominant_agg AS (
    SELECT
        r.{id_col},

        SUM((r.pre_since_opened <> gm.m_pre_since_opened)::INT)           AS pre_since_opened_gnd_count,
        AVG((r.pre_since_opened <> gm.m_pre_since_opened)::INT)           AS pre_since_opened_gnd_share,

        SUM((r.pre_since_confirmed <> gm.m_pre_since_confirmed)::INT)     AS pre_since_confirmed_gnd_count,
        AVG((r.pre_since_confirmed <> gm.m_pre_since_confirmed)::INT)     AS pre_since_confirmed_gnd_share,

        SUM((r.pre_pterm <> gm.m_pre_pterm)::INT)                        AS pre_pterm_gnd_count,
        AVG((r.pre_pterm <> gm.m_pre_pterm)::INT)                        AS pre_pterm_gnd_share,

        SUM((r.pre_fterm <> gm.m_pre_fterm)::INT)                        AS pre_fterm_gnd_count,
        AVG((r.pre_fterm <> gm.m_pre_fterm)::INT)                        AS pre_fterm_gnd_share,

        SUM((r.pre_till_pclose <> gm.m_pre_till_pclose)::INT)            AS pre_till_pclose_gnd_count,
        AVG((r.pre_till_pclose <> gm.m_pre_till_pclose)::INT)            AS pre_till_pclose_gnd_share,

        SUM((r.pre_till_fclose <> gm.m_pre_till_fclose)::INT)            AS pre_till_fclose_gnd_count,
        AVG((r.pre_till_fclose <> gm.m_pre_till_fclose)::INT)            AS pre_till_fclose_gnd_share,

        SUM((r.pre_loans_credit_limit <> gm.m_pre_loans_credit_limit)::INT) AS pre_loans_credit_limit_gnd_count,
        AVG((r.pre_loans_credit_limit <> gm.m_pre_loans_credit_limit)::INT) AS pre_loans_credit_limit_gnd_share,

        SUM((r.pre_loans_next_pay_summ <> gm.m_pre_loans_next_pay_summ)::INT) AS pre_loans_next_pay_summ_gnd_count,
        AVG((r.pre_loans_next_pay_summ <> gm.m_pre_loans_next_pay_summ)::INT) AS pre_loans_next_pay_summ_gnd_share,

        SUM((r.pre_loans_outstanding <> gm.m_pre_loans_outstanding)::INT) AS pre_loans_outstanding_gnd_count,
        AVG((r.pre_loans_outstanding <> gm.m_pre_loans_outstanding)::INT) AS pre_loans_outstanding_gnd_share,

        SUM((r.pre_loans_total_overdue <> gm.m_pre_loans_total_overdue)::INT) AS pre_loans_total_overdue_gnd_count,
        AVG((r.pre_loans_total_overdue <> gm.m_pre_loans_total_overdue)::INT) AS pre_loans_total_overdue_gnd_share,

        SUM((r.pre_loans_max_overdue_sum <> gm.m_pre_loans_max_overdue_sum)::INT) AS pre_loans_max_overdue_sum_gnd_count,
        AVG((r.pre_loans_max_overdue_sum <> gm.m_pre_loans_max_overdue_sum)::INT) AS pre_loans_max_overdue_sum_gnd_share,

        SUM((r.pre_loans_credit_cost_rate <> gm.m_pre_loans_credit_cost_rate)::INT) AS pre_loans_credit_cost_rate_gnd_count,
        AVG((r.pre_loans_credit_cost_rate <> gm.m_pre_loans_credit_cost_rate)::INT) AS pre_loans_credit_cost_rate_gnd_share,

        SUM((r.pre_loans5 <> gm.m_pre_loans5)::INT)                      AS pre_loans5_gnd_count,
        AVG((r.pre_loans5 <> gm.m_pre_loans5)::INT)                      AS pre_loans5_gnd_share,

        SUM((r.pre_loans530 <> gm.m_pre_loans530)::INT)                  AS pre_loans530_gnd_count,
        AVG((r.pre_loans530 <> gm.m_pre_loans530)::INT)                  AS pre_loans530_gnd_share,

        SUM((r.pre_loans3060 <> gm.m_pre_loans3060)::INT)                AS pre_loans3060_gnd_count,
        AVG((r.pre_loans3060 <> gm.m_pre_loans3060)::INT)                AS pre_loans3060_gnd_share,

        SUM((r.pre_loans6090 <> gm.m_pre_loans6090)::INT)                AS pre_loans6090_gnd_count,
        AVG((r.pre_loans6090 <> gm.m_pre_loans6090)::INT)                AS pre_loans6090_gnd_share,

        SUM((r.pre_loans90 <> gm.m_pre_loans90)::INT)                    AS pre_loans90_gnd_count,
        AVG((r.pre_loans90 <> gm.m_pre_loans90)::INT)                    AS pre_loans90_gnd_share,

        SUM((r.pre_util <> gm.m_pre_util)::INT)                          AS pre_util_gnd_count,
        AVG((r.pre_util <> gm.m_pre_util)::INT)                          AS pre_util_gnd_share,

        SUM((r.pre_over2limit <> gm.m_pre_over2limit)::INT)              AS pre_over2limit_gnd_count,
        AVG((r.pre_over2limit <> gm.m_pre_over2limit)::INT)              AS pre_over2limit_gnd_share,

        SUM((r.pre_maxover2limit <> gm.m_pre_maxover2limit)::INT)        AS pre_maxover2limit_gnd_count,
        AVG((r.pre_maxover2limit <> gm.m_pre_maxover2limit)::INT)        AS pre_maxover2limit_gnd_share,

        SUM((r.enc_loans_account_holder_type <> gm.m_enc_loans_account_holder_type)::INT) AS enc_loans_account_holder_type_gnd_count,
        AVG((r.enc_loans_account_holder_type <> gm.m_enc_loans_account_holder_type)::INT) AS enc_loans_account_holder_type_gnd_share,

        SUM((r.enc_loans_credit_status <> gm.m_enc_loans_credit_status)::INT) AS enc_loans_credit_status_gnd_count,
        AVG((r.enc_loans_credit_status <> gm.m_enc_loans_credit_status)::INT) AS enc_loans_credit_status_gnd_share,

        SUM((r.enc_loans_account_cur <> gm.m_enc_loans_account_cur)::INT) AS enc_loans_account_cur_gnd_count,
        AVG((r.enc_loans_account_cur <> gm.m_enc_loans_account_cur)::INT) AS enc_loans_account_cur_gnd_share,

        SUM((r.enc_loans_credit_type <> gm.m_enc_loans_credit_type)::INT) AS enc_loans_credit_type_gnd_count,
        AVG((r.enc_loans_credit_type <> gm.m_enc_loans_credit_type)::INT) AS enc_loans_credit_type_gnd_share

    FROM raw_features r
    CROSS JOIN global_modes gm
    GROUP BY r.{id_col}
),

-- ============================================================================
-- D4. LOCAL non-dominant: count/share кредитов, где бин != СВОЕЙ ЖЕ моды клиента
--     (насколько сильно конкретный кредит выбивается из типичного профиля этого клиента)
-- ============================================================================
local_non_dominant_agg AS (
    SELECT
        r.{id_col},

        SUM((r.pre_loans_credit_limit <> lmm.lm_pre_loans_credit_limit)::INT) AS pre_loans_credit_limit_lnd_count,
        AVG((r.pre_loans_credit_limit <> lmm.lm_pre_loans_credit_limit)::INT) AS pre_loans_credit_limit_lnd_share,

        SUM((r.pre_loans5 <> lmm.lm_pre_loans5)::INT)         AS pre_loans5_lnd_count,
        AVG((r.pre_loans5 <> lmm.lm_pre_loans5)::INT)         AS pre_loans5_lnd_share,

        SUM((r.pre_loans3060 <> lmm.lm_pre_loans3060)::INT)   AS pre_loans3060_lnd_count,
        AVG((r.pre_loans3060 <> lmm.lm_pre_loans3060)::INT)   AS pre_loans3060_lnd_share,

        SUM((r.pre_loans6090 <> lmm.lm_pre_loans6090)::INT)   AS pre_loans6090_lnd_count,
        AVG((r.pre_loans6090 <> lmm.lm_pre_loans6090)::INT)   AS pre_loans6090_lnd_share,

        SUM((r.pre_loans90 <> lmm.lm_pre_loans90)::INT)       AS pre_loans90_lnd_count,
        AVG((r.pre_loans90 <> lmm.lm_pre_loans90)::INT)       AS pre_loans90_lnd_share,

        SUM((r.pre_over2limit <> lmm.lm_pre_over2limit)::INT) AS pre_over2limit_lnd_count,
        AVG((r.pre_over2limit <> lmm.lm_pre_over2limit)::INT) AS pre_over2limit_lnd_share,

        SUM((r.pre_maxover2limit <> lmm.lm_pre_maxover2limit)::INT) AS pre_maxover2limit_lnd_count,
        AVG((r.pre_maxover2limit <> lmm.lm_pre_maxover2limit)::INT) AS pre_maxover2limit_lnd_share,

        SUM((r.enc_loans_credit_type <> lmm.lm_enc_loans_credit_type)::INT) AS enc_loans_credit_type_lnd_count,
        AVG((r.enc_loans_credit_type <> lmm.lm_enc_loans_credit_type)::INT) AS enc_loans_credit_type_lnd_share

    FROM raw_features r
    JOIN local_modes lmm ON r.{id_col} = lmm.{id_col}
    GROUP BY r.{id_col}
),

-- ============================================================================
-- E. РАЗНООБРАЗИЕ ЗНАЧЕНИЙ (distinct count) -- по всем ключевым бинированным
--    и категориальным признакам, не только credit_type
-- ============================================================================
distinct_counts_agg AS (
    SELECT
        {id_col},
        COUNT(DISTINCT pre_loans_credit_limit)     AS distinct_pre_loans_credit_limit_count,
        COUNT(DISTINCT pre_loans_credit_cost_rate) AS distinct_pre_loans_credit_cost_rate_count,
        COUNT(DISTINCT pre_loans5)                 AS distinct_pre_loans5_count,
        COUNT(DISTINCT pre_loans530)               AS distinct_pre_loans530_count,
        COUNT(DISTINCT pre_loans3060)              AS distinct_pre_loans3060_count,
        COUNT(DISTINCT pre_loans6090)              AS distinct_pre_loans6090_count,
        COUNT(DISTINCT pre_loans90)                AS distinct_pre_loans90_count,
        COUNT(DISTINCT pre_util)                   AS distinct_pre_util_count,
        COUNT(DISTINCT pre_over2limit)             AS distinct_pre_over2limit_count,
        COUNT(DISTINCT pre_maxover2limit)          AS distinct_pre_maxover2limit_count,
        COUNT(DISTINCT enc_loans_credit_type)      AS distinct_enc_loans_credit_type_count,
        COUNT(DISTINCT enc_loans_credit_status)    AS distinct_enc_loans_credit_status_count,
        COUNT(DISTINCT enc_loans_account_holder_type) AS distinct_enc_loans_account_holder_type_count
    FROM raw_features
    GROUP BY {id_col}
),

-- ============================================================================
-- F. "МЕНЯЛСЯ ЛИ БИН ВООБЩЕ" + строковая история бинов по rn + first==last
--    STRING_AGG упорядочен по rn -- получаем строку типа "54321" по истории
-- ============================================================================
change_history_agg AS (
    SELECT
        {id_col},

        STRING_AGG(pre_loans_credit_limit::VARCHAR, '' ORDER BY rn) AS pre_loans_credit_limit_bin_sequence,
        (COUNT(DISTINCT pre_loans_credit_limit) > 1)                AS pre_loans_credit_limit_ever_changed,
        (FIRST(pre_loans_credit_limit ORDER BY rn ASC)
            = FIRST(pre_loans_credit_limit ORDER BY rn DESC))        AS pre_loans_credit_limit_first_eq_last,

        STRING_AGG(pre_loans5::VARCHAR, '' ORDER BY rn)             AS pre_loans5_bin_sequence,
        (COUNT(DISTINCT pre_loans5) > 1)                             AS pre_loans5_ever_changed,
        (FIRST(pre_loans5 ORDER BY rn ASC) = FIRST(pre_loans5 ORDER BY rn DESC)) AS pre_loans5_first_eq_last,

        STRING_AGG(pre_loans3060::VARCHAR, '' ORDER BY rn)          AS pre_loans3060_bin_sequence,
        (COUNT(DISTINCT pre_loans3060) > 1)                          AS pre_loans3060_ever_changed,

        STRING_AGG(pre_loans90::VARCHAR, '' ORDER BY rn)             AS pre_loans90_bin_sequence,
        (COUNT(DISTINCT pre_loans90) > 1)                            AS pre_loans90_ever_changed,

        STRING_AGG(enc_loans_credit_type::VARCHAR, '' ORDER BY rn)  AS enc_loans_credit_type_bin_sequence,
        (COUNT(DISTINCT enc_loans_credit_type) > 1)                  AS enc_loans_credit_type_ever_changed,
        (FIRST(enc_loans_credit_type ORDER BY rn ASC)
            = FIRST(enc_loans_credit_type ORDER BY rn DESC))         AS enc_loans_credit_type_first_eq_last,

        STRING_AGG(enc_loans_credit_status::VARCHAR, '' ORDER BY rn) AS enc_loans_credit_status_bin_sequence,
        (COUNT(DISTINCT enc_loans_credit_status) > 1)                 AS enc_loans_credit_status_ever_changed

    FROM raw_features
    GROUP BY {id_col}
),

-- ============================================================================
-- G. enc_paym_* -- НЕ ординальные бины. Разворачиваем в long-формат,
--    выделяем месяцы 11 и 24 отдельно (там своя, отличная от других логика).
-- ============================================================================
paym_long AS (
    UNPIVOT raw_features
    ON COLUMNS('^enc_paym_[0-9]+$')
    INTO
        NAME paym_month_col
        VALUE paym_status
),
paym_long_typed AS (
    SELECT
        {id_col},
        rn,
        -- вытаскиваем номер месяца из имени колонки enc_paym_N -> N
        CAST(regexp_extract(paym_month_col, '[0-9]+') AS INTEGER) AS paym_month,
        paym_status
    FROM paym_long
),

-- обычные месяцы (все, кроме 11 и 24): доминантные бины 0 и 3 по твоему наблюдению
paym_regular_agg AS (
    SELECT
        {id_col},
        -- ИСПРАВЛЕНО: Убрали деление на константу entropy_n.
        -- Оставляем просто -entropy_sum.
        -entropy_sum AS paym_regular_entropy,

        SUM(CASE WHEN paym_status NOT IN ('0', '3') THEN 1 ELSE 0 END) AS paym_regular_outside_0_3_count,
        AVG((paym_status NOT IN ('0', '3'))::INT)                       AS paym_regular_outside_0_3_share,

        MAX((paym_status = '2')::INT) AS paym_regular_ever_bin2

    FROM (
        SELECT
            {id_col},
            paym_status,
            SUM(-p * LN(p)) OVER (PARTITION BY {id_col}) AS entropy_sum
            -- ИСПРАВЛЕНО: Убрали ненужную константу 1 AS entropy_n
        FROM (
            SELECT
                {id_col},
                paym_status,
                COUNT(*) OVER (PARTITION BY {id_col}, paym_status) * 1.0
                    / COUNT(*) OVER (PARTITION BY {id_col}) AS p
            FROM paym_long_typed
            WHERE paym_month NOT IN (11, 24)
        )
    )
    GROUP BY {id_col}, entropy_sum
),

-- особые месяцы 11 и 24: доминантные бины 4 и 1, худший -- 3
paym_special_agg AS (
    SELECT
        {id_col},
        SUM(CASE WHEN paym_month IN (11, 24) AND paym_status NOT IN ('4', '1') THEN 1 ELSE 0 END) AS paym_special_outside_4_1_count,
        MAX(CASE WHEN paym_month IN (11, 24) AND paym_status = '3' THEN 1 ELSE 0 END)              AS paym_special_ever_worst_bin3
    FROM paym_long_typed
    GROUP BY {id_col}
),

-- ============================================================================
-- СБОРКА ВСЕХ АГРЕГАТОВ В ОДНУ СТРОКУ НА id
-- ============================================================================
aggregated_features AS (
    SELECT
        lc.*,
        ll.*   EXCLUDE ({id_col}),
        fl.*   EXCLUDE ({id_col}),
        fa.*   EXCLUDE ({id_col}),
        gnd.*  EXCLUDE ({id_col}),
        lnd.*  EXCLUDE ({id_col}),
        dc.*   EXCLUDE ({id_col}),
        ch.*   EXCLUDE ({id_col}),
        pr.*   EXCLUDE ({id_col}),
        ps.*   EXCLUDE ({id_col})
    FROM loan_counts lc
    LEFT JOIN last_loan               ll  ON lc.{id_col} = ll.{id_col}
    LEFT JOIN first_loan              fl  ON lc.{id_col} = fl.{id_col}
    LEFT JOIN flags_agg               fa  ON lc.{id_col} = fa.{id_col}
    LEFT JOIN global_non_dominant_agg gnd ON lc.{id_col} = gnd.{id_col}
    LEFT JOIN local_non_dominant_agg  lnd ON lc.{id_col} = lnd.{id_col}
    LEFT JOIN distinct_counts_agg     dc  ON lc.{id_col} = dc.{id_col}
    LEFT JOIN change_history_agg      ch  ON lc.{id_col} = ch.{id_col}
    LEFT JOIN paym_regular_agg        pr  ON lc.{id_col} = pr.{id_col}
    LEFT JOIN paym_special_agg        ps  ON lc.{id_col} = ps.{id_col}
)

SELECT
    f.*,
    t.* EXCLUDE ({id_col})
FROM aggregated_features f
LEFT JOIN {target_source} t
  ON f.{id_col} = t.{id_col}