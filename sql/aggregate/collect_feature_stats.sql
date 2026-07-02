WITH filtered_train_data AS (
    SELECT f.*
    FROM {features_source} f
    -- train_ids_df регистрируется в Python прямо перед выполнением
    JOIN train_ids_df t ON f.{id_col} = t.{id_col}
),
unpivoted_features AS (
    SELECT 
        {id_col}, 
        feature_name, 
        CAST(feature_value AS INTEGER) AS feature_value
    FROM filtered_train_data
    UNPIVOT (
        feature_value FOR feature_name IN (
            {cols_for_unpivot}
        )
    )
    WHERE feature_value IS NOT NULL
),
value_counts AS (
    SELECT
        feature_name,
        feature_value,
        COUNT(*) AS cnt,
        SUM(COUNT(*)) OVER (PARTITION BY feature_name) AS total_cnt
    FROM unpivoted_features
    GROUP BY feature_name, feature_value
)
SELECT
    feature_name,
    max_by(feature_value, cnt) AS mode_value,
    list(feature_value) FILTER (WHERE cnt * 1.0 / total_cnt < 0.05) AS rare_bins
FROM value_counts
GROUP BY feature_name;