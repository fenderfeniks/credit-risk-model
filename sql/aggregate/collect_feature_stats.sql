WITH filtered_train_data AS (
    SELECT f.*
    FROM {features_source} f
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
),
value_counts_with_mode AS (
    SELECT
        feature_name,
        feature_value,
        cnt,
        total_cnt,
        max_by(feature_value, cnt) OVER (PARTITION BY feature_name) AS mode_value
    FROM value_counts
)
SELECT
    feature_name,
    max_by(mode_value, 1) AS mode_value,
    list(feature_value) FILTER (
        WHERE cnt * 1.0 / total_cnt < 0.05
        AND feature_value != mode_value
    ) AS rare_bins
FROM value_counts_with_mode
GROUP BY feature_name;