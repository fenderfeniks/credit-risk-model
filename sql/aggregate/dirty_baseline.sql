WITH raw_features AS (
    SELECT * FROM {features_source}
    WHERE {id_col} IS NOT NULL
),
aggregated_features AS (
    SELECT * FROM raw_features
)
SELECT 
    f.*, 
    t.* EXCLUDE ({id_col})
FROM aggregated_features f
LEFT JOIN {target_source} t 
  ON f.{id_col} = t.{id_col}