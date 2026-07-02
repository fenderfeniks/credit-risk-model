import json
import logging
import pandas as pd
import duckdb
from src.core.splitting import split_data

logger = logging.getLogger(__name__)


class GlobalStatCompiler:
    def __init__(self, cfg, project_root):
        self.cfg = cfg
        self.agg_ver = cfg.data.tabular.aggregation_version
        
        self.artifact_path = project_root / cfg.paths.models_dir / f"global_stats_v{self.agg_ver}.json"
        self.sql_path = project_root / cfg.paths.sql_aggregate_dir / cfg.paths.sql_col_stats_file
        
        self.stats_dict = {}
        self.cat_cols = cfg.data.tabular.aggregation_stats_cat_cols

    def fit_and_save(self, features_source: str, train_ids: set, id_col: str):
        if not self.sql_path.exists():
            raise FileNotFoundError(f"Файл статистики не найден: {self.sql_path}")
            
        logger.info(f"Сбор статистик по {len(train_ids)} train_ids...")

        # 1. Читаем SQL-шаблон с диска
        with open(self.sql_path, "r", encoding="utf-8") as f:
            sql_template = f.read()

        # 2. Формируем строку колонок: pre_util, pre_loans_credit_limit, ...
        cols_for_unpivot = ",\n            ".join(self.cat_cols)
        
        # 3. Подставляем переменные
        query = sql_template.format(
            features_source=features_source,
            id_col=id_col,
            cols_for_unpivot=cols_for_unpivot
        )

        # 4. Передаем ID в DuckDB и выполняем
        train_ids_df = pd.DataFrame({id_col: list(train_ids)})
        conn = duckdb.connect()
        conn.register('train_ids_df', train_ids_df)
        df_stats = conn.execute(query).df()
        conn.close()

        # 5. Парсим результат в JSON
        stats = {}
        for _, row in df_stats.iterrows():
            col = row['feature_name']
            mode_val = row['mode_value']
            
            # Безопасное извлечение редких бинов (защита от numpy arrays и NaN)
            rare_raw = row['rare_bins']
            try:
                if len(rare_raw) > 0:
                    rare_clean = [int(x) for x in rare_raw]
                else:
                    rare_clean = [-999]
            except TypeError:
                # Сработает, если rare_raw это NaN (float) или None
                rare_clean = [-999]

            stats[col] = {
                'mode': int(mode_val) if pd.notnull(mode_val) else -999,
                'rare': rare_clean
            }

        self.stats_dict = stats
        
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.artifact_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats_dict, f, indent=4)
        logger.info(f"Статистика сохранена: {self.artifact_path}")

    def load(self):
        with open(self.artifact_path, 'r', encoding='utf-8') as f:
            self.stats_dict = json.load(f)

    def get_sql_format_kwargs(self) -> dict:
        sql_kwargs = {}
        for col, stat in self.stats_dict.items():
            sql_kwargs[f"m_{col}"] = stat['mode']
            sql_kwargs[f"rare_{col}"] = ", ".join(map(str, stat['rare']))
        return sql_kwargs
    

def get_train_ids_fast(cfg, project_root):
    """Сверхбыстрое получение уникальных ID из сырых паркетов."""
    features_glob = (project_root / cfg.paths.raw_dir / cfg.paths.dev_data_file).as_posix()
    target_path = (project_root / cfg.paths.raw_dir / cfg.paths.target_file_name).as_posix()
    id_col = cfg.data.tabular.get('id_col', 'id')
    target_col = cfg.data.tabular.get('target_col', 'target')
    
    # Легкий запрос: берем только нужные две колонки, без агрегации фичей
    query = f"""
        SELECT f.{id_col}, LAST(t.{target_col}) as {target_col}
        FROM read_parquet('{features_glob}') f
        LEFT JOIN read_csv_auto('{target_path}') t ON f.{id_col} = t.{id_col}
        GROUP BY f.{id_col}
    """
    df_ids_targets = duckdb.query(query).to_df()
    
    # Передаем этот легкий датафрейм в ВАШ сплиттер (из splitting.py)
    # Он поделит данные на основе таргета и вернет train_df
    train_df, _, _ = split_data(cfg, df_ids_targets)
    
    return set(train_df[id_col])