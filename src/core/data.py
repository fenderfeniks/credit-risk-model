import json
import logging
from abc import ABC, abstractmethod
import pandas as pd
from omegaconf import DictConfig
import duckdb

logger = logging.getLogger(__name__)

class BaseDataSource(ABC):
    """
    Базовый контракт для источников данных.
    """
    def __init__(self, cfg: DictConfig, project_root):
        self.cfg = cfg
        self.PROJECT_ROOT = project_root
        self.target_path = self.PROJECT_ROOT / self.cfg.paths.raw_dir / self.cfg.paths.target_file_name
        self.id_col = self.cfg.data.tabular.get('id_col', 'id')
        

    def _read_sql_template(self) -> str:
        """Общий метод для чтения SQL-шаблона из файловой системы."""
        sql_path = self.PROJECT_ROOT / self.cfg.paths.sql_aggregate_dir / self.cfg.paths.aggregate_sql_request
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL-шаблон не найден по пути: {sql_path}")
        with open(sql_path, "r", encoding="utf-8") as f:
            return f.read()

           
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass


class DevDuckDBDataSource(BaseDataSource):
    """
    DEV-режим: Загрузка локального файла (или сэмпла) с помощью DuckDB.
    Позволяет фильтровать данные "на диске" до загрузки в RAM.
    """
    def load(self, sql_injections: dict = None) -> pd.DataFrame:
        features_path = self.PROJECT_ROOT / self.cfg.paths.raw_dir / self.cfg.paths.dev_data_file

        features_source = f"read_parquet('{features_path.as_posix()}')"
        target_source = f"read_csv_auto('{self.target_path.as_posix()}')"

        # 1. Берем переданный словарь (защита от None)
        sql_params = sql_injections.copy() if sql_injections else {}
        
        # 2. Добавляем системные параметры
        sql_params.update({
            "features_source": features_source,
            "target_source": target_source,
            "id_col": self.id_col
        })

        # 3. Инжектируем всё в шаблон
        final_sql = self._read_sql_template().format(**sql_params)
        
        # 4. Выполняем
        df = duckdb.query(final_sql).to_df()
        return df


class ProdSparkDataSource(BaseDataSource):
    """
    PROD-режим: Чтение множества файлов train_data_*.pq через PySpark.
    Выполняет тяжелый ETL и собирает итоговый датафрейм.
    """
    def load(self, sql_injections: dict = None) -> pd.DataFrame:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("ProdDataLoader").getOrCreate()

        features_glob = self.PROJECT_ROOT / self.cfg.paths.raw_dir / self.cfg.paths.prod_data_file_globbing
        
        spark.read.parquet(features_glob.as_posix()).createOrReplaceTempView("spark_raw_features")
        spark.read.csv(self.target_path.as_posix(), header=True, inferSchema=True).createOrReplaceTempView("spark_raw_target")

        # 1. Берем переданный словарь
        sql_params = sql_injections.copy() if sql_injections else {}
        
        # 2. Добавляем системные параметры
        sql_params.update({
            "features_source": "spark_raw_features",
            "target_source": "spark_raw_target",
            "id_col": self.id_col
        })

        # 3. Форматируем и выполняем
        final_sql = self._read_sql_template().format(**sql_params)
        pdf = spark.sql(final_sql).toPandas()

        return pdf
    

def get_data_source(cfg: DictConfig, project_root) -> BaseDataSource:
    """
    Фабрика для выбора нужного загрузчика на основе конфига.
    """
    mode = cfg.get("env@_global_", "dev") # или cfg.env, смотря как у тебя названо
    
    if mode == "dev":
        return DevDuckDBDataSource(cfg, project_root)
    elif mode == "prod":
        return ProdSparkDataSource(cfg, project_root)
    else:
        raise ValueError(f"Неизвестный режим запуска: {mode}")
    


def load_eda_feature_data(
    cfg, 
    PROJECT_ROOT, 
    feature_name: str, 
    dom_bin: int
) -> pd.DataFrame:
    """
    Загружает срез данных для одной фичи:
    1. Глоббинг всех train_data_*.pq
    2. Фильтр только одного столбца (исключение доминанты)
    3. LEFT JOIN с CSV-таргетом по id_col
    """
    # Пути из конфига
    raw_dir = PROJECT_ROOT / cfg.paths.raw_dir
    features_glob = (raw_dir / cfg.paths.prod_data_file_globbing).as_posix()
    target_path = (raw_dir / cfg.paths.target_file_name).as_posix()
    id_col = cfg.data.tabular.get('id_col', 'id')
    target_col = cfg.data.tabular.get('target_col', 'target')

    logger.info(f"Загрузка EDA для фичи '{feature_name}' (исключая бин {dom_bin})...")

    # Формируем запрос
    # Используем динамический JOIN, чтобы не грузить лишние колонки
    query = f"""
        WITH filtered_features AS (
            SELECT {id_col}, {feature_name}
            FROM read_parquet('{features_glob}')
            WHERE {feature_name} != {dom_bin} OR {feature_name} IS NULL
        )
        SELECT f.{id_col}, f.{feature_name}, t.{target_col}
        FROM filtered_features f
        LEFT JOIN read_csv_auto('{target_path}') t 
          ON f.{id_col} = t.{id_col}
    """
    
    # Выполняем в DuckDB
    df = duckdb.query(query).to_df()
    
    logger.info(f"Срез загружен. Размер: {df.shape}")
    return df
