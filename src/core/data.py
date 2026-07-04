import logging
from abc import ABC, abstractmethod
import pandas as pd
from omegaconf import DictConfig
import duckdb
from src.core.stats import resolve_features_source

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
        logger.error(f"ВНИМАНИЕ! Python прямо сейчас читает вот этот файл: {sql_path}")
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
        
        features_source = resolve_features_source(self.cfg, self.PROJECT_ROOT)
        target_source = f"read_csv_auto('{self.target_path.as_posix()}')"

        sql_params = sql_injections.copy() if sql_injections else {}
        sql_params.update({
            "features_source": features_source,
            "target_source": target_source,
            "id_col": self.id_col
        })

        template_text = self._read_sql_template()
        
        try:
            final_sql = template_text.format(**sql_params)
        except KeyError as e:
            import re
            # Вытаскиваем все плейсхолдеры из текста через регулярку
            placeholders = re.findall(r'\{([^{}]+)\}', template_text)
            missing = [p for p in placeholders if p not in sql_params]
            
            logger.error("=" * 60)
            logger.error(f"[ОШИБКА РЕНДЕРИНГА] Python не нашел ключи: {missing}")
            
            # Ищем, на каких строках в SQL-файле они находятся
            lines = template_text.split('\n')
            for i, line in enumerate(lines):
                for m in missing:
                    if f"{{{m}}}" in line:
                        logger.error(f"-> Строка {i+1} в SQL-файле содержит: {line.strip()}")
            
            logger.error("=" * 60)
            raise
        df = duckdb.query(final_sql).to_df()
        return df


class ProdSparkDataSource(BaseDataSource):
    """
    PROD-режим: Чтение множества файлов train_data_*.pq через PySpark.
    Выполняет тяжелый ETL и собирает итоговый датафрейм.
    """
    def load(self, sql_injections: dict = None) -> pd.DataFrame:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col
        from pyspark.sql.types import DecimalType, DoubleType

        spark = (
            SparkSession.builder
            .appName("ProdDataLoader")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.shuffle.partitions", "50")
            .getOrCreate()
        )

        features_glob = self.PROJECT_ROOT / self.cfg.paths.raw_dir / self.cfg.paths.prod_data_file_globbing

        spark.read.parquet(features_glob.as_posix()).createOrReplaceTempView("spark_raw_features_full")
        
        # ДОБАВЛЕНО: Сэмплирование по ID для PROD
        sample_pct = self.cfg.data.get("sample_pct", 1.0)
        if sample_pct < 1.0:
            spark.sql(
                f"CREATE OR REPLACE TEMP VIEW spark_raw_features AS "
                f"SELECT * FROM spark_raw_features_full "
                f"WHERE ABS(CAST({self.id_col} AS BIGINT)) % 10000 / 100.0 <= {sample_pct * 100}"
            )
        else:
            # Если сэмплирование не нужно, просто делаем алиас
            spark.sql("CREATE OR REPLACE TEMP VIEW spark_raw_features AS SELECT * FROM spark_raw_features_full")

        spark.read.csv(self.target_path.as_posix(), header=True, inferSchema=True).createOrReplaceTempView("spark_raw_target")

        sql_params = sql_injections.copy() if sql_injections else {}
        sql_params.update({
            "features_source": "spark_raw_features",
            "target_source": "spark_raw_target",
            "id_col": self.id_col
        })

        final_sql = self._read_sql_template().format(**sql_params)
        df = spark.sql(final_sql)

        # Приводим все Decimal-колонки к Double — иначе pandas получит object-колонки
        # с decimal.Decimal, что ломает numpy.quantile/percentile ниже по пайплайну
        decimal_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, DecimalType)]
        for c in decimal_cols:
            df = df.withColumn(c, col(c).cast(DoubleType()))

        pdf = df.toPandas()
        return pdf
    

def get_data_source(cfg: DictConfig, project_root) -> BaseDataSource:
    """
    Фабрика для выбора нужного загрузчика на основе конфига.
    """
    mode = cfg.get("env", "dev") # или cfg.env, смотря как у тебя названо
    
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
