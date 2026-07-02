# src/scripts/upload_raw_data.py
import logging
import hydra
from omegaconf import DictConfig
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from src.core.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

def get_db_engine(cfg: DictConfig):
    db = cfg.database
    conn_string = f"postgresql://{db.user}:{quote_plus(db.password)}@{db.host}:{db.port}/{db.name}"
    return create_engine(conn_string)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info("Старт скрипта загрузки сырых данных в PostgreSQL...")
    engine = get_db_engine(cfg)
    
    # 1. Сначала подготавливаем (создаем) пустые таблицы, выполняя create_tables.sql
    sql_init_path = PROJECT_ROOT / cfg.paths.sql_dir / "init_db" / "create_tables.sql"
    logger.info(f"Читаем схему таблиц из {sql_init_path.name}...")
    
    with open(sql_init_path, "r") as f:
        create_tables_sql = f.read()
        
    with engine.begin() as conn:
        conn.execute(create_tables_sql)
    logger.info("Таблицы успешно пересозданы (очищены).")

    # 2. Загружаем ga_sessions
    sessions_file_path = PROJECT_ROOT / cfg.paths.raw_dir / cfg.paths.raw_sessions_file
    if sessions_file_path.exists():
        logger.info(f"Читаем файл сессий: {sessions_file_path.name}...")
        # Меняй на pd.read_pickle, если файлы в формате .pkl
        df_sessions = pd.read_csv(sessions_file_path) 
        
        logger.info(f"Заливаем {len(df_sessions)} строк в таблицу {cfg.paths.raw_sessions_table}...")
        df_sessions.to_sql(cfg.paths.raw_sessions_table, con=engine, if_exists='append', index=False)
        logger.info("Сессии успешно загружены.")
    else:
        logger.warning(f"Файл сессий не найден по пути: {sessions_file_path}")

    # 3. Загружаем ga_hits
    hits_file_path = PROJECT_ROOT / cfg.paths.raw_dir / cfg.paths.raw_hits_file
    if hits_file_path.exists():
        logger.info(f"Читаем файл хитов: {hits_file_path.name}...")
        # Учитывая, что файл хитов огромный, читаем его чанками (батчами), чтобы не взорвать RAM
        chunk_size = 100_000
        logger.info(f"Заливаем хиты в таблицу {cfg.paths.raw_hits_table} чанками по {chunk_size} строк...")
        
        # Меняй на pd.read_csv или специальный генератор для pickle, если файлы не csv
        for chunk in pd.read_csv(hits_file_path, chunksize=chunk_size):
            chunk.to_sql(cfg.paths.raw_hits_table, con=engine, if_exists='append', index=False)
            
        logger.info("Хиты успешно загружены.")
    else:
        logger.warning(f"Файл хитов не найден по пути: {hits_file_path}")

if __name__ == "__main__":
    main()