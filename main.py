from __future__ import annotations

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from dotenv import load_dotenv

# Импортируем только ООП-компоненты из ядра
from src.core.data import get_data_source
from src.core.splitting import split_data
from src.core.pipeline import MLPipeline
from src.core.utils import PROJECT_ROOT
from src.core.metrics import calculate_metrics
from src.core.utils import register_config_schema
from src.core.stats import GlobalStatCompiler, get_train_ids_fast, resolve_features_source

load_dotenv()

register_config_schema()

logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)



def setup_logging(cfg: DictConfig):
    """Настройка динамического логирования в файл на основе конфига Hydra."""
    log_file_path = PROJECT_ROOT / cfg.logging.log_file
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()

    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path) for h in root_logger.handlers):
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root_logger.addHandler(file_handler)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    mode = cfg.mode.lower()
    if mode == "eval": mode = "evaluate"

    # 1. Инициализируем логирование в файл /logs/pipeline.log
    setup_logging(cfg)

    logger.info(f"=== ЗАПУСК ORCHESTRATOR | РЕЖИМ: {mode.upper()} ===")

    # 2. НАСТРОЙКА ИНФРАСТРУКТУРЫ АРТЕФАКТОВ (Без прямого вызова MLflow)
    # ЛЕНИВО: tracker нужен только train/tune/evaluate (они логируют в MLflow).
    # inference его не использует — не тянем mlflow-зависимость в этом режиме
    # и не плодим experiment runs на простом батч-предсказании.
    stats_compiler = GlobalStatCompiler(cfg, PROJECT_ROOT)

    features_source = resolve_features_source(cfg, PROJECT_ROOT)
    # ------------------------------------------

    loader = get_data_source(cfg=cfg, project_root=PROJECT_ROOT)

    tracker = None
    if mode in ("train", "tune", "evaluate"):
        from src.core.artifacts import ArtifactManager
        tracker = ArtifactManager(cfg, project_root=PROJECT_ROOT)
        experiment_name = cfg.logging.mlflow.experiments.get(mode, "default_experiment")

        # 1. Получаем ID трейна
        train_ids = get_train_ids_fast(cfg, PROJECT_ROOT)
        
        # 2. Считаем статистики по трейну (сохранятся на диск и в память класса)
        stats_compiler.fit_and_save(
            features_source=features_source,
            train_ids=train_ids,
            id_col=cfg.data.tabular.get('id_col', 'id')
        )
        
        # 3. ИСПРАВЛЕНИЕ: Забираем свежепосчитанные константы
        sql_injections = stats_compiler.get_sql_format_kwargs()
    
        # Явно указываем локальный путь для тяжелых артефактов и передаем трекеру
        local_artifact_repo = (PROJECT_ROOT / cfg.paths.logs_dir / "mlruns").as_uri()
        tracker.set_experiment(experiment_name, artifact_location=local_artifact_repo)

    elif mode == 'inference':
        # В инференсе просто читаем готовое с диска
        stats_compiler.load()
        sql_injections = stats_compiler.get_sql_format_kwargs()
    

    df = loader.load(sql_injections=sql_injections)
    
    target = cfg.data.tabular.target_col

    # ==========================================================
    # РЕЖИМ 1: ТРЕНИРОВКА
    # ==========================================================
    if mode == "eda":
        logger.info("Запуск модуля EDA...")
        # run_eda(cfg)
    elif mode == "train":

        train_df, val_df, _ = split_data(cfg=cfg, df=df)

        X_train, y_train = train_df.drop(columns=[target]), train_df[target]
        X_val, y_val = val_df.drop(columns=[target]), val_df[target]

        # Запускаем через контекстный менеджер трекера
        with tracker.start_run(run_name=cfg.run_name):
            pipeline = MLPipeline(cfg, tracker=tracker, project_root=PROJECT_ROOT)
            pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=True)

    # ==========================================================
    # РЕЖИМ 2: ТЮНИНГ (OPTUNA)
    # ==========================================================
    elif mode == "tune":
        from src.core.tuner import OptunaTuner
        train_df, val_df, _ = split_data(cfg=cfg, df=df)

        X_train, y_train = train_df.drop(columns=[target]), train_df[target]
        X_val, y_val = val_df.drop(columns=[target]), val_df[target]

        with tracker.start_run(run_name=f"{cfg.run_name}_optuna"):
            tuner = OptunaTuner(cfg, tracker=tracker, project_root=PROJECT_ROOT)
            best_params = tuner.tune(X_train, y_train, X_val, y_val)

            logger.info(f"Тюнинг завершен. Лучшие параметры найдены: {best_params}")
            logger.info("Обучение финальной модели...")
            
            # 1. Обновляем конфиг в памяти
            for key, value in best_params.items():
                OmegaConf.update(cfg, f"model.params.{key}", value, force_add=True)

            # 2. ЖЕЛЕЗОБЕТОННЫЙ ФИКС: Передаем эстафету пайплайну
            with tracker.start_run(run_name=f"{cfg.run_name}_final", nested=True):

                pipeline = MLPipeline(cfg, tracker=tracker, project_root=PROJECT_ROOT)
                # Обучаем финальную модель (теперь она ТОЧНО возьмет лучшие параметры!)
                pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=True)


    # ==========================================================
    # РЕЖИМ 3: ОЦЕНКА (EVALUATE) - Тестирование отложенной выборки
    # ==========================================================
    elif mode == "evaluate":
        logger.info("Запуск оценки модели на тестовой выборке...")
        _, _, test_df = split_data(cfg=cfg, df=df)

        if test_df is None or test_df.empty:
            raise ValueError("Тестовая выборка пуста! Проверьте логику DataLoader.")

        X_test, y_test = test_df.drop(columns=[target]), test_df[target]

        # --- ПРАВКА 1: Используем model_version ---
        with tracker.start_run(run_name=f"{cfg.model.name}_v{cfg.model.model_version}_eval"):
            # Восстанавливаем пайплайн
            pipeline = MLPipeline(cfg, tracker=tracker, project_root=PROJECT_ROOT)
            pipeline.load()

            # Делаем предсказания
            X_test_clean = pipeline.preprocessor.transform(X_test)
            y_pred = pipeline.model.predict(X_test_clean)

            y_prob = None
            if cfg.task_type in ['binary', 'multiclass']:
                try:
                    if hasattr(pipeline.model, 'predict_proba'):
                        y_prob = pipeline.model.predict_proba(X_test_clean)
                except NotImplementedError:
                    pass

            # Считаем и логируем метрики
            test_metrics = calculate_metrics(y_test, y_pred, task_type=cfg.task_type, y_prob=y_prob)

            metrics_to_log = {f"test_{k}": v for k, v in test_metrics.items()}
            tracker.log_metrics(metrics_to_log)

            logger.info("=== РЕЗУЛЬТАТЫ EVALUATION ===")
            for m_name, m_value in test_metrics.items():
                logger.info(f"Test {m_name.upper()}: {m_value:.4f}")

    # ==========================================================
    # РЕЖИМ 4: ИНФЕРЕНС (INFERENCE) - Прогноз новых (слепых) данных
    # ==========================================================
    elif mode == "inference":
        logger.info("Запуск инференса на новых данных...")

        X_new = df.drop(columns=[target]) if target in df.columns else df

        pipeline = MLPipeline(cfg, tracker=tracker, project_root=PROJECT_ROOT)
        pipeline.load()

        # Сквозной предикт (очистка + прогноз)
        predictions = pipeline.predict(X_new)

        result_df = pd.DataFrame({"prediction": predictions})

        if cfg.task_type in ['binary', 'multiclass']:
            X_new_clean = pipeline.preprocessor.transform(X_new)
            try:
                if hasattr(pipeline.model, 'predict_proba'):
                    probs = pipeline.model.predict_proba(X_new_clean)
                    if cfg.task_type == 'binary':
                        result_df["probability"] = probs[:, 1]
            except NotImplementedError:
                pass

        # --- ПРАВКА 2: Используем model_version для пути сохранения ---
        output_path = PROJECT_ROOT / cfg.paths.data_dir / f"predictions_{cfg.model.name}_v{cfg.model.model_version}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)

        logger.info(f"Инференс успешно завершен! Результаты сохранены в: {output_path}")

    else:
        raise ValueError(f"Неизвестный режим: {mode}")


if __name__ == "__main__":
    main()