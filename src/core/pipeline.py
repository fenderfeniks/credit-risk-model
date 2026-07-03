from __future__ import annotations

import logging
import pandas as pd
import json
import joblib
from sklearn.pipeline import Pipeline
from omegaconf import DictConfig, OmegaConf

# Внутренние импорты
from src.core.features import TabularPreprocessor, FeatureEngineer
from src.core.models import get_model
from src.core.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Главный оркестратор.
    Сохраняет состояние обучения внутри себя для последующего инференса.
    """

    def __init__(self, cfg: DictConfig, tracker, project_root):
        self.PROJECT_ROOT = project_root
        self.cfg = cfg
        self.preprocessor = None
        self.model = None
        # Инициализируем единый менеджер артефактов
        self.tracker = tracker

    def _get_preprocessor(self) -> Pipeline:
        """Собирает Scikit-Learn пайплайн для обработки фичей."""
        return Pipeline([
            ('engineer', FeatureEngineer(self.cfg)),
            ('cleaner', TabularPreprocessor(self.cfg))     
        ])

    def train(self, X_train, y_train, X_val=None, y_val=None,
              save_artifacts: bool = True, use_tracker: bool = True):
        """Полный цикл обучения: препроцессинг -> модель."""
        logger.info("Начало работы ML-пайплайна...")

        # 1. Собираем и обучаем препроцессор
        self.preprocessor = self._get_preprocessor()
        logger.info("Применение препроцессинга к Train...")

        # Сначала фитим сам пайплайн, чтобы sklearn зафиксировал состояние "fitted"
        self.preprocessor.fit(X_train, y_train)

        # Теперь трансформируем данные
        X_train_clean = self.preprocessor.transform(X_train)

        # 2. Обрабатываем валидацию
        X_val_clean = None
        if X_val is not None:
            logger.info("Применение препроцессинга к Validation...")
            X_val_clean = self.preprocessor.transform(X_val)

        # 3. Инициализация и обучение модели
        self.model = get_model(self.cfg, self.PROJECT_ROOT)

        active_tracker = self.tracker if use_tracker else None
        # ПЕРЕДАЕМ ТРЕКЕР В МОДЕЛЬ (Dependency Injection)
        self.model.fit(
            X_train_clean, y_train,
            X_val=X_val_clean, y_val=y_val,
            tracker=active_tracker
        )
        if hasattr(self.model, 'get_feature_importance'):
            fi_df = self.model.get_feature_importance(X_train_clean) 
        else:
            logger.warning(f"У модели {type(self.model).__name__} отсутствует метод get_feature_importance.")
            fi_df = pd.DataFrame({'Feature': X_train_clean.columns, 'Importance': 0})

        # === ФИНАЛЬНАЯ ОЦЕНКА МЕТРИК ===
        if X_val is not None and y_val is not None:
            logger.info("Расчет финальных бизнес-метрик на Validation...")

            # 1. Предсказываем классы / значения
            y_val_pred = self.model.predict(X_val_clean)

            error_df = X_val.copy() 
            error_df['Actual'] = y_val.values if hasattr(y_val, 'values') else y_val
            error_df['Predicted'] = y_val_pred

            # 2. Пытаемся получить вероятности (если это классификация)
            y_val_prob = None
            if self.cfg.task_type in ['binary', 'multiclass']:
                try:
                    if hasattr(self.model, 'predict_proba'):
                        y_val_prob = self.model.predict_proba(X_val_clean)
                except NotImplementedError:
                    pass  # Для кастомных моделей, где predict_proba не реализован

            # 3. Считаем все метрики разом
            final_metrics = calculate_metrics(
                y_true=y_val, 
                y_pred=y_val_pred, 
                y_prob=y_val_prob,
                task_type=self.cfg.task_type
            )

            # 4. Логируем через менеджер артефактов
            if active_tracker:
                #намеренный внутренний импорт для облегчения API docker image
                from src.eda.visualisation import ReportBuilder
                import matplotlib.pyplot as plt

                final_metrics_to_log = {f"final_val_{k}": v for k, v in final_metrics.items()}
                active_tracker.log_metrics(final_metrics_to_log)

                reporter = ReportBuilder(self.cfg, self.PROJECT_ROOT)
                # 2. Генерируем и логируем сложные графики анализа ошибок
                try:
                    logger.info("Отправка продвинутой графической аналитики в MLflow...")

                    # Анализ общей ошибки
                    fig_err = reporter.error_analyse(self.model, error_df, X_val_clean)
                    active_tracker.log_figure(fig_err, "error_analysis.png", "plots")
                    plt.close(fig_err) # Освобождаем память!

                    # ROC-AUC кривая (только бинарная классификация, если есть вероятности)
                    if self.cfg.task_type == 'binary' and y_val_prob is not None:
                        fig_roc = reporter.roc_auc_curve(y_val, y_val_prob)
                        if fig_roc:
                            active_tracker.log_figure(fig_roc, "roc_auc_curve.png", "plots")
                            plt.close(fig_roc)

                    # Тренды ошибок по категориальным признакам
                    fig_trends = reporter.search_trends(error_df)
                    if fig_trends: # Если вернулась фигура
                        active_tracker.log_figure(fig_trends, "feature_error_trends.png", "plots")
                        plt.close(fig_trends)

                    # Анализ важности признаков
                    fi_figs = reporter.feature_importance(fi_df)
                    active_tracker.log_figure(fi_figs['top_importance'], "importance_top.png", "plots")
                    active_tracker.log_figure(fi_figs['worst_importance'], "importance_worst.png", "plots")
                    plt.close(fi_figs['top_importance'])
                    plt.close(fi_figs['worst_importance'])
                    
                except Exception as plot_err:
                    logger.warning(f"Метрики записаны, но не удалось построить/залогировать графики: {plot_err}")

            for m_name, m_value in final_metrics.items():
                logger.info(f"Final {m_name.upper()}: {m_value:.4f}")

        # 4. Сохранение артефактов
        if save_artifacts:
            logger.info("Сохранение артефактов...")

            # Сохраняем конфиг в трекер
            if active_tracker:
                active_tracker.log_dict(OmegaConf.to_container(self.cfg, resolve=True), "config.json")

            # Подготавливаем папку
            models_dir = self.PROJECT_ROOT / self.cfg.paths.models_dir
            models_dir.mkdir(parents=True, exist_ok=True)

            # --- ПРАВКА 1: Берём новые версии из раздельных конфигов ---
            prep_ver = self.cfg.data.tabular.preprocessing_version
            feat_ver = self.cfg.data.tabular.features_version
            model_ver = self.cfg.model.model_version

            # 1. Сохраняем схему фичей для FastAPI
            feature_types = X_train.dtypes.apply(lambda x: x.name).to_dict()
            schema_name = f"feature_schema_v{feat_ver}.json"
            schema_path = models_dir / schema_name

            with open(schema_path, "w") as f:
                json.dump(feature_types, f, indent=4)

            if active_tracker:
                active_tracker.log_dict(feature_types, schema_name, "schemas")
            logger.info(f"Схема фичей сохранена in {schema_path}")

            # 2. Сохраняем препроцессор (версия из data.tabular)
            prep_path = models_dir / f"preprocessing_v{prep_ver}.pkl"
            joblib.dump(self.preprocessor, prep_path)
            if active_tracker:
                active_tracker.log_artifact(str(prep_path), "preprocessing")

            # 3. Сохраняем модель (версия из model)
            model_path = self.model.save() # Внутри обертки CatBoostWrapper подхватится модель_версия
            if active_tracker:
                active_tracker.log_artifact(model_path, "models")

            # 4. Логируем параметры из конфига
            if active_tracker and hasattr(self.cfg.model, 'params'):
                active_tracker.log_params(
                    OmegaConf.to_container(self.cfg.model.params, resolve=True)
                )

        return self.model

    def predict(self, X):
        """Экспортный метод для инференса (используется в FastAPI и тестах)."""
        if self.preprocessor is None or self.model is None:
            raise ValueError("Пайплайн еще не обучен! Вызовите метод train() перед predict().")

        # Сквозной инференс: чистим -> прогнозируем
        X_clean = self.preprocessor.transform(X)
        return self.model.predict(X_clean)

    def load(self) -> "MLPipeline":
        """
        Восстанавливает состояние пайплайна из сохраненных артефактов на диске.
        Использует раздельные версии компонентов.
        """
        # --- ПРАВКА 2: Извлекаем раздельные версии для загрузки ---
        prep_ver = self.cfg.data.tabular.preprocessing_version
        model_ver = self.cfg.model.model_version
        
        logger.info(f"Загрузка артефактов (Prep: v{prep_ver}, Model: v{model_ver})...")

        models_dir = self.PROJECT_ROOT / self.cfg.paths.models_dir
        if not models_dir.exists():
            raise FileNotFoundError(f"Директория с моделями не найдена: {models_dir}")

        # 1. Загрузка препроцессора по его личной версии
        prep_path = models_dir / f"preprocessing_v{prep_ver}.pkl"
        if not prep_path.exists():
            raise FileNotFoundError(f"Препроцессор не найден: {prep_path}")
        self.preprocessor = joblib.load(prep_path)
        logger.info("Препроцессор успешно загружен.")

        # 2. Инициализация архитектуры модели и загрузка весов по версии модели
        self.model = get_model(self.cfg, self.PROJECT_ROOT)
        model_path = self.model.get_artifact_path(models_dir, model_ver)
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
            
        self.model.load(str(model_path))

        logger.info("Пайплайн успешно восстановлен и готов к инференсу!")
        return self