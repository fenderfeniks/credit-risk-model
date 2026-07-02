from pathlib import Path

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ShapExplainer():
    """Класс для глобальной и локальной интерпретации моделей с помощью SHAP.

    Позволяет оценивать вклад признаков в предсказания модели на всей выборке,
    строить графики зависимостей, а также изолированно анализировать кластеры
    ошибочных предсказаний (False Positives).

    Attributes:
        model_wrapper: Обёртка над ML-моделью, содержащая нативный объект в .model.
        native_model: Нативная древовидная модель (например, CatBoostRegressor/Classifier).
        X_val_clean_df (pd.DataFrame): Предобработанная матрица валидационных признаков.
        y_val (pd.Series): Вектор истинных значений валидационного таргета.
        explainer (shap.TreeExplainer): Инициализированный инструмент расчета SHAP.
        shap_values (shap.Explanation): Рассчитанная матрица SHAP-значений.
    """
    def __init__(self, X_val_clean_df:pd.DataFrame, y_val:pd.Series, model_wrapper, config, project_root):
        self.cfg = config
        self.model_version = config.model.model_version
        self.run_name = config.run_name
        self.model_wrapper = model_wrapper
        self.native_model = model_wrapper.model
        self.X_val_clean_df = X_val_clean_df
        self.y_val = y_val

        self.reports_dir = Path(project_root / self.cfg.paths.reports_dir / self.run_name)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        print("Инициализация TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.native_model)
        print("Расчет SHAP-значений для валидационной выборки...")
        self.shap_values = self.explainer(X_val_clean_df)
        print("Расчет успешно завершен!")

    def global_interpretation(self, max_display=15):
        """Выполняет глобальный анализ влияния признаков на модель.

        Строит график Beeswarm для распределения SHAP-значений, Scatter Dependence plot
        для наиболее важного признака, а также экспортирует упорядоченный CSV-отчет
        со средними абсолютными значениями важности.

        Args:
            max_display (int): Максимальное количество отображаемых фичей на графике. По умолчанию 15.

        Returns:
            pd.DataFrame: Таблица с колонками ['Feature', 'Importance (mean |SHAP|)']
        """
        plt.figure()

        # Beeswarm plot показывает распределение влияния каждого признака
        # Каждая точка — это один юзер/сессия.
        # Направление вправо — увеличивает предсказание, влево — уменьшает.
        shap.plots.beeswarm(self.shap_values, max_display=max_display, show=False)

        plt.title(f"Глобальное влияние признаков на предсказание модели (Top {max_display})", fontsize=14, pad=20)
        plt.tight_layout()

        # Сохраняем график в папку отчетов
        plt.savefig(self.reports_dir / "shap_global_summary.png", bbox_inches='tight')

        plt.show()

        global_importance = np.abs(self.shap_values.values).mean(0)
        feature_importance_df = pd.DataFrame({
            'Feature': self.X_val_clean_df.columns,
            'Importance (mean |SHAP|)': global_importance
        }).sort_values(by='Importance (mean |SHAP|)', ascending=False)
        # Выбираем самый важный признак (первый в списке важности)
        top_feature = feature_importance_df.iloc[0]['Feature']

        plt.figure(figsize=(10, 6))
        # SHAP dependence plot покажет зависимость и автоматически подберет фичу-партнера для раскраски
        shap.plots.scatter(self.shap_values[:, top_feature], color=self.shap_values, show=False)

        plt.title(f"Зависимость SHAP-значения от физического значения фичи: {top_feature}", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(self.reports_dir / f"shap_dependence_{top_feature}.png", bbox_inches='tight')
        plt.show()

        print("=== ТОП ПРИЗНАКОВ ПО ВЛИЯНИЮ НА МОДЕЛЬ ===")

        # Сохраняем текстовую таблицу в отчеты
        feature_importance_df.to_csv(self.reports_dir / f"shap_feature_importance_v{self.model_version}.csv", index=False)

        return feature_importance_df.head(max_display)
    

    def local_interpretation(self, error_df:pd.DataFrame, sample_index:int = 0, max_display:int = 10):
        """Выполняет локальный анализ предсказания для конкретного объекта.

        Генерирует график Waterfall для выбранного индекса строки. Если в переданном
        error_df зафиксированы ложные срабатывания (False Positives), автоматически
        находит 10 наиболее критичных ошибок и визуализирует их структуру через SHAP Heatmap.

        Args:
            error_df (pd.DataFrame): Датафрейм анализа ошибок с колонками ['Actual', 'Predicted'].
            sample_index (int): Позиционный индекс объекта для построения Waterfall. По умолчанию 0.
            max_display (int): Количество факторов, отображаемых в графиках локального анализа.
        """
        plt.figure()

        # Waterfall plot наглядно показывает стартовую базовую вероятность (E[f(X)])
        # и то, как каждый фактор шаг за шагом прибавил или отнял проценты до финального ответа.
        shap.plots.waterfall(self.shap_values[sample_index], max_display=max_display, show=False)

        plt.title(f"Обоснование предсказания для объекта с индексом {sample_index}", fontsize=14, pad=20)
        plt.tight_layout()

        # Сохраняем локальный отчет
        plt.savefig(self.reports_dir / f"shap_local_user_{sample_index}.png", bbox_inches='tight')
        plt.show()

        print(f"Фактическое значение таргета для этого объекта: {self.y_val.iloc[sample_index]}")

        fp_indices = np.where((error_df['Actual'] == 0) & (error_df['Predicted'] == 1))[0]

        if len(fp_indices) > 0:
            print(f"=== АНАЛИЗ SHAP ДЛЯ ЛОЖНЫХ СРАБАТЫВАНИЙ (Найдено объектов: {len(fp_indices)}) ===")
            
            # Берем топ-10 самых наглых ошибок FP
            if 'Probability' in error_df.columns or hasattr(self.model_wrapper, 'predict_proba'):
                probs = self.model_wrapper.predict_proba(self.X_val_clean_df)[:, 1]
                error_df['Probability'] = probs
                worst_fp_idx = error_df.iloc[fp_indices].sort_values(by='Probability', ascending=False).head(10).index
                # Переводим индексы датафрейма в позиционные индексы массива numpy
                pos_indices = [error_df.index.get_loc(idx) for idx in worst_fp_idx]
            else:
                pos_indices = fp_indices[:max_display]
                
            # Строим Heatmap для этих ошибок
            plt.figure(figsize=(12, 6))
            shap.plots.heatmap(self.shap_values[pos_indices], max_display=10, show=False)
            plt.title("Что триггерит ложные срабатывания (False Positives) в модели? (Топ-10 ошибок)", fontsize=12, pad=20)
            plt.tight_layout()
            plt.savefig(self.reports_dir / "shap_heatmap_false_positives.png", bbox_inches='tight')
            plt.show()
        else:
            print("Ложных срабатываний для анализа не найдено.")    

