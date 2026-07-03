from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.core.models.base import BaseModelWrapper

logger = logging.getLogger(__name__)


class ReportBuilder:
    """
    Единая точка построения графической аналитики после обучения/оценки модели.

    Инкапсулирует повторяющийся boilerplate (reports_dir, версии, сохранение на диск),
    общий для всех графиков. Каждый публичный метод возвращает matplotlib Figure
    (или dict фигур), готовую для передачи в tracker.log_figure(...).

    Использование:
        reporter = ReportBuilder(cfg, project_root)
        fig_err = reporter.error_analyse(model, error_df, X_val_clean)
        fig_roc = reporter.roc_auc_curve(y_true, y_prob)
    """

    def __init__(self, cfg, project_root: Path):
        self.cfg = cfg
        self.project_root = project_root
        self.task_type = cfg.task_type
        self.run_name = cfg.run_name

        self.reports_dir = Path(project_root / cfg.paths.reports_dir / self.run_name)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Версии, часто фигурирующие в именах файлов отчетов
        self.preprocessing_version = cfg.data.tabular.preprocessing_version
        self.features_version = cfg.data.tabular.features_version

    # ==========================================================
    # ВНУТРЕННИЕ УТИЛИТЫ
    # ==========================================================

    def _save(self, fig, filename: str) -> Path:
        """Единая точка сохранения фигуры на диск в reports_dir."""
        output_path = self.reports_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        return output_path

    def _dropped_by_config(self) -> list:
        return list(self.cfg.data.tabular.get('drop_cols', []))

    # ==========================================================
    # 1. ОБЩЕЕ КАЧЕСТВО ПРЕДСКАЗАНИЙ (Confusion Matrix / Факт vs Прогноз)
    # ==========================================================
    def error_analyse(self, model: BaseModelWrapper, error_df: pd.DataFrame, X_val_clean: pd.DataFrame):
        """Визуализирует общее качество предсказаний на основе типа задачи (task_type).

        Для классификации (binary/multiclass) строит матрицу ошибок (Confusion Matrix).
        Для регрессии строит графики Факт vs Прогноз и гистограмму распределения остатков.

        Args:
            model (BaseModelWrapper): Обученная модель в защищенной обертке.
            error_df (pd.DataFrame): Сырой датафрейм валидации с колонками ['Actual', 'Predicted'].
            X_val_clean (pd.DataFrame): Очищенная матрица признаков, переданная в модель.
        """
        if self.task_type in ['binary', 'multiclass']:
            error_df['Is_Error'] = (error_df['Actual'] != error_df['Predicted']).astype(int)

            if hasattr(model, 'predict_proba') and self.task_type == 'binary':
                error_df['Probability'] = model.predict_proba(X_val_clean)[:, 1]

            cm = confusion_matrix(error_df['Actual'], error_df['Predicted'])

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax,
                xticklabels=np.unique(error_df['Actual']),
                yticklabels=np.unique(error_df['Actual'])
            )
            plt.title(f"Матрица ошибок (Confusion Matrix) | Режим: {self.task_type.upper()}")
            plt.ylabel("Реальные значения (Actual)")
            plt.xlabel("Предсказания модели (Predicted)")

        elif self.task_type == 'regression':
            error_df['Error'] = error_df['Predicted'] - error_df['Actual']
            error_df['AbsError'] = error_df['Error'].abs()

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            sns.scatterplot(data=error_df, x='Actual', y='Predicted', alpha=0.6, ax=axes[0])
            max_val = max(error_df['Actual'].max(), error_df['Predicted'].max())
            min_val = min(error_df['Actual'].min(), error_df['Predicted'].min())
            axes[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Идеальный прогноз')
            axes[0].set_title("Соотношение: Факт vs Предсказание")
            axes[0].set_ylabel("Предсказано моделью")
            axes[0].set_xlabel("Реальный таргет")
            axes[0].legend()

            sns.histplot(data=error_df, x='Error', kde=True, ax=axes[1], color='purple')
            axes[1].axvline(0, color='red', linestyle='--')
            axes[1].set_title("Распределение величины ошибок (Остатки / Residuals)")
            axes[1].set_xlabel("Величина ошибки (Predicted - Actual)")
            axes[1].set_ylabel("Количество строк")

        else:
            raise ValueError(f"Неизвестный task_type для error_analyse: {self.task_type}")

        plt.tight_layout()
        self._save(fig, f"error_analyse_{self.preprocessing_version}.png")

        return fig

    # ==========================================================
    # 2. ROC-AUC CURVE (только бинарная классификация)
    # ==========================================================
    def roc_auc_curve(self, y_true, y_prob):
        """Строит ROC-AUC кривую для бинарной классификации.

        Args:
            y_true: Истинные метки классов (0/1).
            y_prob: Вероятности положительного класса. Если передан 2D массив
                (колонка на класс), автоматически берется столбец с индексом 1.
        """
        if self.task_type != 'binary':
            logger.warning(f"roc_auc_curve поддерживает только task_type='binary', получено '{self.task_type}'. Пропуск.")
            return None

        y_prob = np.asarray(y_prob)
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color='#1E88E5', linewidth=2, label=f"ROC-AUC = {auc_score:.4f}")
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1, label='Случайный классификатор')
        ax.set_title(f"ROC-AUC Curve ({self.cfg.model.name})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc='lower right')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()
        self._save(fig, f"roc_auc_curve_{self.preprocessing_version}.png")

        return fig

    # ==========================================================
    # 3. ТРЕНДЫ ОШИБОК ПО КАТЕГОРИАЛЬНЫМ ПРИЗНАКАМ
    # ==========================================================
    def search_trends(self, error_df: pd.DataFrame, top_n_features: int = 6, top_k_categories: int = 15):
        """Ищет скрытые тренды и аномалии в ошибках модели в разрезе категориальных признаков.

        Генерирует комплексное графическое полотно. Для классификации визуализирует
        долю ошибок внутри каждой категории, для регрессии — разброс остатков с помощью boxplot.
        Автоматически группирует редкие категории в заглушку '... OTHER ...'.

        Args:
            error_df (pd.DataFrame): Датафрейм анализа ошибок.
            top_n_features (int): Количество анализируемых признаков (размерность сетки графиков).
            top_k_categories (int): Максимальное количество категорий внутри одной фичи для вывода.
        """
        exclude_cols = ['Actual', 'Predicted', 'Is_Error', 'Probability', 'Prediction_Type',
                        'Error', 'AbsError', 'Confidence_Mistake', 'session_id', 'client_id', 'Is_Worst']
        dropped_by_config = self._dropped_by_config()

        cat_cols = [col for col in error_df.columns if col not in exclude_cols and col not in dropped_by_config and error_df[col].dtype == 'object']

        features_to_analyze = cat_cols[:top_n_features]
        n_features = len(features_to_analyze)

        if n_features == 0:
            logger.info("Категориальных признаков для анализа не найдено.")
            return None

        logger.info(f"=== ЗАПУСК ПОСТРОЕНИЯ СВОДНОГО ОТЧЕТА ТРЕНДОВ ({n_features} ФИЧЕЙ) ===")

        n_cols = 2
        n_rows = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes = axes.flatten()

        if self.task_type in ['binary', 'multiclass']:
            error_df['Is_Error'] = (error_df['Actual'] != error_df['Predicted']).astype(int)

            for i, col in enumerate(features_to_analyze):
                top_categories = error_df[col].value_counts().index[:top_k_categories]

                plot_df = error_df.copy()
                plot_df[col] = plot_df[col].apply(lambda x: x if x in top_categories else '... OTHER ...')
                plot_order = list(top_categories) + ['... OTHER ...'] if '... OTHER ...' in plot_df[col].values else list(top_categories)
                plot_df[col] = pd.Categorical(plot_df[col], categories=plot_order, ordered=True)

                sns.histplot(
                    data=plot_df, x=col, hue='Is_Error', multiple='fill',
                    shrink=0.8, palette={0: '#2ed573', 1: '#ff4757'}, ax=axes[i], legend=True if i == 0 else False
                )

                axes[i].set_title(f"Доля ошибок по фиче: {col}", fontsize=12, fontweight='bold')
                axes[i].set_ylabel("Доля ответов")
                axes[i].set_xlabel("")
                axes[i].axhline(0.5, color='black', linestyle=':', alpha=0.3)
                axes[i].tick_params(axis='x', rotation=35)

                for tick in axes[i].get_xticklabels():
                    tick.set_horizontalalignment('right')

        elif self.task_type == 'regression':
            for i, col in enumerate(features_to_analyze):
                top_categories = error_df[col].value_counts().index[:top_k_categories]

                plot_df = error_df.copy()
                plot_df[col] = plot_df[col].apply(lambda x: x if x in top_categories else '... OTHER ...')
                plot_order = list(top_categories) + ['... OTHER ...'] if '... OTHER ...' in plot_df[col].values else list(top_categories)
                plot_df[col] = pd.Categorical(plot_df[col], categories=plot_order, ordered=True)

                sns.boxplot(
                    data=plot_df, x=col, y='Error', order=plot_order, palette='coolwarm', ax=axes[i]
                )

                axes[i].axhline(0, color='red', linestyle='--', linewidth=1.2)
                axes[i].set_title(f"Разброс ошибок (Residuals) по фиче: {col}", fontsize=12, fontweight='bold')
                axes[i].set_ylabel("Ошибка (Predicted - Actual)")
                axes[i].set_xlabel("")
                axes[i].tick_params(axis='x', rotation=35)
                for tick in axes[i].get_xticklabels():
                    tick.set_horizontalalignment('right')

        for j in range(n_features, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        output_path = self._save(fig, f"features_error_trends_v{self.preprocessing_version}.png")

        logger.info(f"✅ Сводное полотно трендов успешно сохранено в: {output_path}")

        return fig

    # ==========================================================
    # 4. ДОСЬЕ ПО ХУДШИМ ПРЕДСКАЗАНИЯМ
    # ==========================================================
    def worst_preds(self, error_df: pd.DataFrame, top_features_count: int = 5):
        """Выполняет глубокий профайлинг и досье для худших предсказаний модели.

        Для классификации выделяет топ-5 False Positive (самые уверенные ошибки) и топ-5
        False Negative. Для регрессии находит топ-10 выбросов по Absolute Error.
        Печатает текстовые карточки объектов с их физическими признаками в консоль
        и дублирует отчет в текстовый файл (.txt) в директорию отчетов.

        Args:
            error_df (pd.DataFrame): Датафрейм анализа ошибок.
            top_features_count (int): Количество бизнес-фичей, выводимых в текстовое досье объектов.
        """
        logger.info(f"=== ЗАПУСК ЛОКАЛЬНОГО АНАЛИЗА ОШИБОК | РЕЖИМ: {self.task_type.upper()} ===")

        internal_cols = ['Actual', 'Predicted', 'Is_Error', 'Probability', 'Confidence_Mistake', 'Prediction_Type', 'Is_Worst']
        dropped_by_config = self._dropped_by_config()
        available_features = [col for col in error_df.columns if col not in internal_cols and col not in dropped_by_config]
        features_to_print = available_features[:top_features_count]

        plot_path = self.reports_dir / f"worst_errors_plots_v{self.preprocessing_version}.png"
        text_path = self.reports_dir / f"worst_errors_profile_v{self.preprocessing_version}.txt"

        report_text = ""
        fig = None

        if self.task_type in ['binary', 'multiclass']:
            if 'Probability' in error_df.columns and self.task_type == 'binary':
                error_df['Confidence_Mistake'] = np.where(
                    error_df['Actual'] == 1, 1 - error_df['Probability'], error_df['Probability']
                )

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                sns.histplot(data=error_df[error_df['Is_Error'] == 1], x='Probability',
                            hue='Actual', multiple='dodge', bins=20, ax=axes[0], palette={0: '#ff4757', 1: '#1e90ff'})
                axes[0].set_title("Распределение ошибок по вероятностям")

                worst_fp = error_df[(error_df['Actual'] == 0) & (error_df['Is_Error'] == 1)].sort_values(by='Probability', ascending=False).head(5)
                worst_fn = error_df[(error_df['Actual'] == 1) & (error_df['Is_Error'] == 1)].sort_values(by='Probability', ascending=True).head(5)
                worst_cases = pd.concat([worst_fp, worst_fn])

                worst_plot_data = worst_cases[['Actual', 'Probability']].copy()
                worst_plot_data['Label'] = [f"Факт: {int(a)} | Предикт: {p:.3f}" for a, p in zip(worst_plot_data['Actual'], worst_plot_data['Probability'])]
                worst_plot_data = worst_plot_data.set_index('Label')

                sns.heatmap(worst_cases[['Probability']], annot=True, cmap='Reds', cbar=False, fmt='.3f', ax=axes[1], annot_kws={"size": 14})
                axes[1].set_title("ТОП самых самоуверенных ошибок")
                axes[1].set_yticklabels(worst_plot_data.index, rotation=0)

                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')

                report_text = f"=== ОТЧЕТ ПО ХУДШИМ ОШИБКАМ КЛАССИФИКАЦИИ ===\n"
                report_text += f"Версия фичей: {self.features_version} | Версия препроцессинга: {self.preprocessing_version}\n"
                report_text += "="*70 + "\n\n"

                report_text += " ТОП-5 FALSE POSITIVE (Модель уверенно ждала конверсию, но её не было):\n"
                for idx, row in worst_fp.iterrows():
                    report_text += f"  • Строка [ID {idx}] | Вероятность модели: {row['Probability']:.3f}\n"
                    for f in features_to_print:
                        report_text += f"    - {f}: {row[f]}\n"
                    report_text += "  " + "-"*45 + "\n"

                report_text += "\n ТОП-5 FALSE NEGATIVE (Конверсия была, но модель её полностью пропустила):\n"
                for idx, row in worst_fn.iterrows():
                    report_text += f"  • Строка [ID {idx}] | Вероятность модели: {row['Probability']:.3f}\n"
                    for f in features_to_print:
                        report_text += f"    - {f}: {row[f]}\n"
                    report_text += "  " + "-"*45 + "\n"
            else:
                worst_cases = error_df[error_df['Is_Error'] == 1].head(10)
                report_text = "=== ТОП-10 ОШИБОК КЛАССИФИКАЦИИ (БЕЗ ВЕРОЯТНОСТЕЙ) ===\n"
                for idx, row in worst_cases.iterrows():
                    report_text += f"  • Строка [ID {idx}] | Факт: {row['Actual']} | Предикт: {row['Predicted']}\n"
                    for f in features_to_print:
                        report_text += f"    - {f}: {row[f]}\n"
                    report_text += "  " + "-"*45 + "\n"

        elif self.task_type == 'regression':
            worst_indices = error_df.sort_values(by='AbsError', ascending=False).head(10).index
            error_df['Is_Worst'] = error_df.index.isin(worst_indices).astype(int)

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            sns.scatterplot(data=error_df, x='Actual', y='Predicted', hue='Is_Worst',
                            palette={0: '#747d8c', 1: '#ff4757'}, alpha=0.7, ax=axes[0])
            axes[0].set_title("Точки максимального крушения модели")

            worst_cases = error_df.loc[worst_indices].copy()
            worst_cases['Index_Str'] = worst_cases.index.astype(str)

            sns.barplot(data=worst_cases, x='Error', y='Index_Str', ax=axes[1], palette='vlag')
            axes[1].axvline(0, color='black', linestyle='--')
            axes[1].set_title("Величина отклонения в ТОП-10 худших прогнозах")

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')

            report_text = f"=== ОТЧЕТ ПО КАТАСТРОФИЧЕСКИМ ВЫБРОСАМ РЕГРЕССИИ ===\n"
            report_text += f"Версия фичей: {self.features_version} | Версия препроцессинга: {self.preprocessing_version}\n"
            report_text += "="*70 + "\n\n"

            report_text += " ТОП-10 ХУДШИХ ПРОГНОЗОВ (Где модель промахнулась сильнее всего):\n"
            for idx, row in worst_cases.iterrows():
                report_text += f"  • Строка [ID {idx}] | Реально: {row['Actual']:.2f} | Предсказано: {row['Predicted']:.2f} | Ошибка: {row['Error']:.2f}\n"
                for f in features_to_print:
                    report_text += f"    - {f}: {row[f]}\n"
                report_text += "  " + "-"*45 + "\n"

        logger.info(report_text)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("✅ Локальные отчеты успешно сгенерированы и сохранены:")
        logger.info(f"   - Графики сохранены в: {plot_path}")
        logger.info(f"   - Текстовое досье сохранено в: {text_path}")

        return fig

    # ==========================================================
    # 5. ВАЖНОСТЬ ПРИЗНАКОВ
    # ==========================================================
    def feature_importance(self, fi_df: pd.DataFrame, features_count: int = 15):
        figures = {}

        fig_top = plt.figure()
        sns.barplot(
            data=fi_df.head(features_count),
            x='Importance',
            y='Feature',
            palette='viridis',
            alpha=self.cfg.logging.plots.alpha
        )
        plt.title(f"Топ-{features_count} самых важных признаков ({self.cfg.model.name})")
        plt.tight_layout()
        figures['top_importance'] = fig_top
        self._save(fig_top, f"features_importance_top{features_count}_{self.preprocessing_version}.png")

        worst_features = fi_df.sort_values(by='Importance', ascending=True).head(features_count)

        fig_worst = plt.figure()
        sns.barplot(
            data=worst_features,
            x='Importance',
            y='Feature',
            palette='viridis',
            alpha=self.cfg.logging.plots.alpha
        )
        plt.title(f"Топ-{features_count} самых худших признаков ({self.cfg.model.name})")
        plt.tight_layout()
        figures['worst_importance'] = fig_worst
        self._save(fig_worst, f"features_importance_worst_{features_count}_{self.preprocessing_version}.png")

        return figures


# ============================================================
# ФУНКЦИИ БЕЗ ПРИВЯЗКИ К cfg/project_root — оставлены как есть
# (используются напрямую из EDA-ноутбуков, не завязаны на reports_dir)
# ============================================================

def plot_sweetviz_style(df: pd.DataFrame, feature: str, target: str = 'target'):
    """
    Отрисовывает график распределения бинов и линию таргета в стиле Sweetviz.
    """
    stats = df.groupby(feature).agg(
        Count=(target, 'size'),
        Target_Sum=(target, 'sum'),
        Target_Mean=(target, 'mean')
    ).reset_index()

    if stats.empty:
        print(f"ПРОПУСК: Признак {feature} пуст после фильтрации. График не строится.")
        return

    stats['Percent'] = (stats['Count'] / len(df)) * 100
    stats = stats.sort_values('Count', ascending=False).reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(stats))

    ax1.barh(y_pos, stats['Percent'], color='#1E88E5', height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(stats[feature])
    ax1.invert_yaxis()
    ax1.set_xlabel('Доля в датасете (%)')
    ax1.set_xlim(0, max(stats['Percent']) * 1.1)

    ax2 = ax1.twiny()
    ax2.plot(stats['Target_Mean'] * 100, y_pos, color='#0D47A1', marker='o', linewidth=2, markersize=6)
    ax2.set_xlabel('Конверсия в таргет (%)', color='#0D47A1')

    max_target = max(stats['Target_Mean'] * 100)
    ax2.set_xlim(0, max(max_target * 1.2, 5.0))

    plt.title(f'Анализ признака: {feature}', pad=20)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.show()

    print(f"{'TOP CATEGORIES':<20} | {'Count':<10} | {'%':<6} | {'Target Cnt':<10} | {'Target %':<8}")
    print("-" * 65)

    for _, row in stats.iterrows():
        print(f"{row[feature]:<20} | {int(row['Count']):<10} | {row['Percent']:>4.1f}% | {int(row['Target_Sum']):<10} | {row['Target_Mean']*100:>6.1f}%")

    total_target_pct = df[target].mean() * 100
    print("-" * 65)
    print(f"{'ALL':<20} | {len(df):<10} | 100.0% | {int(df[target].sum()):<10} | {total_target_pct:>6.1f}%\n")