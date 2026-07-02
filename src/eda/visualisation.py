from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.core.models.base import BaseModelWrapper
from pathlib import Path


def error_analyse(model: BaseModelWrapper, error_df: pd.DataFrame, X_val_clean: pd.DataFrame, cfg, project_root:Path):
    """Визуализирует общее качество предсказаний на основе типа задачи (task_type).

    Для классификации (binary/multiclass) строит матрицу ошибок (Confusion Matrix).
    Для регрессии строит графики Факт vs Прогноз и гистограмму распределения остатков.

    Args:
        model (BaseModelWrapper): Обученная модель в защищенной обертке.
        error_df (pd.DataFrame): Сырой датафрейм валидации с колонками ['Actual', 'Predicted'].
        X_val_clean (pd.DataFrame): Очищенная матрица признаков, переданная в модель.
    """
    task_type = cfg.task_type
    run_name = cfg.run_name
    reports_dir = Path(project_root / cfg.paths.reports_dir / run_name)
    # ============================================================================
    # ВЕТКА 1: БИНАРНАЯ ИЛИ МНОГОКЛАССОВАЯ КЛАССИФИКАЦИЯ (Матрица ошибок)
    # ============================================================================
    if task_type in ['binary', 'multiclass']:
        error_df['Is_Error'] = (error_df['Actual'] != error_df['Predicted']).astype(int)
        
        if hasattr(model, 'predict_proba') and task_type == 'binary':
            error_df['Probability'] = model.predict_proba(X_val_clean)[:, 1]

        # Строим красивую матрицу ошибок (Confusion Matrix)
        cm = confusion_matrix(error_df['Actual'], error_df['Predicted'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Если классов много, аннотации автоматически адаптируются
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            ax=ax,
            xticklabels=np.unique(error_df['Actual']), 
            yticklabels=np.unique(error_df['Actual'])
        )
        plt.title(f"Матрица ошибок (Confusion Matrix) | Режим: {task_type.upper()}")
        plt.ylabel("Реальные значения (Actual)")
        plt.xlabel("Предсказания модели (Predicted)")


    # ============================================================================
    # ВЕТКА 2: РЕГРЕССИЯ (Анализ остатков и плотности ошибок)
    # ============================================================================
    elif task_type == 'regression':
        error_df['Error'] = error_df['Predicted'] - error_df['Actual']
        error_df['AbsError'] = error_df['Error'].abs()
        
        # Строим двухпанельный график анализа распределения ошибок
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # График 1: Scatter plot (Факт vs Предсказание) — идеальный прогноз идет по диагонали
        sns.scatterplot(data=error_df, x='Actual', y='Predicted', alpha=PLOT_ALPHA, ax=axes[0])
        max_val = max(error_df['Actual'].max(), error_df['Predicted'].max())
        min_val = min(error_df['Actual'].min(), error_df['Predicted'].min())
        axes[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Идеальный прогноз')
        axes[0].set_title("Соотношение: Факт vs Предсказание")
        axes[0].set_ylabel("Предсказано моделью")
        axes[0].set_xlabel("Реальный таргет")
        axes[0].legend()
        
        # График 2: Распределение остатков (Где гуще всего ошибки)
        sns.histplot(data=error_df, x='Error', kde=True, ax=axes[1], color='purple')
        axes[1].axvline(0, color='red', linestyle='--')
        axes[1].set_title("Распределение величины ошибок (Остатки / Residuals)")
        axes[1].set_xlabel("Величина ошибки (Predicted - Actual)")
        axes[1].set_ylabel("Количество строк")
        
    plt.tight_layout()

    output_plot_path = reports_dir / f"error_analyse_{cfg.data.tabular.preprocessing_version}.png"
    fig.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    #plt.show()
    
    return fig

    
def search_trends( error_df: pd.DataFrame, cfg, project_root, top_n_features:int = 6, top_k_categories:int = 15):
    """Ищет скрытые тренды и аномалии в ошибках модели в разрезе категориальных признаков.

    Генерирует комплексное графическое полотно. Для классификации визуализирует
    долю ошибок внутри каждой категории, для регрессии — разброс остатков с помощью boxplot.
    Автоматически группирует редкие категории в заглушку '... OTHER ...'.

    Args:
        error_df (pd.DataFrame): Датафрейм анализа ошибок.
        top_n_features (int): Количество анализируемых признаков (размерность сетки графиков).
        top_k_categories (int): Максимальное количество категорий внутри одной фичи для вывода.
    """
    task_type = cfg.task_type
    run_name = cfg.run_name
    reports_dir = Path(project_root / cfg.paths.reports_dir / run_name)
    # Исключаем служебные метки и задропленные в конфигурации фичи
    exclude_cols = ['Actual', 'Predicted', 'Is_Error', 'Probability', 'Prediction_Type', 
                    'Error', 'AbsError', 'Confidence_Mistake', 'session_id', 'client_id', 'Is_Worst']
    dropped_by_config = list(cfg.data.tabular.get('drop_cols', []))

    cat_cols = [col for col in error_df.columns if col not in exclude_cols and col not in dropped_by_config and error_df[col].dtype == 'object']
    num_cols = [col for col in error_df.columns if col not in exclude_cols and col not in dropped_by_config and error_df[col].dtype in [np.number, 'int64', 'float64']]

    features_to_analyze = cat_cols[:top_n_features]
    n_features = len(features_to_analyze)

    if n_features == 0:
        print("Категориальных признаков для анализа не найдено.")
    else:
        print(f"=== ЗАПУСК ПОСТРОЕНИЯ СВОДНОГО ОТЧЕТА ТРЕНДОВ ({n_features} ФИЧЕЙ) ===")
        
        # Динамически рассчитываем сетку: 2 графика в ряд
        n_cols = 2
        n_rows = int(np.ceil(n_features / n_cols))
        
        # Создаем ОДНО ОБЩЕЕ ПОЛОТНО
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes = axes.flatten() # Делаем массив одномерным для удобства итерации

        # ============================================================================
        # ВЕТКА КЛАССИФИКАЦИИ
        # ============================================================================
        if task_type in ['binary', 'multiclass']:
            error_df['Is_Error'] = (error_df['Actual'] != error_df['Predicted']).astype(int)
            error_rates = {}
            
            for i, col in enumerate(features_to_analyze):
                top_categories = error_df[col].value_counts().index[:top_k_categories]
                
                plot_df = error_df.copy()
                plot_df[col] = plot_df[col].apply(lambda x: x if x in top_categories else '... OTHER ...')
                plot_order = list(top_categories) + ['... OTHER ...'] if '... OTHER ...' in plot_df[col].values else list(top_categories)
                plot_df[col] = pd.Categorical(plot_df[col], categories=plot_order, ordered=True)
                
                # Считаем точные рейты ошибок для аннотаций
                col_error_rates = error_df.groupby(col)['Is_Error'].mean() * 100
                
                # Рисуем на конкретном сабплоте сетки
                sns.histplot(
                    data=plot_df, x=col, hue='Is_Error', multiple='fill', 
                    shrink=0.8, palette={0: '#2ed573', 1: '#ff4757'}, ax=axes[i], legend=True if i == 0 else False
                )
                
                # Добавляем текстовые подписи процента ошибок прямо НАД каждым столбцом
                axes[i].set_title(f"Доля ошибок по фиче: {col}", fontsize=12, fontweight='bold')
                axes[i].set_ylabel("Доля ответов")
                axes[i].set_xlabel("")
                axes[i].axhline(0.5, color='black', linestyle=':', alpha=0.3)
                axes[i].tick_params(axis='x', rotation=35)
                
                # Специфический сдвиг подписей для выравнивания
                for tick in axes[i].get_xticklabels():
                    tick.set_horizontalalignment('right')

        # ============================================================================
        # ВЕТКА РЕГРЕССИИ
        # ============================================================================
        elif task_type == 'regression':
            for i, col in enumerate(features_to_analyze):
                top_categories = error_df[col].value_counts().index[:top_k_categories]
                
                plot_df = error_df.copy()
                plot_df[col] = plot_df[col].apply(lambda x: x if x in top_categories else '... OTHER ...')
                plot_order = list(top_categories) + ['... OTHER ...'] if '... OTHER ...' in plot_df[col].values else list(top_categories)
                plot_df[col] = pd.Categorical(plot_df[col], categories=plot_order, ordered=True)
                
                # Строим Boxplot
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

        # Прячем неиспользованные ячейки в сетке графиков, если фичей нечетное количество
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Наводим красоту и сохраняем всё полотно целиком
        plt.tight_layout()
        output_plot_path = reports_dir / f"features_error_trends_v{cfg.data.tabular.preprocessing_version}.png"
        plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
        #plt.show()
        
        print(f"✅ Сводное полотно трендов успешно сохранено в: {output_plot_path}")

        return fig



def worst_preds(error_df: pd.DataFrame, cfg, project_root, top_features_count:int = 5):
    """Выполняет глубокий профайлинг и досье для худших предсказаний модели.

    Для классификации выделяет топ-5 False Positive (самые уверенные ошибки) и топ-5
    False Negative. Для регрессии находит топ-10 выбросов по Absolute Error.
    Печатает текстовые карточки объектов с их физическими признаками в консоль
    и дублирует отчет в текстовый файл (.txt) в директорию отчетов.

    Args:
        error_df (pd.DataFrame): Датафрейм анализа ошибок.
        top_features_count (int): Количество бизнес-фичей, выводимых в текстовое досье объектов.
    """
    task_type = cfg.task_type
    run_name = cfg.run_name
    reports_dir = Path(project_root / cfg.paths.reports_dir / run_name)

    print(f"=== ЗАПУСК ЛОКАЛЬНОГО АНАЛИЗА ОШИБОК | РЕЖИМ: {task_type.upper()} ===")

    # Автоматически определяем доступные реальные фичи (исключаем служебные метки и дропы)
    internal_cols = ['Actual', 'Predicted', 'Is_Error', 'Probability', 'Confidence_Mistake', 'Prediction_Type', 'Is_Worst']
    dropped_by_config = list(cfg.data.tabular.get('drop_cols', []))
    available_features = [col for col in error_df.columns if col not in internal_cols and col not in dropped_by_config]
    features_to_print = available_features[:top_features_count]

    # Инициализируем переменные путей, чтобы они гарантированно существовали при сохранении
    plot_path = reports_dir / f"worst_errors_plots_v{cfg.data.tabular.preprocessing_version}.png"
    text_path = reports_dir / f"worst_errors_profile_v{cfg.data.tabular.preprocessing_version}.txt"

    # ============================================================================
    # СЦЕНАРИЙ 1: БИНАРНАЯ И МНОГОКЛАССОВАЯ КЛАССИФИКАЦИЯ
    # ============================================================================
    if task_type in ['binary', 'multiclass']:
        if 'Probability' in error_df.columns and task_type == 'binary':
            error_df['Confidence_Mistake'] = np.where(
                error_df['Actual'] == 1, 1 - error_df['Probability'], error_df['Probability']      
            )
            
            # 1. Строим "простыню" из двух графиков классификации
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
            #plt.show()

            # 2. Формируем текстовый профиль для классификации
            report_text = f"=== ОТЧЕТ ПО ХУДШИМ ОШИБКАМ КЛАССИФИКАЦИИ ===\n"
            report_text += f"Версия фичей: {cfg.data.tabular.features_version} | Версия препроцессинга: {cfg.data.tabular.preprocessing_version}\n"
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
            # Если это многокласс или нет вероятностей
            worst_cases = error_df[error_df['Is_Error'] == 1].head(10)
            report_text = "=== ТОП-10 ОШИБОК КЛАССИФИКАЦИИ (БЕЗ ВЕРОЯТНОСТЕЙ) ===\n"
            for idx, row in worst_cases.iterrows():
                report_text += f"  • Строка [ID {idx}] | Факт: {row['Actual']} | Предикт: {row['Predicted']}\n"
                for f in features_to_print:
                    report_text += f"    - {f}: {row[f]}\n"
                report_text += "  " + "-"*45 + "\n"

    # ============================================================================
    # СЦЕНАРИЙ 2: РЕГРЕССИЯ
    # ============================================================================
    elif task_type == 'regression':
        worst_indices = error_df.sort_values(by='AbsError', ascending=False).head(10).index
        error_df['Is_Worst'] = error_df.index.isin(worst_indices).astype(int)
        
        # 1. Строим "простыню" из двух графиков регрессии
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
        #plt.show()

        # 2. Формируем текстовый профиль для регрессии
        report_text = f"=== ОТЧЕТ ПО КАТАСТРОФИЧЕСКИМ ВЫБРОСАМ РЕГРЕССИИ ===\n"
        report_text += f"Версия фичей: {cfg.data.tabular.features_version} | Версия препроцессинга: {cfg.data.tabular.preprocessing_version}\n"
        report_text += "="*70 + "\n\n"
        
        report_text += " ТОП-10 ХУДШИХ ПРОГНОЗОВ (Где модель промахнулась сильнее всего):\n"
        for idx, row in worst_cases.iterrows():
            report_text += f"  • Строка [ID {idx}] | Реально: {row['Actual']:.2f} | Предсказано: {row['Predicted']:.2f} | Ошибка: {row['Error']:.2f}\n"
            for f in features_to_print:
                report_text += f"    - {f}: {row[f]}\n"
            report_text += "  " + "-"*45 + "\n"

    # ============================================================================
    # ФИНАЛЬНЫЙ ВЫВОД И ЛОКАЛЬНОЕ СОХРАНЕНИЕ
    # ============================================================================
    # Печатаем карточки прямо в консоль ноутбука для удобного чтения
    print(report_text)

    # Сохраняем текстовый файл в локальную директорию reports/
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"✅ Локальные отчеты успешно сгенерированы и сохранены:")
    print(f"   - Графики сохранены в: {plot_path}")
    print(f"   - Текстовое досье сохранено в: {text_path}")

    return fig



def feature_importance(fi_df:pd.DataFrame, cfg, project_root, features_count:int = 15):
    figures = {}
    task_type = cfg.task_type
    run_name = cfg.run_name
    reports_dir = Path(project_root / cfg.paths.reports_dir / run_name)

    fig_top = plt.figure()
    sns.barplot(
        data=fi_df.head(features_count),  # Показываем топ-15
        x='Importance',
        y='Feature',
        palette='viridis',
        alpha=cfg.logging.plots.alpha
    )
    plt.title(f"Топ-15 самых важных признаков ({cfg.model.name})")
    plt.tight_layout()
    figures['top_importance'] = fig_top
    output_plot_path = reports_dir / f"features_importance_top{features_count}_{cfg.data.tabular.preprocessing_version}.png"
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')

    worst_features = fi_df.sort_values(by='Importance', ascending=True).head(features_count)

    fig_worst = plt.figure()
    sns.barplot(
        data=worst_features,  # Теперь здесь лежат самые бесполезные фичи сверху вниз
        x='Importance',
        y='Feature',
        palette='viridis',
        alpha=cfg.logging.plots.alpha
    )
    plt.title(f"Топ-{features_count} самых худших признаков ({cfg.model.name})")
    plt.tight_layout()
    figures['worst_importance'] = fig_worst
    output_plot_path = reports_dir / f"features_importance_worst_{features_count}_{cfg.data.tabular.preprocessing_version}.png"
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    #plt.show()

    return figures

def plot_sweetviz_style(df: pd.DataFrame, feature: str, target: str = 'target'):
    """
    Отрисовывает график распределения бинов и линию таргета в стиле Sweetviz.
    """
    # 1. Считаем агрегации
    stats = df.groupby(feature).agg(
        Count=(target, 'size'),
        Target_Sum=(target, 'sum'),
        Target_Mean=(target, 'mean')
    ).reset_index()
    
    # --- ДОБАВЛЯЕМ ПРОВЕРКУ ---
    if stats.empty:
        print(f"ПРОПУСК: Признак {feature} пуст после фильтрации. График не строится.")
        return
    # ---------------------------

    # Считаем проценты и сортируем...
    stats['Percent'] = (stats['Count'] / len(df)) * 100
    stats = stats.sort_values('Count', ascending=False).reset_index(drop=True)
    
    # 2. Настройка графика
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Позиции по оси Y (чтобы линия и бары точно совпали)
    y_pos = np.arange(len(stats))
    
    # Горизонтальный Bar Chart (Доля бина в выборке)
    # Цвет как у Sweetviz (светло-синий)
    ax1.barh(y_pos, stats['Percent'], color='#1E88E5', height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(stats[feature])
    ax1.invert_yaxis()  # Самый частый бин будет сверху
    ax1.set_xlabel('Доля в датасете (%)')
    ax1.set_xlim(0, max(stats['Percent']) * 1.1) # Немного отступа справа
    
    # Ломаная линия таргета (Конверсия)
    ax2 = ax1.twiny()
    # Цвет линии таргета (темно-синий)
    ax2.plot(stats['Target_Mean'] * 100, y_pos, color='#0D47A1', marker='o', linewidth=2, markersize=6)
    ax2.set_xlabel('Конверсия в таргет (%)', color='#0D47A1')
    
    # Если таргет очень маленький, делаем шкалу от 0, чтобы линия не прыгала
    max_target = max(stats['Target_Mean'] * 100)
    ax2.set_xlim(0, max(max_target * 1.2, 5.0)) 
    
    plt.title(f'Анализ признака: {feature}', pad=20)
    
    # Убираем лишние рамки для красоты
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Отрисовка таблицы под графиком
    print(f"{'TOP CATEGORIES':<20} | {'Count':<10} | {'%':<6} | {'Target Cnt':<10} | {'Target %':<8}")
    print("-" * 65)
    
    for _, row in stats.iterrows():
        print(f"{row[feature]:<20} | {int(row['Count']):<10} | {row['Percent']:>4.1f}% | {int(row['Target_Sum']):<10} | {row['Target_Mean']*100:>6.1f}%")
    
    # Итоги
    total_target_pct = df[target].mean() * 100
    print("-" * 65)
    print(f"{'ALL':<20} | {len(df):<10} | 100.0% | {int(df[target].sum()):<10} | {total_target_pct:>6.1f}%\n")