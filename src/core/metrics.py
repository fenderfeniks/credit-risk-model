from __future__ import annotations

import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report
)


logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, y_prob=None, task_type: str = "binary") -> dict:
    """
    Универсальная считалка метрик для любой модели.

    :param y_true: Истинные значения (классы или числа)
    :param y_pred: Предсказанные значения (классы или числа)
    :param task_type: 'binary', 'multiclass' или 'regression'
    :param y_prob: (Опционально) Вероятности предсказаний для ROC-AUC
    """
    metrics = {}

    if task_type in ['binary', 'multiclass']:

        print(classification_report(y_true, y_pred))
        metrics['accuracy'] = round(accuracy_score(y_true, y_pred), 4)
        # weighted отлично работает и для сбалансированных, и для дисбалансных выборок
        metrics['f1_weighted'] = round(f1_score(y_true, y_pred, average='weighted'), 4)

        # === ЛОГИКА ROC-AUC ===
        if y_prob is not None:
            try:
                if task_type == 'binary':
                    # Защита: иногда модели возвращают 1D массив, иногда 2D (по колонке на класс)
                    # Нам нужна вероятность положительного класса (обычно индекс 1)
                    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                        prob_pos = y_prob[:, 1]
                    else:
                        prob_pos = y_prob
                    metrics['roc_auc'] = round(roc_auc_score(y_true, prob_pos), 4)

                elif task_type == 'multiclass':
                    # Для мультикласса нужна 2D матрица вероятностей всех классов
                    metrics['roc_auc_ovr'] = round(
                        roc_auc_score(y_true, y_prob, multi_class='ovr'), 
                        4
                    )
            except ValueError as e:
                # roc_auc может упасть, если в валидационном батче представлены не все классы
                logger.warning(f"Не удалось рассчитать ROC-AUC (возможно, не все классы в выборке): {e}")

    elif task_type == 'regression':
        metrics['rmse'] = round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)
        metrics['mae'] = round(mean_absolute_error(y_true, y_pred), 4)
        metrics['r2'] = round(r2_score(y_true, y_pred), 4)

    print(metrics)
    return metrics