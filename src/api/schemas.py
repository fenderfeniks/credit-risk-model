from __future__ import annotations

import json
import logging
from pathlib import Path
from pydantic import BaseModel, create_model
from typing import Any, Optional, Union, List, Dict

logger = logging.getLogger(__name__)


def build_dynamic_request_model(schema_path: Path):
    """
    Фабрика. Читает json со схемой колонок и возвращает готовый Pydantic класс.
    Путь к схеме передаётся явно из main.py (единственного места, где
    собираются cfg и PROJECT_ROOT), а не вычисляется здесь повторно.
    """
    if not schema_path.exists():
        logger.warning(
            f"[SCHEMAS] Файл схемы не найден: {schema_path}. "
            f"Обучите модель командой 'python main.py mode=train'. "
            f"Эндпоинт /predict работать не будет."
        )
        return create_model("PredictionRequest", placeholder=(str, "Модель еще не обучена"))

    with open(schema_path, "r") as f:
        feature_schema = json.load(f)

    # Маппинг типов Pandas/Numpy в стандартные типы Python
    type_mapping = {
        "int64": int,
        "int32": int,
        "float64": float,
        "float32": float,
        "object": str,
        "bool": bool,
        "category": str
    }

    fields: Dict[str, Any] = {}
    for col_name, dtype in feature_schema.items():
        python_type = type_mapping.get(dtype, Any)
        fields[col_name] = (python_type, ...)

    return create_model("PredictionRequest", **fields)


# ============================================================
# ГОТОВЫЕ ЭКСПОРТЫ ДЛЯ API
# ============================================================
# Ответ от API всегда статичен, поэтому пишем его руками
class PredictionResponse(BaseModel):
    prediction: Union[float, int, str]
    probability: Optional[float] = None
    # Приятный бонус: отдаем весь массив вероятностей
    probabilities: Optional[List[float]] = None