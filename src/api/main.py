from __future__ import annotations

import time
import logging
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
from prometheus_client import make_asgi_app, Counter, Histogram

from src.api.schemas import build_dynamic_request_model, PredictionResponse
from src.api.dependencies import verify_api_key, get_ml_model, get_preprocessor
from src.core.utils import load_hydra_config, PROJECT_ROOT
from src.core.models import get_model

logger = logging.getLogger(__name__)

PREDICTION_COUNTER = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['predicted_class']
)

INFERENCE_LATENCY = Histogram(
    'ml_inference_latency_seconds',
    'Time spent processing the prediction',
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
)

PREDICTION_CONFIDENCE = Histogram(
    'ml_prediction_confidence',
    'Probability of the predicted class',
    buckets=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
)

# ============================================================
# 1. ЗАГРУЗКА КОНФИГА — ЕДИНСТВЕННЫЙ РАЗ, ЕДИНСТВЕННЫЙ ИСТОЧНИК
# ============================================================
try:
    cfg = load_hydra_config("config")
    API_TITLE = cfg.project_name
    MODEL_VERSION = cfg.model.model_version
    API_VERSION = MODEL_VERSION
    API_DESC = f"Микросервис инференса. Модель: {cfg.model.name}"
    HAS_CONFIG = True
except Exception as e:
    logger.warning(f"Не удалось загрузить конфиг для метаданных: {e}")
    cfg = None
    API_TITLE = "ML Production API"
    MODEL_VERSION = "1.0.0"
    API_VERSION = "1.0.0"
    API_DESC = "Ожидание загрузки модели..."
    HAS_CONFIG = False

# PredictionRequest зависит от схемы фичей, которая зависит от cfg и PROJECT_ROOT.
# Строим её здесь, передавая cfg/PROJECT_ROOT явно, а не позволяя schemas.py
# читать конфиг самостоятельно.
if HAS_CONFIG:
    SCHEMA_PATH = PROJECT_ROOT / cfg.paths.models_dir / f"feature_schema_v{cfg.data.tabular.features_version}.json"
else:
    SCHEMA_PATH = PROJECT_ROOT / "models" / "feature_schema_v1.0.0.json"

PredictionRequest = build_dynamic_request_model(SCHEMA_PATH)


# ============================================================
# 2. LIFESPAN — кладём cfg и PROJECT_ROOT в app.state, чтобы все
#    остальные модули (dependencies, эндпоинты) брали их оттуда,
#    а не дёргали load_hydra_config() повторно.
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Загрузка моделей в память...")

    app.state.cfg = cfg
    app.state.project_root = PROJECT_ROOT

    if not HAS_CONFIG:
        logger.error("Запуск без конфигов. Инференс будет возвращать ошибку 503.")
        app.state.model = None
        app.state.preprocessor = None
        yield
        return

    try:
        models_dir = PROJECT_ROOT / cfg.paths.models_dir

        prep_ver = cfg.data.tabular.preprocessing_version
        model_ver = cfg.model.model_version

        # Загрузка препроцессора
        prep_path = models_dir / f"preprocessing_v{prep_ver}.pkl"
        app.state.preprocessor = joblib.load(prep_path)

        # Загрузка модели — cfg и PROJECT_ROOT передаём явно, как и везде в проекте
        model_wrapper = get_model(cfg, PROJECT_ROOT)
        model_path = models_dir / f"{cfg.model.name}_v{model_ver}{model_wrapper.file_extension}"

        model_wrapper.load(str(model_path))
        app.state.model = model_wrapper

        logger.info("API успешно инициализировано.")
    except Exception as e:
        logger.error(f"Ошибка загрузки весов: {e}")
        app.state.model = None
        app.state.preprocessor = None

    yield

    logger.info("Остановка сервера. Очистка памяти...")
    app.state.model = None
    app.state.preprocessor = None


# ============================================================
# 3. ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ
# ============================================================
app = FastAPI(
    title=API_TITLE,
    description=API_DESC,
    version=API_VERSION,
    lifespan=lifespan
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ============================================================
# 4. ГЛАВНЫЙ ЭНДПОИНТ ИНФЕРЕНСА
# ============================================================
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest,
            api_key: str = Depends(verify_api_key),
            model_wrapper=Depends(get_ml_model),
            preprocessor=Depends(get_preprocessor)):
    """
    ВАЖНО: Функция объявлена как `def`, а не `async def`.
    FastAPI запустит её в Threadpool, чтобы Pandas не заблокировал сервер.
    """
    if model_wrapper is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена или отключена.")

    start_time = time.perf_counter()

    try:
        df_input = pd.DataFrame([request.model_dump()])
        df_clean = preprocessor.transform(df_input)

        pred_value = model_wrapper.predict(df_clean)[0]
        if isinstance(pred_value, np.generic):
            pred_value = pred_value.item()

        probability = None
        probabilities = None
        try:
            if hasattr(model_wrapper, 'predict_proba'):
                proba_array = model_wrapper.predict_proba(df_clean)[0]
                probability = float(np.max(proba_array))
                probabilities = proba_array.tolist()
                PREDICTION_CONFIDENCE.observe(probability)
        except NotImplementedError:
            pass

        PREDICTION_COUNTER.labels(predicted_class=str(pred_value)).inc()

        process_time = time.perf_counter() - start_time
        INFERENCE_LATENCY.observe(process_time)

        return PredictionResponse(
            prediction=pred_value,
            probability=probability,
            probabilities=probabilities
        )
    except Exception as e:
        logger.error(f"Ошибка во время инференса: {str(e)} | Данные: {request.model_dump()}")
        raise HTTPException(status_code=500, detail="Ошибка обработки предсказания внутри модели.")


# ============================================================
# 5. ЭНДПОИНТ ПРОВЕРКИ ЗДОРОВЬЯ (Healthcheck)
# ============================================================
@app.get("/health", tags=["System"])
def health_check(request: Request):
    if not request.app.state.model:
        return {"status": "degraded", "detail": "Модель не загружена"}
    return {"status": "ok", "model_version": MODEL_VERSION}