# Credit Risk Model

Промышленный ML-пайплайн для кредитного скоринга: предсказание вероятности дефолта заемщика на основе агрегированной кредитной истории. Проект построен как production-grade система полного цикла — от SQL-агрегации сырых данных до FastAPI-сервиса с мониторингом и оркестрацией переобучения в Airflow.

![Python](https://img.shields.io/badge/python-3.10-blue)
![CatBoost](https://img.shields.io/badge/model-CatBoost-yellow)
![XGBoost](https://img.shields.io/badge/model-XGBoost-4285F4)
![LightGBM](https://img.shields.io/badge/model-LightGBM-9ACD32)
![Hydra](https://img.shields.io/badge/config-Hydra-89b8cd)
![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)
![Optuna](https://img.shields.io/badge/tuning-Optuna-6dc7e6)
![SHAP](https://img.shields.io/badge/explainability-SHAP-ff69b4)
![DuckDB](https://img.shields.io/badge/dev-DuckDB-FFF000)
![PySpark](https://img.shields.io/badge/prod-PySpark-E25A1C)
![FastAPI](https://img.shields.io/badge/serving-FastAPI-009688)
![Prometheus](https://img.shields.io/badge/monitoring-Prometheus-E6522C)
![Airflow](https://img.shields.io/badge/orchestration-Airflow-017CEE)
![Docker](https://img.shields.io/badge/deploy-Docker-2496ED)
![uv](https://img.shields.io/badge/package_manager-uv-DE5FE9)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Содержание

- [Задача](#задача)
- [Ключевые результаты](#ключевые-результаты)
- [Архитектура пайплайна](#архитектура-пайплайна)
- [Структура проекта](#структура-проекта)
- [Стек технологий](#стек-технологий)
- [Установка и запуск](#установка-и-запуск)
- [Конфигурация (Hydra)](#конфигурация-hydra)
- [Пайплайн признаков](#пайплайн-признаков)
- [Эволюция модели: от baseline до прода](#эволюция-модели-от-baseline-до-прода)
- [Интерпретируемость (SHAP)](#интерпретируемость-shap)
- [Deployment: API, Docker, Airflow](#deployment-api-docker-airflow)
- [Тестирование](#тестирование)
- [Дальнейшее развитие](#дальнейшее-развитие)

---

## Задача

Бинарная классификация: по агрегированной кредитной истории клиента (типы кредитов, лимиты, просрочки, платежное поведение) предсказать вероятность дефолта (`flag = 1`).

Особенности задачи, определившие архитектуру пайплайна:

- **Сильный дисбаланс классов** — соотношение "нет дефолта" к "дефолт" составляет примерно **31:1**.
- **Сырые данные — построчная история платежей**, а не готовая витрина: каждый кредит клиента представлен несколькими строками с кодами статусов платежей, которые нужно агрегировать до уровня клиента.
- **Асимметричная стоимость ошибок**: пропуск реального дефолта (False Negative) в кредитном скоринге обходится бизнесу дороже, чем ложная тревога на благонадежном клиенте (False Positive) — это напрямую повлияло на выбор целевой метрики и стратегию тюнинга.
- **Переход DEV → PROD** с ростом объема данных в разы (175k → 420k+ клиентов), что потребовало отдельной адаптации гиперпараметров.

---

## Ключевые результаты

Итоговая модель — **CatBoostClassifier**, обученная на PROD-выборке (420k+ клиентов) с гиперпараметрами, адаптированными под возросший объем данных (см. [итерацию v1.6](#v16-адаптация-гиперпараметров-под-prod-масштаб)).

| Метрика | Dirty Baseline | Финальная модель (v1.6) |
|---|---|---|
| ROC-AUC | 0.641 | **0.743** |
| Accuracy | 0.724 | **0.917** |
| F1-score (weighted) | 0.819 | **0.928** |
| Recall (класс 1, дефолт) | 0.47 | 0.25 |
| Precision (класс 1, дефолт) | 0.04 | 0.14 |

> Recall и Precision целевого класса не улучшались монотонно на всем пути — это осознанный trade-off. Подробности и промежуточные конфигурации (включая модель с Recall дефолтов 71–80%) разобраны в разделе [«Эволюция модели»](#эволюция-модели-от-baseline-до-прода).

**Confusion Matrix — Dirty Baseline:**

![Confusion Matrix Dirty Baseline](reports/dirty_baseline/confusion_matrix.png)
*Плейсхолдер: `reports/dirty_baseline/confusion_matrix.png`*

**Confusion Matrix — финальная модель:**

![Confusion Matrix Final](reports/final/confusion_matrix.png)
*Плейсхолдер: `reports/final/confusion_matrix.png`*

**Top-15 Feature Importance — Dirty Baseline:**

![Feature Importance Dirty Baseline](reports/dirty_baseline/feature_importance_top15.png)
*Плейсхолдер: `reports/dirty_baseline/feature_importance_top15.png`*

**Top-15 Feature Importance — финальная модель:**

![Feature Importance Final](reports/final/feature_importance_top15.png)
*Плейсхолдер: `reports/final/feature_importance_top15.png`*

**ROC-AUC Curve — Dirty Baseline:**

![ROC-AUC Dirty Baseline](reports/dirty_baseline/roc_auc_curve.png)
*Плейсхолдер: `reports/dirty_baseline/roc_auc_curve.png`*

**ROC-AUC Curve — финальная модель:**

![ROC-AUC Final](reports/final/roc_auc_curve.png)
*Плейсхолдер: `reports/final/roc_auc_curve.png`*

---

## Архитектура пайплайна

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Raw Data    │────▶│   SQL Aggregation  │────▶│  Feature          │────▶│  Model      │
│  (DuckDB/    │     │   (DuckDB dev /    │     │  Engineering +    │     │  (CatBoost/ │
│   PySpark)   │     │    PySpark prod)   │     │  Preprocessing    │     │   XGB/LGBM) │
└──────────────┘     └───────────────────┘     └──────────────────┘     └──────┬──────┘
                                                                                │
                     ┌──────────────────────────────────────────────────────────┘
                     ▼
        ┌────────────────────┐     ┌───────────────┐     ┌──────────────────┐
        │  MLflow Tracking /  │     │  FastAPI       │     │  Prometheus /     │
        │  Optuna Tuning      │     │  Serving       │────▶│  Grafana мониторинг│
        └────────────────────┘     └───────────────┘     └──────────────────┘
                     ▲
                     │
        ┌────────────┴────────────┐
        │  Airflow DAGs            │
        │  (DockerOperator):       │
        │  retrain_pipeline,       │
        │  batch_inference,        │
        │  deploy_model            │
        └──────────────────────────┘
```

Оркестрация всего цикла (train / tune / evaluate / inference) идет через единую точку входа `main.py`, управляемую конфигами Hydra — режим выбирается флагом `mode=...`, без дублирования кода между DEV и PROD.

---

## Структура проекта

```
credit-risk-model/
├── main.py                     # Точка входа — оркестратор режимов (train/tune/evaluate/inference)
├── configs/                    # Hydra-конфиги (композируются в config.yaml)
│   ├── config.yaml
│   ├── data/                   # Версии агрегации/фичей/препроцессинга, сплиты
│   ├── model/                  # catboost.yaml, xgboost.yaml, lightgbm.yaml
│   ├── training/                
│   ├── paths/
│   ├── env/                    # dev.yaml / prod.yaml — переключение DuckDB ↔ PySpark
│   ├── security/
│   ├── logging/
│   └── deploy/                 # docker-compose.yml, .env
├── sql/aggregate/               # SQL-агрегация кредитной истории (DuckDB + PySpark версии)
├── src/
│   ├── core/
│   │   ├── data.py              # DevDuckDBDataSource / ProdSparkDataSource
│   │   ├── features.py          # FeatureEngineer, TabularPreprocessor
│   │   ├── splitting.py         # Группировка сплитов по client_id (anti-leakage)
│   │   ├── stats.py             # GlobalStatCompiler — статистики по train для SQL-инъекций
│   │   ├── pipeline.py          # MLPipeline — оркестратор train/predict/load
│   │   ├── tuner.py             # OptunaTuner
│   │   ├── metrics.py
│   │   ├── artifacts.py         # ArtifactManager (обертка над MLflow)
│   │   └── models/              # CatBoostWrapper, XGBoostWrapper, LightGBMWrapper, PyTorch
│   ├── api/                     # FastAPI-сервис инференса (main.py, schemas.py, dependencies.py)
│   └── eda/                     # ReportBuilder, SHAP-эксплейнер, визуализация
├── dags/                        # Airflow DAGs (DockerOperator): retrain, batch inference, deploy
├── docker/                      # Dockerfile'ы: api / train / airflow / mlflow
├── notebooks/                   # EDA → feature engineering → baseline → error analysis → explainability
├── tests/                       # unit + integration (pytest)
├── Makefile
└── pyproject.toml               # core + extras: api / train / eda
```

---

## Стек технологий

| Категория | Инструменты |
|---|---|
| Модели | CatBoost (production), XGBoost, LightGBM (сравнительный анализ), PyTorch-обертка |
| Обработка данных | DuckDB (dev), PySpark (prod), Pandas |
| Конфигурация | Hydra + OmegaConf |
| Трекинг экспериментов | MLflow |
| Подбор гиперпараметров | Optuna |
| Интерпретируемость | SHAP |
| Serving | FastAPI, Pydantic, Uvicorn |
| Мониторинг | Prometheus |
| Оркестрация | Airflow (DockerOperator) |
| Инфраструктура | Docker, Docker Compose |
| Версионирование данных | DVC |
| Тестирование | Pytest (unit + integration) |
| Пакетный менеджер | uv |

---

## Установка и запуск

### Требования

- Python 3.10 (проект жестко закреплен на `>=3.10, <3.11`)
- Windows (dev-окружение автора, PowerShell) или Linux
- Docker + Docker Compose (для prod-инфраструктуры)

### Локальная установка

```powershell
# Создание окружения и установка зависимостей
uv venv
uv pip install -e ".[dev,eda]"

# Либо через Makefile (pip-эквивалент)
make venv
make install
```

### Запуск через Hydra-режимы

```powershell
# Обучение модели
python -m main mode=train

# Подбор гиперпараметров через Optuna + обучение финальной модели
python -m main mode=tune

# Оценка на отложенной тестовой выборке
python -m main mode=evaluate

# Инференс на новых данных
python -m main mode=inference
```

Или эквивалентно через `Makefile`:

```powershell
make run-train
make run-tune
make run-evaluate
make run-inference
```

### Тесты

```powershell
make test              # unit + integration
make test-unit         # только быстрые unit-тесты
make test-integration  # тяжелые интеграционные тесты
```

### MLflow UI

```powershell
make mlflow
# UI поднимется на http://localhost:5000
```

---

## Конфигурация (Hydra)

Проект использует композицию конфигов Hydra — `configs/config.yaml` собирает дефолты из отдельных групп (`data`, `model`, `training`, `paths`, `env`, `logging`, `security`):

```yaml
defaults:
  - base_config
  - paths: paths
  - data: default
  - security: security
  - model: catboost      # переключение модели: catboost / xgboost / lightgbm
  - training: default
  - logging: default
  - env@_global_: dev    # dev = DuckDB локально, prod = PySpark + партиции
  - _self_
```

Ключевые принципы конфигурации:

- **Раздельное версионирование** — `preprocessing_version`, `features_version`, `aggregation_version` и `model_version` версионируются независимо друг от друга, что позволяет переиспользовать препроцессор между экспериментами с разными моделями.
- **`env: dev/prod`** переключает не только источник данных (DuckDB ↔ PySpark), но и объем выборки (`sample_pct`).
- **Смена модели** — одной строкой `model=xgboost` или `model=lightgbm` в CLI-оверрайде, без изменения кода пайплайна благодаря единому контракту `BaseModelWrapper`.

---

## Пайплайн признаков

### 1. SQL-агрегация (DuckDB / PySpark)

Сырые данные — построчная история по каждому кредиту клиента. Агрегация сворачивает их до одной строки на клиента, с полным устранением утечки данных (Data Leakage) на этапе расчета статистик. Логические блоки признаков:

- **Базовые счетчики** — количество кредитов, плотность кредитной истории, доля закрытых кредитов.
- **Просрочки и утилизация** — доли и флаги отсутствия просрочек на горизонтах 5/30/60/90 дней, утилизация лимитов, соотношение тяжелых просрочек к легким.
- **Платежное поведение** — динамика платежей, частота смены статуса, доли конкретных кодов платежей.
- **Категориальные признаки** — стабильность категорий, сравнение с глобальными модами и редкими бинами.

### 2. Feature Engineering (`FeatureEngineer`)

Кастомный sklearn-трансформер генерирует признаки поверх агрегатов без обращения к сырым последовательностям:

- **Макро-индексы**: `overall_typicalness_score`, `overall_anomaly_score`.
- **Бизнес-флаги экстремумов**: `is_absolutely_perfect_payer`, `is_chronic_defaulter`.
- **Категориальные склейки и переходы**: `cat_last_util_and_time`, `cat_limit_transition`, `cat_status_transition`, `cat_util_and_history`, `cat_diversity_profile`.

### 3. Препроцессинг (`TabularPreprocessor`)

- Удаление "мусорных" колонок (пропуски > 90%, константы > 99.9%).
- Схлопывание хвостов редких категорий.
- Импутация (median для числовых, "Unknown" для категориальных).
- Обработка выбросов (IQR/Z-score).
- Guard по кардинальности — предотвращает превращение ID-подобных числовых полей в псевдо-категории.

### 4. Anti-leakage сплиттинг

Train/val/test разбиваются на уровне `client_id`, чтобы кредиты одного клиента не попадали одновременно в разные выборки.

---

## Эволюция модели: от baseline до прода

Ниже — краткая хронология итераций, полный разбор каждой — в `notebooks/` и в отчетах MLflow.

### Dirty Baseline

Модель на «сырых» признаках без агрегации (только merge с таргетом), без Feature Engineering. Пайплайн предобработки отработал вхолостую — данные изначально не содержали пропусков/констант, требующих фильтрации.

- **Конфигурация**: CatBoost, дефолтные гиперпараметры, `auto_class_weights='Balanced'`.
- **Результат**: ROC-AUC 0.641, Accuracy 0.724, Recall (класс 1) 0.47, Precision (класс 1) 0.04.
- Топ-3 признака по важности: `pre_loans_credit_cost_rate`, `pre_loans_credit_limit`, `enc_loans_credit_type`.
- Изменение типов данных с числовых на категориальные дало прирост: ROC-AUC 0.6607, Accuracy 0.7178.

### v1.0 — Полноценная SQL-агрегация

Первый осмысленный шаг после dirty baseline: полная агрегация кредитной истории на уровне клиента с устранением Data Leakage.

- **Результат**: ROC-AUC 0.7492, Accuracy 0.70, Recall (класс 1) 0.67.
- Балансировка классов дает ожидаемый эффект: модель хорошо находит дефолты, но ценой ~29% ложных срабатываний.

### v1.1 — Feature Engineering

Добавлен `FeatureEngineer` с макро-индексами типичности/аномалий, бизнес-флагами и категориальными склейками/переходами.

- **Результат**: ROC-AUC 0.7533 (+0.0041), Accuracy 0.7131 (+0.0131).
- Новые признаки (`overall_typicalness_score`, `cat_util_and_history`, `cat_diversity_profile`) закрепились в топе важности.

### v1.2 — Сравнение архитектур (CatBoost vs XGBoost vs LightGBM)

Оценка "out-of-the-box" способности альтернативных фреймворков справляться с дисбалансом при дефолтных гиперпараметрах.

| Модель | ROC-AUC | Recall (класс 1) | Precision (класс 1) |
|---|---|---|---|
| CatBoost | 0.7533 | **0.66** | 0.07 |
| XGBoost | 0.6569 | 0.03 | 0.13 |
| LightGBM | 0.7173 | 0.00 | 0.00 |

Несмотря на явные веса (`scale_pos_weight`, `is_unbalance`), XGBoost и LightGBM не справились с поиском дефолтов "из коробки" — CatBoost выбран ядром пайплайна.

### v1.3 — Тюнинг гиперпараметров CatBoost (Optuna, 50 trials)

Отказ от жесткого `auto_class_weights: 'Balanced'` в пользу динамического подбора `scale_pos_weight` (диапазон 15–60).

- **Оптимальные параметры**: `depth=4`, `min_data_in_leaf=2`, `learning_rate≈0.0736`, `scale_pos_weight=37.66`.
- **Результат**: ROC-AUC 0.7618 (🏆 +0.0085), Recall (класс 1) 0.71 (🏆 +0.05), Accuracy 0.6758 (-0.0373).
- Trade-off: рост Recall дефолтов ценой падения Accuracy — осознанное решение в пользу бизнес-приоритета (пропуск дефолта дороже ложной тревоги).

### v1.4 — Интерпретируемость (SHAP)

SHAP-анализ затюненной модели v1.3 для понимания направленности влияния признаков и природы ложноположительных срабатываний. Подробности — в разделе [«Интерпретируемость»](#интерпретируемость-shap).

### v1.5 — Масштабирование на PROD (PySpark, 420k+ клиентов)

Переход с DEV-выборки (DuckDB, ~175k клиентов) на PROD-архитектуру: ETL на PySpark, объем выборки увеличен до 20% всего пула (420 210 клиентов). Гиперпараметры взяты без изменений из v1.3.

- **Результат**: Recall (класс 1) 0.80 (🚀 +0.09), но Accuracy рухнула до 0.5565 (-0.1193), ROC-AUC 0.7410.
- **Диагноз — Data Shift**: гиперпараметры, затюненные на малом сэмпле (`scale_pos_weight≈37.66`), оказались слишком агрессивными для возросшего и более разнообразного объема данных — модель предсказывала дефолт почти в половине случаев.

### v1.6 — Адаптация гиперпараметров под PROD-масштаб

Повторный тюнинг Optuna (25 trials) с зафиксированными параметрами сходимости (`learning_rate`, `l2_leaf_reg`, `subsample`) и расширенным пространством поиска для структуры деревьев и баланса классов:

- `depth: [4, 8]`, `min_data_in_leaf: [10, 200]`, `scale_pos_weight: [10.0, 40.0]`.

**Итоговая конфигурация**: `depth=5`, `min_data_in_leaf=144`, `scale_pos_weight=10.24` (радикальное снижение с 37.66).

- **Результат**: Accuracy 0.9166, F1-weighted 0.9283, ROC-AUC 0.7430 (стабилен).
- Optuna прижала `scale_pos_weight` почти к нижней границе диапазона — модель вернулась к адекватному скорингу вместо "блокировки всех подряд".
- Trade-off: Recall дефолтов упал до 0.25, зато Precision вырос более чем в два раза (до 0.14).
- **Ключевой инсайт**: ROC-AUC остался стабильным на всем пути v1.3 → v1.6 несмотря на резкие скачки Accuracy/Recall — модель стабильно хорошо ранжирует клиентов по риску, а колебания метрик классификации вызваны исключительно порогом отсечения (threshold = 0.5). Следующий логичный шаг — тюнинг classification threshold вместо жесткой перебалансировки весов.

Эта итерация — текущая production-конфигурация (`configs/model/catboost.yaml`, `model_version: 0.0.2`).

---

## Интерпретируемость (SHAP)

Для прозрачности решений модели (важно для комплаенса и кредитного комитета) применен SHAP-анализ.

- **Глобальная важность**: `pre_loans_credit_cost_rate` (стоимость кредита) — ключевой драйвер риска; доля своевременных платежей (`paym_row_share_code_0_mean`) — второй по значимости признак, с пороговым нелинейным эффектом (риск скачкообразно растет после определенного порога).
- **Анализ False Positives (SHAP Heatmap)**: ложные срабатывания формируются из суперпозиции нескольких умеренно-рисковых факторов, которые из-за балансировки весов "перевешивают" финальное предсказание — модель "перестраховывается" на пограничных клиентах.
- **Локальная интерпретация (Waterfall/Force Plot)**: пошаговая декомпозиция скорингового балла конкретного клиента на базовое значение и вклад каждого признака — готова к использованию в кредитном комитете.
- **Worst Errors Analysis**: модель чаще всего "уверенно" ошибается на классе 0, что подтверждает необходимость перехода от `auto_class_weights` к тонкой настройке порога отсечения на этапе пост-процессинга.

Полный SHAP-разбор — в `notebooks/5.0-explainability.ipynb` и `src/eda/shap_explainer.py`.

---

## Deployment: API, Docker, Airflow

### FastAPI Serving

`src/api/main.py` — микросервис инференса:

- Динамическая Pydantic-схема запроса строится из сохраненного `feature_schema_v{version}.json`.
- Модель и препроцессор грузятся один раз при старте (`lifespan`), не на каждый запрос.
- `/predict` — синхронный `def`-эндпоинт (FastAPI сам уводит его в threadpool, чтобы Pandas не блокировал event loop).
- `/health` — healthcheck с версией модели.
- `/metrics` — Prometheus-метрики: счетчик предсказаний по классам, латентность инференса, распределение confidence.

### Docker-образы

Раздельные образы под разные роли, без пересекающихся зависимостей (`pyproject.toml` extras):

- **`api`** — core + FastAPI/Prometheus, легкий образ для serving и batch-инференса.
- **`train`** — core + MLflow/Optuna/PySpark/matplotlib, для train/tune/evaluate.
- **`airflow`** — намеренно без ML-зависимостей: только оркестрация через `DockerOperator`.
- **`mlflow`** — сервер трекинга экспериментов.

### Airflow DAGs

- **`retrain_pipeline`** — `train → evaluate → manual gate`. Автоматического деплоя по порогу нет намеренно: инженер оценивает метрики в MLflow UI перед ручным запуском `deploy_model`.
- **`batch_inference`** — ежедневный прогон новых данных через задеплоенную модель (легкий `api`-образ, без mlflow/optuna).
- **`deploy_model`** — рестарт `api`-контейнера с новыми весами, запускается вручную.

Каждая ML-задача поднимает отдельный контейнер через `docker.sock` (`DockerOperator`) и удаляется после завершения — Airflow-контейнер остается легким и не тянет ML-стек.

---

## Тестирование

`tests/` разделены на:

- **`unit/`** — контракты sklearn-обертки, детекция утечки данных, целостность цикла save/load/predict, тесты фичей и метрик.
- **`integration/`** — полный прогон пайплайна, воспроизводимость, поведение модели, задержка инференса (`test_latency.py`).

```powershell
make test-unit
make test-integration
```

CI настроен через `.github/workflows/ci.yml`.

---

## Дальнейшее развитие

- Тюнинг classification threshold вместо жесткой перебалансировки весов классов (см. вывод из v1.6 и SHAP-анализа).
- Калибровка вероятностей (Platt scaling / isotonic regression) для более честного скоринга.
- Мониторинг data drift между DEV и PROD выборками в проде.
- Расширение SHAP-мониторинга ошибок в постоянный дашборд, а не разовый ноутбук.

---

## Автор

Максим Новиков — [mallienotxc@gmail.com](mailto:mallienotxc@gmail.com)
