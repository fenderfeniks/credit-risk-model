from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ============================================================
# БАЗОВЫЕ КОМПОНЕНТЫ (Вложенные схемы)
# ============================================================

@dataclass
class DvcConfig:
    remote: str
    s3_bucket: Optional[str] = None
    endpoint_url: Optional[str] = None


@dataclass
class PathsConfig:
    # Твои старые базовые поля
    data_dir: str
    models_dir: str
    reports_dir: str
    logs_dir: str
    src_dir: str
    sql_dir: str
    
    raw_dir: str
    processed_dir: str
    features_dir: str
    sql_aggregate_dir: str
    
    data_file_name: str
    
    dev_data_file: str
    prod_data_file_globbing: str
    target_file_name: str

    aggregate_sql_request: str
    sql_col_stats_file: str

    

# --- Блоки Данных ---

@dataclass
class TabularDataConfig:
    preprocessing_version: str
    features_version: str
    
    preprocessing_changelog: str
    features_changelog: str

    aggregation_version: str
    aggregation_changelog: str

    target_col: str
    id_col: str
    drop_cols: List[str]

    num_fill_strategy: str
    cat_fill_strategy: str

    outlier_method: str
    outlier_threshold: float
    exact_excludes: List[str]
    prefix_excludes: List[str]
    suffix_excludes: List[str]
    aggregation_stats_cat_cols: List[str]

    max_cardinality_for_categorical: int = 100
    max_missing_pct: float = 0.90
    max_constant_pct: float = 0.99
    max_row_missing_pct: float = 0.50
    top_n_categories: int = 20
    skip_imputation_cols: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    test_size: float
    val_size: float
    sample_pct: float
    # Делаем их опциональными, так как в проекте может быть только текст или только таблицы
    tabular: TabularDataConfig


# --- Блоки Обучения и Моделей ---

@dataclass
class ModelConfig:
    name: str
    model_version: str
    model_changelog: str
    # Гиперпараметры могут быть любыми (CatBoost, нейронка и т.д.),
    # поэтому оставляем гибкий словарь
    params: Dict[str, Any] = field(default_factory=dict)
    # Словарь параметров для поиска (например: {"depth": [4, 10], "lr": [0.01, 0.1]})
    optuna_search_space: Dict[str, Any] = field(default_factory=dict)

# --- Глобальные настройки обучения ---
@dataclass
class MlTrainingConfig:
    cv_folds: int
    early_stopping_rounds: int
    verbose: int


@dataclass
class DlTrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    early_stopping_patience: int
    optimizer: str
    scheduler: str
    momentum: float = 0.9

@dataclass
class OptunaConfig:
    n_trials: int
    direction: str  # "maximize" или "minimize"
    sampler: str    # "tpe" или "random"



@dataclass
class TrainingConfig:
    device: str
    num_workers: int
    pin_memory: bool
    ml: Optional[MlTrainingConfig] = None
    dl: Optional[DlTrainingConfig] = None
    optuna: Optional[OptunaConfig] = None


# --- Блоки Логирования ---

# --- MLflow ---
@dataclass
class MlflowConfig:
    tracking_uri: str
    artifact_uri_rel: str
    experiment_name: str
    experiments: Dict[str, str]


@dataclass
class PlotsConfig:
    fig_size: List[int]
    dpi: int
    font_size: int
    style: str
    grid: bool
    alpha: float
    spines_top: bool
    spines_right: bool


@dataclass
class LoggingConfig:
    level: str
    log_file: str
    mlflow: MlflowConfig
    plots: PlotsConfig

@dataclass
class SecurityConfig:
    allowed_keys: str = ""


# ============================================================
# ГЛАВНЫЙ КЛАСС (Оркестратор схемы)
# ============================================================

@dataclass
class AppConfig:
    """
    Главный класс конфигурации. Точно повторяет структуру config.yaml.
    """
    project_name: str
    run_name: str
    mode: str
    task_type: str
    seed: int
    loss_function: str
    metrics: List[str]
    env: str

    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig

    security: Optional[SecurityConfig] = None
    dvc: Optional[DvcConfig] = None