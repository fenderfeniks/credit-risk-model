"""
test_main.py — интеграционные smoke-тесты оркестратора (main.py).

Покрытие:
- Режим TRAIN: загрузка, сплит, обучение, сохранение моделей.
- Режим INFERENCE: чтение сохраненных моделей, предикт, выгрузка CSV.
- Режим EVALUATE: оценка отложенной выборки.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

# Импортируем функцию main из корня проекта
from main import main

@pytest.fixture
def setup_main_mocks(sample_data):
    """
    Мокает тяжелые I/O компоненты, чтобы main() работал в песочнице, 
    не требуя реальных SQL-файлов и базы данных.
    """
    # 1. Мокаем загрузчик данных, чтобы он сразу отдавал sample_data
    mock_loader = MagicMock()
    mock_loader.load.return_value = sample_data

    # 2. Мокаем компилятор статистик (чтобы не дергал SQL на диске)
    mock_stats = MagicMock()
    mock_stats.get_sql_format_kwargs.return_value = {}

    with patch('main.get_data_source', return_value=mock_loader), \
         patch('main.GlobalStatCompiler', return_value=mock_stats), \
         patch('main.get_train_ids_fast', return_value=set(sample_data['session_id'])), \
         patch('main.resolve_features_source', return_value="mock_source"):
        yield

# ---------------------------------------------------------------------------
# Тест 1: Режим TRAIN
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_main_train_mode(mock_config, setup_main_mocks, tmp_path):
    """Smoke-тест режима обучения."""
    OmegaConf.update(mock_config, "mode", "train")
    OmegaConf.update(mock_config, "data.tabular.id_col", "session_id")
    
    # Запускаем main, он не должен упасть
    main(mock_config)
    
    # Проверяем, что пайплайн отработал и сохранил артефакты
    models_dir = tmp_path / mock_config.paths.models_dir
    prep_ver = mock_config.data.tabular.preprocessing_version
    
    assert (models_dir / f"preprocessing_v{prep_ver}.pkl").exists(), "Препроцессор не сохранен!"

# ---------------------------------------------------------------------------
# Тест 2: Режим EVALUATE
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_main_evaluate_mode(mock_config, setup_main_mocks, tmp_path):
    """Smoke-тест режима оценки (требует обученной модели на диске)."""
    OmegaConf.update(mock_config, "mode", "evaluate")
    OmegaConf.update(mock_config, "data.tabular.id_col", "session_id")
    
    # Сначала "обучаем", чтобы создать веса на диске для загрузки
    OmegaConf.update(mock_config, "mode", "train")
    main(mock_config)
    
    # Теперь оцениваем
    OmegaConf.update(mock_config, "mode", "evaluate")
    main(mock_config)
    # Если функция дошла до конца без ошибок — тест пройден.

# ---------------------------------------------------------------------------
# Тест 3: Режим INFERENCE
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_main_inference_mode(mock_config, setup_main_mocks, tmp_path):
    """Smoke-тест режима инференса. Проверяем генерацию CSV."""
    OmegaConf.update(mock_config, "mode", "inference")
    OmegaConf.update(mock_config, "data.tabular.id_col", "session_id")
    
    # 1. Сначала "обучаем", чтобы создать веса
    OmegaConf.update(mock_config, "mode", "train")
    main(mock_config)
    
    # 2. Переключаем в инференс
    OmegaConf.update(mock_config, "mode", "inference")
    main(mock_config)
    
    # 3. Проверяем, что результат сохранился куда надо
    data_dir = tmp_path / mock_config.paths.data_dir
    model_name = mock_config.model.name
    model_ver = mock_config.model.model_version
    
    output_csv = data_dir / f"predictions_{model_name}_v{model_ver}.csv"
    assert output_csv.exists(), f"Файл предсказаний не создан: {output_csv}"
    
    # Проверяем структуру CSV
    df_res = pd.read_csv(output_csv)
    assert "prediction" in df_res.columns
    assert "probability" in df_res.columns # Т.к. task_type='binary'