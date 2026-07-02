# ============================================================
# КОНФИГУРАЦИЯ ИНТЕРПРЕТАТОРОВ
# ============================================================
PYTHON = python
PYTEST = pytest

.PHONY: help venv install test test-unit test-integration run-train run-tune run-evaluate run-inference clean

# Команда по умолчанию (показывает список доступных команд)
help:
	@echo "Доступные команды для автоматизации проекта:"
	@echo "  make venv              - Создать виртуальное окружение venv"
	@echo "  make install           - Установить зависимости и зарегистрировать пакет"
	@echo "  make test              - Запустить ВСЕ тесты (unit + integration)"
	@echo "  make test-unit         - Запустить только быстрые юнит-тесты"
	@echo "  make test-integration  - Запустить тяжелые интеграционные тесты"
	@echo "  make run-train         - Запустить обучение модели (mode=train)"
	@echo "  make run-tune          - Запустить подбор гиперпараметров (mode=tune)"
	@echo "  make run-evaluate      - Оценка модели на тестовой выборке (mode=evaluate)"
	@echo "  make run-inference     - Прогноз на новых данных (mode=inference)"
	@echo "  make clean             - Полная очистка проекта от кэша, pyc файлов и логов"

# ============================================================
# ОКРУЖЕНИЕ И ЗАВИСИМОСТИ
# ============================================================
venv:
	$(PYTHON) -m venv .venv
	@echo "Окружение создано. Активация:"
	@echo "  Windows : .venv\\Scripts\\activate"
	@echo "  Linux/Mac: source .venv/bin/activate"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

# ============================================================
# СЕРВИСЫ
# ============================================================
mlflow:
	mlflow ui --backend-store-uri sqlite:///logs/mlflow.db --default-artifact-root ./logs/mlruns

# ============================================================
# ТЕСТИРОВАНИЕ (Pytest)
# ============================================================
test:
	$(PYTEST) tests/ -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v

# ============================================================
# ЗАПУСК ПРОЕКТА (Через контекст модуля пакета)
# ============================================================
run-train:
	$(PYTHON) -m main mode=train

run-tune:
	$(PYTHON) -m main mode=tune

run-evaluate:
	$(PYTHON) -m main mode=evaluate

run-inference:
	$(PYTHON) -m main mode=inference

# ============================================================
# ОЧИСТКА ПРОЕКТА (Кроссплатформенная)
# ============================================================
clean:
	@echo "Очистка кэша и временных файлов..."
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.pytest_cache'), pathlib.Path('outputs'), pathlib.Path('multirun'), pathlib.Path('.hydra'), pathlib.Path('catboost_info')] if p.exists()]"
	$(PYTHON) -c "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc') if p.is_file()]"
	@echo "Проект полностью очищен!"