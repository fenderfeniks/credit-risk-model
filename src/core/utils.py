from __future__ import annotations

from pathlib import Path
from omegaconf import DictConfig
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.config_store import ConfigStore
import functools

from src.core.config_schema import AppConfig


def get_project_root() -> Path:
    """
    Динамически находит корень проекта, поднимаясь по директориям вверх,
    пока не найдет папку 'configs' или 'pyproject.toml'.
    """
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "configs").exists() or (parent / "pyproject.toml").exists():
            return parent

    # Fallback на случай нестандартной структуры
    return Path(__file__).resolve().parent.parent.parent


# ГЛОБАЛЬНАЯ КОНСТАНТА КОРНЯ ПРОЕКТА
PROJECT_ROOT = get_project_root()

def register_config_schema():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=AppConfig)


@functools.lru_cache(maxsize=1)
def load_hydra_config(config_name: str = "config") -> DictConfig:
    """Загружает конфиг один раз, валидирует по схеме AppConfig и кеширует его."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    register_config_schema()

    if not GlobalHydra.instance().is_initialized():
        abs_config_path = str(PROJECT_ROOT / "configs")  # <-- было через os.getcwd()
        initialize_config_dir(version_base=None, config_dir=abs_config_path)

    return compose(config_name=config_name)


def clear_config_cache():
    """Сбрасывает кэш конфига (использовать только в Unit-тестах)."""
    load_hydra_config.cache_clear()
    GlobalHydra.instance().clear()
