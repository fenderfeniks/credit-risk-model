from __future__ import annotations

import logging
import optuna
from omegaconf import DictConfig, OmegaConf
from optuna.samplers import TPESampler, RandomSampler

from src.core.pipeline import MLPipeline

logger = logging.getLogger(__name__)


class OptunaTuner:
    def __init__(self, cfg, project_root, tracker):
        self.project_root = project_root
        self.cfg = cfg
        self.optuna_cfg = cfg.training.optuna
        self.tracker = tracker

    def _get_sampler(self):
        name = self.optuna_cfg.sampler.lower()
        if name == "tpe":
            return TPESampler()
        elif name == "random":
            return RandomSampler()
        else:
            logger.warning(f"Неизвестный sampler '{name}', используется TPE.")
            return TPESampler()

    def tune(self, X_train, y_train, X_val, y_val):
        logger.info(f"Старт Optuna (Trials: {self.optuna_cfg.n_trials})")

        def objective(trial):
            trial_cfg = OmegaConf.create(
                OmegaConf.to_container(self.cfg, resolve=True)
            )
            search_space = trial_cfg.model.get("optuna_search_space", {})
            current_trial_params = {}

            # ФИКС: цикл теперь ТОЛЬКО собирает параметры для ВСЕХ ключей search_space.
            # Раньше "with tracker.start_run(...)" и "return score" были внутри
            # этого цикла (на одном уровне отступа с if isinstance(...)), поэтому
            # функция обучала модель и завершалась уже на первой итерации —
            # остальные гиперпараметры из search_space никогда не применялись,
            # и все trial'ы фактически получали одинаковый (неполный) набор params.
            for param_name, space_info in search_space.items():
                if isinstance(space_info, (dict, DictConfig)):
                    p_type = space_info.get("type", "float")
                    bounds = space_info.get("bounds", [])
                    is_log = space_info.get("log", False)
                    if len(bounds) == 2:
                        if p_type == "int":
                            value = trial.suggest_int(param_name, bounds[0], bounds[1])
                        else:
                            value = trial.suggest_float(param_name, bounds[0], bounds[1], log=is_log)
                        OmegaConf.update(trial_cfg, f"model.params.{param_name}", value)
                        current_trial_params[param_name] = value

            # === СТАРТ ВЛОЖЕННОГО РАНА ЧЕРЕЗ ЕДИНЫЙ ТРЕКЕР ===
            # Теперь выполняется РОВНО ОДИН РАЗ за trial, после того как ВСЕ
            # параметры из search_space собраны и применены в trial_cfg.
            with self.tracker.start_run(run_name=f"trial_{trial.number}", nested=True):
                self.tracker.log_params(current_trial_params)
                pipeline = MLPipeline(cfg=trial_cfg, project_root=self.project_root, tracker=self.tracker)
                pipeline.train(X_train, y_train, X_val, y_val, save_artifacts=False, use_tracker=False)
                target_metric = trial_cfg.metrics[0]
                score = pipeline.model.get_best_val_score(target_metric)
                self.tracker.log_metrics({f"val_{target_metric}": score})

            return score

        study = optuna.create_study(
            direction=self.optuna_cfg.direction,
            sampler=self._get_sampler()
        )
        # catch=(Exception,) — один неудачный trial (например, NaN-loss на кривой комбинации
        # гиперпараметров) не должен убивать всё исследование целиком
        study.optimize(objective, n_trials=self.optuna_cfg.n_trials, catch=(Exception,))

        logger.info(f"Лучшие параметры: {study.best_params}")
        return study.best_params