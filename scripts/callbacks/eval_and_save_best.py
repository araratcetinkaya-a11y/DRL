# scripts/callbacks/eval_and_save_best.py

from __future__ import annotations
import os
import csv
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm


def _now_stamp() -> str:
    # 2025_09_10_04_39 formatı (YYYY_MM_DD_HH_MM)
    return time.strftime("%Y_%m_%d_%H_%M", time.localtime())


def _seed_env_legacy(env: Union[gym.Env, VecEnv], seed: Optional[int]) -> None:
    """
    Gym 0.19 ve SB3 vektör/tekil env uyumlu seed ayarı.
    - Tek env: env.seed(seed)
    - VecEnv: env.env_method('seed', seed)
    """
    try:
        if hasattr(env, "env_method"):  # VecEnv
            env.env_method("seed", seed)
        else:
            env.seed(seed)
    except Exception:
        # Seed atılamasa bile değerlendirmeyi sürdür
        pass


def _reset_legacy(env: Union[gym.Env, VecEnv]) -> Any:
    """
    Gym 0.19 reset -> sadece obs döner.
    VecEnv ise (obs, infos) gelebilir; SB3 VecEnv reset() genelde sadece obs döndürür.
    """
    obs = env.reset()
    return obs


def _step_once_legacy(env: Union[gym.Env, VecEnv], action: Any) -> Tuple[Any, float, bool, Dict]:
    """
    Gym 0.19 step imzası: obs, reward, done, info
    VecEnv ise tekli DummyVecEnv olduğunda obs/reward/done/info vektör döner; bunu açarız.
    """
    obs, reward, done, info = env.step(action)
    # VecEnv için tekli durum:
    if isinstance(reward, (np.ndarray, list)):
        # Tekli env varsayımı: len == 1
        reward = float(reward[0])
        done = bool(done[0])
        info = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else info
        obs = obs[0] if isinstance(obs, (np.ndarray, list)) and len(obs) > 0 else obs
    return obs, float(reward), bool(done), info


class EvalAndSaveBestCallback(BaseCallback):
    """
    Her eval_freq adımda:
      - eval_seeds listesini döngüsel kullanarak n_eval_episodes bölüm değerlendirir,
      - success_rate'i hesaplar (info['is_success'] True veya info['reason']=="reach"),
      - success_rate iyileşirse modeli şu hiyerarşiye kaydeder:
          best_evals/<env_name>/<YYYY_MM_DD_HH_MM>_<dynamic_name>_<policy_name>/
    Ayrıca eval sonuçlarını eval_log_dir altına CSV olarak ekler.
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        eval_freq: int,
        n_eval_episodes: int,
        eval_seeds: List[Optional[int]],
        deterministic_eval: bool,
        env_name: str,
        dynamic_name: str,
        policy_name: str,
        eval_log_dir: Union[str, os.PathLike] = "./logs_eval",
        best_root_dir: Union[str, os.PathLike] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_seeds = list(eval_seeds) if len(eval_seeds) > 0 else [None]
        self.deterministic_eval = bool(deterministic_eval)
        self.env_name = env_name
        self.dynamic_name = dynamic_name
        self.policy_name = policy_name
        self.eval_log_dir = Path(eval_log_dir)
        self.eval_log_dir.mkdir(parents=True, exist_ok=True)
        # best root default: <repo_cwd>/best_evals
        self.best_root_dir = Path(best_root_dir) if best_root_dir is not None else (Path.cwd() / "best_evals")
        self.best_root_dir.mkdir(parents=True, exist_ok=True)
        self.best_success_rate: float = -1.0
        self._eval_step_last: int = 0  # son eval yapılan global_step

        # CSV başlığı
        self._csv_path = self.eval_log_dir / f"eval_{self.env_name}_{self.policy_name}.csv"
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["global_step", "mean_reward", "std_reward", "success_rate", "improved", "best_success_rate", "timestamp"])

    def _on_step(self) -> bool:
        # Her adımda çağrılır; eval zamanı mı?
        step = int(self.num_timesteps)
        if self._should_trigger_eval(step):
            self._do_eval_and_maybe_save(step)
            self._eval_step_last = step
        return True

    def _should_trigger_eval(self, step: int) -> bool:
        if self.eval_freq <= 0:
            return False
        if step < self.eval_freq:
            return False
        # step farkı ile tetikle (örn. 20000'den 30000'e zıplanınca 30000'de yine tetikler)
        return (step - self._eval_step_last) >= self.eval_freq

    def _do_eval_and_maybe_save(self, step: int) -> None:
        model: BaseAlgorithm = self.model
        seeds = self.eval_seeds
        n_eps = self.n_eval_episodes

        rewards: List[float] = []
        successes = 0

        # n_eval_episodes kadar bölüm çalıştır
        for ep in range(n_eps):
            seed = seeds[ep % len(seeds)]
            _seed_env_legacy(self.eval_env, seed)
            obs = _reset_legacy(self.eval_env)

            done = False
            ep_rew = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=self.deterministic_eval)
                obs, r, done, info = _step_once_legacy(self.eval_env, action)
                ep_rew += r

            rewards.append(ep_rew)

            is_succ = False
            if isinstance(info, dict):
                if info.get("is_success", False):
                    is_succ = True
                elif str(info.get("reason", "")).lower() == "reach":
                    is_succ = True
            successes += int(is_succ)

        mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
        std_reward = float(np.std(rewards)) if len(rewards) > 0 else 0.0
        success_rate = float(successes) / float(n_eps) if n_eps > 0 else 0.0

        improved = success_rate > self.best_success_rate
        if improved:
            self.best_success_rate = success_rate
            self._save_best(step)

        # CSV log
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, f"{mean_reward:.6f}", f"{std_reward:.6f}", f"{success_rate:.6f}", int(improved), f"{self.best_success_rate:.6f}", _now_stamp()])

        # Terminal çıktısı
        if self.verbose > 0:
            print(f"[EVAL] step={step}  meanR={mean_reward:.2f}±{std_reward:.2f}  succ={success_rate*100:.1f}%  "
                  f"{'**BEST**' if improved else ''}")

    def _save_best(self, step: int) -> None:
        # best_evals/<env_name>/<YYYY_MM_DD_HH_MM>_<dynamic>_<policy>/
        stamp = _now_stamp()
        dst_dir = self.best_root_dir / self.env_name / f"{stamp}_{self.dynamic_name}_{self.policy_name}"
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Model
        model_path = dst_dir / "model.zip"
        self.model.save(str(model_path))

        # VecNormalize istatistikleri varsa kaydet (opsiyonel)
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            vecenv = self.model.get_env()
            if isinstance(vecenv, VecEnv):
                # VecNormalize wrapper arayalım
                maybe_vn = None
                if hasattr(vecenv, "venv") and isinstance(vecenv.venv, VecNormalize):
                    maybe_vn = vecenv.venv
                elif hasattr(vecenv, "envs") and len(vecenv.envs) > 0:
                    # Tekli env içinde aranabilir
                    for e in vecenv.envs:
                        if isinstance(e, VecNormalize):
                            maybe_vn = e
                            break
                if maybe_vn is not None:
                    vn_path = dst_dir / "vecnormalize.pkl"
                    maybe_vn.save(str(vn_path))
        except Exception:
            pass

        # Küçük bir metadata dosyası
        meta_path = dst_dir / "eval_meta.txt"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"global_step={self.num_timesteps}\n")
            f.write(f"best_success_rate={self.best_success_rate:.6f}\n")
            f.write(f"timestamp={_now_stamp()}\n")
