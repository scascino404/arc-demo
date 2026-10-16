"""Training runs on disk: a directory with the PPO model (model.zip) and the
observation normalization statistics (vecnormalize.pkl)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from .env import ArcVecEnv
from .task import TaskConfig

MODEL = "model.zip"
NORMALIZE = "vecnormalize.pkl"


def default_threads() -> int:
    """Half the cores: SMT threads add nothing to the solver or to torch."""
    return max(1, (os.cpu_count() or 2) // 2)


def save_run(path: Path, model: PPO, env: VecNormalize) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save(path / MODEL)
    env.save(str(path / NORMALIZE))


def load_normalize(run: Path, env: VecEnv, training: bool = False) -> VecNormalize:
    """Wraps env in the run's normalization; frozen unless training."""
    env = VecNormalize.load(str(run / NORMALIZE), env)
    env.training = training
    if not training:
        env.norm_reward = False
    return env


def load_model(run: Path, env: VecEnv | None = None, **kwargs) -> PPO:
    return PPO.load(run / MODEL, env=env, device="cpu", **kwargs)


def load_policy(run: Path, num_envs: int, cfg: TaskConfig, threads: int = 1,
                seed: int | None = None, render_mode: str | None = None
                ) -> tuple[VecNormalize, Callable[..., np.ndarray]]:
    """A normalized ArcVecEnv and the run's policy as act(obs, deterministic=True)."""
    env = load_normalize(run, ArcVecEnv(num_envs, cfg, threads, seed, render_mode))
    policy = load_model(run).policy
    # what model.predict computes, without switching every module to eval
    # mode again on each call
    policy.set_training_mode(False)

    def act(obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            dist = policy.get_distribution(torch.as_tensor(obs))
            return (dist.mode() if deterministic else dist.sample()).numpy()

    return env, act
