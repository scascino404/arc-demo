"""arc-demo: quadruped locomotion over block terrain, on a small C89 physics engine."""

import gymnasium as gym

from .env import ArcEnv, ArcTask, ArcVecEnv, TaskConfig
from .native import Renderer, Sim

gym.register(id="Arc-v0", entry_point="arc.env:ArcEnv", max_episode_steps=None)

__all__ = ["ArcEnv", "ArcTask", "ArcVecEnv", "Renderer", "Sim", "TaskConfig"]
