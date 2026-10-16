"""Environments over `ArcTask`:

- `ArcVecEnv`: a Stable Baselines3 VecEnv with auto-reset; one ctypes call
  steps every world, parallelized with OpenMP in C.
- `ArcEnv`: a single-world Gymnasium env (registered as "Arc-v0").

Both draw world 0 through a `Viewer`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from .native import Renderer
from .task import CAUSES, ArcTask, TaskConfig


def _spaces(task: ArcTask) -> tuple[spaces.Box, spaces.Box]:
    obs = spaces.Box(-np.inf, np.inf, shape=(task.obs_size,), dtype=np.float32)
    act = spaces.Box(-1.0, 1.0, shape=(task.nj,), dtype=np.float32)
    return obs, act


class Viewer:
    """Renderer for one world of a task, created on the first frame: a window
    for render mode "human", offscreen images for "rgb_array"."""

    def __init__(self, render_mode: str | None):
        self.render_mode = render_mode
        self.renderer: Renderer | None = None
        self._view = 0

    def render(self, task: ArcTask, world: int = 0) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.renderer is None:
            human = self.render_mode == "human"
            self.renderer = Renderer(*((1280, 720) if human else (640, 360)), visible=human)
            self.renderer.set_view(self._view)
        active = 0 if task.in_view[world] else -1  # lit while the robot sees it
        img = self.renderer.frame(task.sim, world, task.markers(world), active,
                                  rgb=self.render_mode == "rgb_array")
        if self.render_mode == "human":
            self.renderer.set_title(
                f"arc - checkpoints {task.checkpoints[world]} - difficulty "
                f"{task.difficulty[world]:.1f} - t {task.steps[world] * task.cfg.dt:.1f} s")
        return img

    def set_view(self, preset: int) -> None:
        """Camera preset around the robot's heading: 0 behind, 1 side, 2 front."""
        self._view = preset
        if self.renderer is not None:
            self.renderer.set_view(preset)

    def keys(self) -> list[int]:
        """Keys pressed in the window since the previous call (SDL keycodes)."""
        return self.renderer.keys() if self.renderer is not None else []

    @property
    def closed(self) -> bool:
        return self.renderer is not None and self.renderer.closed

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class ArcVecEnv(VecEnv):
    """Stable Baselines3 vectorized env over one batched native simulation."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, num_envs: int, config: TaskConfig | None = None, threads: int = 1,
                 seed: int | None = None, render_mode: str | None = None):
        self.task = ArcTask(num_envs, config, threads, seed)
        obs_space, act_space = _spaces(self.task)
        self.render_mode = render_mode  # read by VecEnv.__init__
        super().__init__(num_envs, obs_space, act_space)
        self.viewer = Viewer(render_mode)
        self._actions: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        if self._seeds[0] is not None:
            self.task.rng = np.random.default_rng(self._seeds[0])
        self._reset_seeds()
        self.task.reset(np.arange(self.num_envs))
        self._reset_options()
        return self.task.observe()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        reward, terminated, truncated, stats = self.task.step(self._actions)
        obs = self.task.observe()
        done = terminated | truncated
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        ended = np.flatnonzero(done)
        for i in ended:
            infos[i] = {
                "terminal_observation": obs[i].copy(),
                "TimeLimit.truncated": bool(truncated[i]),
                "checkpoints": int(stats["checkpoints"][i]),
                "difficulty": float(stats["difficulty"][i]),
                "fell": bool(stats["fell"][i]),
                "cause": CAUSES[stats["cause"][i]],
            }
        if len(ended):
            self.task.reset(ended)
            obs[ended] = self.task.observe()[ended]
        return obs, reward, done, infos

    def render(self, mode: str | None = None) -> np.ndarray | None:
        return self.viewer.render(self.task)

    def get_images(self):
        return [self.viewer.render(self.task)]

    def close(self) -> None:
        self.viewer.close()
        self.task.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self, method_name)(*method_args, **method_kwargs)
                for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))


class ArcEnv(gym.Env):
    """Single-world Gymnasium env, for checks, debugging and rendering."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: TaskConfig | None = None, render_mode: str | None = None):
        self.task = ArcTask(1, config)
        self.observation_space, self.action_space = _spaces(self.task)
        self.render_mode = render_mode
        self.viewer = Viewer(render_mode)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.task.rng = np.random.default_rng(seed)
        self.task.reset(np.array([0]))
        return self.task.observe()[0], {}

    def step(self, action):
        reward, terminated, truncated, stats = self.task.step(np.asarray(action)[None])
        info = {k: v[0].item() for k, v in stats.items()}
        info["cause"] = CAUSES[info["cause"]]
        return self.task.observe()[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), info

    def render(self):
        return self.viewer.render(self.task)

    def close(self):
        self.viewer.close()
        self.task.close()
