"""Checkpoint-reaching task for the quadruped, vectorized over N worlds.

`ArcTask` holds the task logic (observations, rewards, terminations,
checkpoints, terrain curriculum) as numpy operations over all worlds of one
native `Sim`. Two thin front ends expose it:

- `ArcVecEnv`: a Stable Baselines3 VecEnv with auto-reset; one ctypes call
  steps every world, parallelized with OpenMP in C.
- `ArcEnv`: a single-world Gymnasium env (registered as "Arc-v0").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from .native import Renderer, Sim


@dataclass
class TaskConfig:
    # simulation
    dt: float = 1.0 / 30.0
    substeps: int = 20
    grid: int = 64
    cell: float = 0.4
    # episode
    max_steps: int = 1000
    action_scale: float = 0.6         # rad per unit action, around the default stance
    checkpoint_radius: float = 0.5       # torso center within this distance (drawn ring)
    checkpoint_min_dist: float = 3.0
    checkpoint_max_dist: float = 6.0
    border: float = 1.5               # keep checkpoints this far from the boundary walls
    # Rewards, per control step. The progress term is capped so that speed is
    # not rewarded beyond max_speed (an uncapped one breeds frantic bounding
    # gaits), and the time cost makes hovering or orbiting near a checkpoint
    # a losing strategy. Its discounted sum, w_time / (1 - gamma) = 5, stays
    # below w_fall, so ending the episode on purpose never pays.
    w_progress: float = 10.0          # per meter towards the checkpoint
    max_speed: float = 1.2            # m/s; faster approach earns nothing more
    # The eye looks along the body's +x axis. Progress towards a checkpoint
    # only pays while the checkpoint is in view, fully when it is dead ahead
    # and fading out at the edge of the field of view; turning the eye
    # towards the checkpoint pays w_turn per radian; and a checkpoint only
    # counts when reached while in view. So: look first, then walk eye-first.
    fov: float = 45.0                 # half angle of the field of view, degrees
    w_turn: float = 1.0               # per radian of gaze error reduction
    w_checkpoint: float = 10.0
    w_time: float = 0.05
    w_fall: float = 10.0
    w_energy: float = 0.002           # per joule of positive + negative joint work
    w_action_rate: float = 0.005      # per squared action change
    w_body_contact: float = 0.02      # knees / shins touching
    w_torso_contact: float = 0.5      # torso touching (scraping over a step)
    w_body_rate: float = 0.01         # per (rad/s)^2 of torso roll / pitch rate
    w_bounce: float = 0.05            # per (m/s)^2 of torso vertical velocity
    # terminations: flipping over, or the torso resting on the ground too long
    flip_up_z: float = 0.3            # torso up axis z below this (tilt > 72 degrees)
    beached_steps: int = 30           # consecutive steps of torso contact
    # terrain curriculum: per-world difficulty level in [0, max_difficulty]
    curriculum: bool = True
    difficulty: float = 0.0           # fixed difficulty when curriculum is off
    max_difficulty: float = 1.0
    level_step: float = 0.1
    level_up_checkpoints: int = 3     # promote after reaching this many in an episode
    level_down_steps: int = 300       # demote after falling this early without any checkpoint


# termination causes reported in infos["cause"]
CAUSES = ("", "flip", "beached", "unstable")

# heightmap sample grid around the torso, (forward, left) in meters; it covers
# the feet (about 0.9 m from the torso center) plus some look-ahead
_F = np.arange(-1.0, 1.41, 0.2)
_L = np.arange(-1.0, 1.01, 0.2)
HEIGHT_OFFSETS = np.array([(f, l) for f in _F for l in _L])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz quaternions to (N, 3, 3) rotation matrices (body to world)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def gaze_error(rot: np.ndarray, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Unsigned angle (rad) between the eye's gaze (body +x, projected on the
    ground) and the direction from the body to the target."""
    fwd = rot[:, :2, 0]
    d = target - pos[:, :2]
    cross = fwd[:, 0] * d[:, 1] - fwd[:, 1] * d[:, 0]
    return np.abs(np.arctan2(cross, np.sum(fwd * d, axis=1)))


class ArcTask:
    def __init__(self, num_worlds: int, config: TaskConfig | None = None, threads: int = 1,
                 seed: int | None = None):
        self.cfg = cfg = config or TaskConfig()
        self.n = num_worlds
        self.sim = Sim(num_worlds, cfg.dt, cfg.substeps, cfg.grid, cfg.cell, threads)
        self.nj = self.sim.num_joints
        self.rng = np.random.default_rng(seed)

        n, nj = num_worlds, self.nj
        self.obs_size = 3 + 3 + 3 + nj + nj + nj + 3 + len(HEIGHT_OFFSETS) + self.sim.num_legs
        self.target = np.zeros((n, 2))
        self.target_z = np.zeros(n)
        self.prev_dist = np.zeros(n)
        self.prev_gaze = np.zeros(n)          # angle between gaze and checkpoint
        self.in_view = np.zeros(n, dtype=bool)
        self.prev_action = np.zeros((n, nj))
        self.torso_steps = np.zeros(n, dtype=np.int64)
        self.steps = np.zeros(n, dtype=np.int64)
        self.checkpoints = np.zeros(n, dtype=np.int64)
        self.ep_return = np.zeros(n)
        self.level = np.full(n, 0.0 if cfg.curriculum else cfg.difficulty)
        self.difficulty = self.level.copy()
        self._heights = np.empty((n, len(HEIGHT_OFFSETS)))

    # -- episode management -------------------------------------------------

    def reset(self, ids: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids):
            return
        seeds = self.rng.integers(0, 2**32, size=len(ids), dtype=np.uint32)
        self.difficulty[ids] = self.level[ids]
        self.sim.reset(ids, seeds, self.difficulty[ids])
        self.prev_action[ids] = 0.0
        self.torso_steps[ids] = 0
        self.steps[ids] = 0
        self.checkpoints[ids] = 0
        self.ep_return[ids] = 0.0
        self.sim.observe()
        self._new_targets(ids)

    def _new_targets(self, ids: np.ndarray) -> None:
        """Places the next checkpoint 3-6 m from the robot, inside the terrain."""
        cfg = self.cfg
        pos = self.sim.root[ids, :2]
        limit = self.sim.half_extent - cfg.border
        target = np.zeros((len(ids), 2))
        pending = np.ones(len(ids), dtype=bool)
        for _ in range(20):
            k = np.flatnonzero(pending)
            if not len(k):
                break
            ang = self.rng.uniform(-np.pi, np.pi, len(k))
            dist = self.rng.uniform(cfg.checkpoint_min_dist, cfg.checkpoint_max_dist, len(k))
            cand = pos[k] + dist[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
            ok = np.all(np.abs(cand) < limit, axis=1)
            target[k[ok]] = cand[ok]
            pending[k[ok]] = False
        # fallback: towards the center
        k = np.flatnonzero(pending)
        target[k] = 0.5 * pos[k]
        self.target[ids] = target
        self.target_z[ids] = self.sim.terrain_at(ids, target)
        self.prev_dist[ids] = np.linalg.norm(target - pos, axis=1)
        self.prev_gaze[ids] = gaze_error(self.sim.rot[ids], pos, target)
        self.in_view[ids] = self.prev_gaze[ids] < np.radians(self.cfg.fov)

    # -- stepping -----------------------------------------------------------

    def step(self, actions: np.ndarray):
        """Advances all worlds. Returns rewards, terminated, truncated, and a dict
        of per-world episode statistics arrays (valid where an episode ended)."""
        cfg, sim = self.cfg, self.sim
        a = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        sim.step(sim.default_angles + cfg.action_scale * a)
        root, joints, contacts, flags = sim.observe()
        nj, nl = self.nj, sim.num_legs
        self.steps += 1

        pos = root[:, :3]
        rot = sim.rot
        to_target = self.target - pos[:, :2]
        dist = np.hypot(to_target[:, 0], to_target[:, 1])
        gaze = gaze_error(rot, pos, self.target)
        cos_fov = np.cos(np.radians(cfg.fov))
        view = np.clip((np.cos(gaze) - cos_fov) / (1.0 - cos_fov), 0.0, 1.0)
        self.in_view = gaze < np.radians(cfg.fov)
        progress = np.minimum(self.prev_dist - dist, cfg.max_speed * cfg.dt)
        # approaching pays only as much as the checkpoint is in view; moving
        # away always costs, so no back-and-forth cycle can earn reward
        progress = np.where(progress > 0, progress * view, progress)
        turn = self.prev_gaze - gaze
        reached = (dist < cfg.checkpoint_radius) & self.in_view

        body_rate = np.matmul(root[:, None, 10:13], rot)[:, 0, :2]  # roll, pitch
        torso = contacts[:, nl] > 0
        self.torso_steps = np.where(torso, self.torso_steps + 1, 0)
        power = np.abs(joints[:, nj:2 * nj] * joints[:, 2 * nj:]).sum(1)
        rate = np.square(a - self.prev_action).sum(1)
        reward = (cfg.w_progress * progress
                  + cfg.w_turn * turn
                  + cfg.w_checkpoint * reached
                  - cfg.w_time
                  - cfg.w_energy * power * cfg.dt
                  - cfg.w_action_rate * rate
                  - cfg.w_body_contact * (contacts[:, nl + 1] > 0)
                  - cfg.w_torso_contact * torso
                  - cfg.w_body_rate * np.square(body_rate).sum(1)
                  - cfg.w_bounce * np.square(root[:, 9]))

        # the boundary walls keep robots in; escaping anyway means a solver failure
        out = np.any(np.abs(pos[:, :2]) > sim.half_extent + 0.5, axis=1)
        bad = (flags != 0) | out | ~np.all(np.isfinite(root), axis=1)
        flip = rot[:, 2, 2] < cfg.flip_up_z
        beached = self.torso_steps > cfg.beached_steps
        fell = flip | beached | bad
        cause = np.where(bad, 3, np.where(flip, 1, np.where(beached, 2, 0)))
        reward = np.where(fell, reward - cfg.w_fall, reward)
        reward = np.where(bad, -cfg.w_fall, reward)

        self.prev_dist = dist
        self.prev_gaze = gaze
        self.prev_action = a
        self.checkpoints += reached
        self.ep_return += reward
        hit = np.flatnonzero(reached & ~fell)
        if len(hit):
            self._new_targets(hit)

        terminated = fell
        truncated = (self.steps >= cfg.max_steps) & ~terminated
        stats = {"checkpoints": self.checkpoints.copy(), "difficulty": self.difficulty.copy(),
                 "fell": fell.copy(), "cause": cause, "steps": self.steps.copy()}
        done = np.flatnonzero(terminated | truncated)
        if len(done) and cfg.curriculum:
            self._update_levels(done, fell)
        return reward.astype(np.float32), terminated, truncated, stats

    def _update_levels(self, done: np.ndarray, fell: np.ndarray) -> None:
        cfg = self.cfg
        up = self.checkpoints[done] >= cfg.level_up_checkpoints
        down = fell[done] & (self.checkpoints[done] == 0) & (self.steps[done] < cfg.level_down_steps)
        lvl = self.level[done] + cfg.level_step * (up.astype(float) - down.astype(float))
        self.level[done] = np.clip(np.round(lvl, 6), 0.0, cfg.max_difficulty)

    # -- observations -------------------------------------------------------

    def observe(self) -> np.ndarray:
        sim, nj, nl = self.sim, self.nj, self.sim.num_legs
        root, joints, contacts, rot = sim.root, sim.joints, sim.contacts, sim.rot
        gravity = -rot[:, 2, :]                       # world -z in body frame
        lin = np.matmul(root[:, None, 7:10], rot)[:, 0]  # rot^T v
        ang = np.matmul(root[:, None, 10:13], rot)[:, 0]

        # checkpoint in the heading frame
        heading = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
        d = self.target - root[:, :2]
        c, s = np.cos(heading), np.sin(heading)
        fwd, left = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        dist = np.maximum(np.hypot(fwd, left), 1e-6)
        target = np.stack([fwd / dist, left / dist, np.minimum(dist, 6.0) / 6.0], 1)

        heights = sim.heights(HEIGHT_OFFSETS, out=self._heights)
        rel_heights = np.clip(root[:, 2:3] - sim.stand_height - heights, -1.0, 1.0)

        obs = np.concatenate([
            gravity, lin, 0.25 * ang,
            joints[:, :nj] - sim.default_angles,
            0.1 * joints[:, nj:2 * nj],
            self.prev_action,
            target,
            rel_heights,
            contacts[:, :nl],
        ], axis=1, dtype=np.float32)
        obs[~np.isfinite(obs)] = 0.0
        return obs

    def markers(self, i: int) -> np.ndarray:
        """Checkpoints to draw for world i, rows of (x, y, z, radius)."""
        return np.array([[self.target[i, 0], self.target[i, 1], self.target_z[i],
                          self.cfg.checkpoint_radius]])

    def close(self) -> None:
        self.sim.close()


def _spaces(task: ArcTask) -> tuple[spaces.Box, spaces.Box]:
    obs = spaces.Box(-np.inf, np.inf, shape=(task.obs_size,), dtype=np.float32)
    act = spaces.Box(-1.0, 1.0, shape=(task.nj,), dtype=np.float32)
    return obs, act


class _View:
    """Lazily created renderer shared by both front ends."""

    def __init__(self, render_mode: str | None, width: int = 1280, height: int = 720):
        self.render_mode = render_mode
        self.size = (width, height) if render_mode == "human" else (640, 360)
        self.renderer: Renderer | None = None

    def render(self, task: ArcTask, world: int = 0) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.renderer is None:
            self.renderer = Renderer(*self.size, visible=self.render_mode == "human")
        active = 0 if task.in_view[world] else -1  # lit while the robot sees it
        img = self.renderer.frame(task.sim, world, task.markers(world), active,
                                  rgb=self.render_mode == "rgb_array")
        if self.render_mode == "human":
            self.renderer.set_title(
                f"arc - checkpoints {task.checkpoints[world]} - difficulty "
                f"{task.difficulty[world]:.1f} - t {task.steps[world] * task.cfg.dt:.1f} s")
        return img

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
        self._view = _View(render_mode)
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
        return self._view.render(self.task, 0)

    def get_images(self):
        return [self._view.render(self.task, 0)]

    @property
    def window_closed(self) -> bool:
        return self._view.closed

    def close(self) -> None:
        self._view.close()
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
        self._view = _View(render_mode)

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
        return self._view.render(self.task, 0)

    def close(self):
        self._view.close()
        self.task.close()
