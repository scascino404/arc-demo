"""Environment checks: API conformance, observation sanity, auto-reset,
determinism, checkpoint logic and rendering."""

import os

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from arc import ArcEnv, ArcVecEnv, TaskConfig
from arc.env import HEIGHT_OFFSETS


def test_gymnasium_api():
    env = ArcEnv()
    gym_check_env(env, skip_render_check=True)
    sb3_check_env(env, warn=True)
    env.close()


def test_obs_at_rest_is_sane():
    env = ArcVecEnv(4, TaskConfig(curriculum=False), seed=0)
    obs = env.reset()
    assert obs.shape == (4, env.task.obs_size) and np.all(np.isfinite(obs))
    for _ in range(180):  # the torso bobs on the legs for a few seconds
        obs, *_ = env.step(np.zeros((4, 12), dtype=np.float32))
    # standing still on flat ground: gravity points down in body frame,
    # small velocities, all feet touching, flat heightmap
    assert np.allclose(obs[:, :3], [0, 0, -1], atol=0.05)
    assert np.abs(obs[:, 3:9]).max() < 0.1
    nh = len(HEIGHT_OFFSETS)
    assert np.abs(obs[:, -4 - nh:-4]).max() < 0.1
    assert np.all(obs[:, -4:] > 0.99)
    env.close()


def test_auto_reset_and_terminal_obs():
    cfg = TaskConfig(max_steps=20, curriculum=False)
    env = ArcVecEnv(3, cfg, seed=1)
    env.reset()
    for t in range(20):
        obs, rew, done, infos = env.step(np.zeros((3, 12), dtype=np.float32))
    assert done.all()
    for info in infos:
        assert info["TimeLimit.truncated"]
        assert info["terminal_observation"].shape == obs[0].shape
    # new episodes started
    assert np.all(env.task.steps == 0)
    env.close()


def test_falls_terminate():
    env = ArcVecEnv(8, TaskConfig(curriculum=False, difficulty=1.0), seed=2)
    env.reset()
    rng = np.random.default_rng(0)
    falls = 0
    for _ in range(300):
        _, rew, done, infos = env.step(rng.uniform(-1, 1, (8, 12)).astype(np.float32))
        assert np.all(np.isfinite(rew))
        falls += sum(i.get("fell", False) for i in infos)
    assert falls > 0  # random flailing on hard terrain must tip some robots over
    env.close()


def test_determinism():
    def rollout():
        env = ArcVecEnv(4, TaskConfig(difficulty=0.5, curriculum=False), threads=2)
        env.seed(123)
        obs = env.reset()
        rng = np.random.default_rng(5)
        total = 0.0
        for _ in range(100):
            obs, rew, _, _ = env.step(rng.uniform(-1, 1, (4, 12)).astype(np.float32))
            total += rew.sum()
        env.close()
        return obs, total

    (o1, r1), (o2, r2) = rollout(), rollout()
    assert np.array_equal(o1, o2) and r1 == r2


def test_checkpoint_reached_moves_target():
    env = ArcVecEnv(1, TaskConfig(curriculum=False), seed=3)
    env.reset()
    task = env.task
    # put the checkpoint right under the robot
    task.target[0] = task.sim.root[0, :2]
    task.prev_dist[0] = 0.5
    old = task.target[0].copy()
    _, rew, _, _ = env.step(np.zeros((1, 12), dtype=np.float32))
    assert rew[0] > 0.8 * task.cfg.w_checkpoint
    assert task.checkpoints[0] == 1
    assert np.linalg.norm(task.target[0] - old) >= task.cfg.checkpoint_min_dist - 0.5
    env.close()


def test_curriculum_levels():
    cfg = TaskConfig()
    env = ArcVecEnv(2, cfg, seed=4)
    env.reset()
    task = env.task
    task.checkpoints[:] = [cfg.level_up_checkpoints, 0]
    task._update_levels(np.array([0, 1]), np.array([False, True]))
    assert task.level[0] == pytest.approx(cfg.level_step) and task.level[1] == 0.0
    env.close()


@pytest.mark.skipif(not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
                    reason="needs a display for the OpenGL context")
def test_rgb_render():
    env = ArcEnv(render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img.shape == (360, 640, 3) and img.dtype == np.uint8
    assert img.std() > 10  # not a blank frame
    env.close()
