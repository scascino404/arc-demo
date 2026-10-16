"""Environment checks: API conformance, observation sanity, auto-reset,
determinism, checkpoint logic and rendering."""

import os

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from arc import ArcEnv, ArcVecEnv, TaskConfig
from arc.task import HEIGHT_OFFSETS


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


def test_void_terminates():
    # the torso past the edge of the terrain ends the episode, as a fall
    env = ArcVecEnv(2, TaskConfig(curriculum=False), seed=0)
    env.reset()
    sim = env.task.sim
    observe = sim.observe

    def observe_off_edge():
        out = observe()
        sim.root[1, 0] = sim.half_extent + 0.1
        return out

    sim.observe = observe_off_edge
    _, rew, done, infos = env.step(np.zeros((2, 12), dtype=np.float32))
    assert not done[0]
    assert done[1] and infos[1]["fell"] and infos[1]["cause"] == "void"
    assert rew[1] < -env.task.cfg.w_fall + 1.0
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


# -- reward robustness ---------------------------------------------------------

def test_no_reward_for_ending_episodes_on_purpose():
    cfg = TaskConfig()
    gamma = 0.99  # PPO discount used by scripts/train.py
    # standing still forever costs less than falling once
    assert cfg.w_time / (1 - gamma) < cfg.w_fall


def test_hovering_near_a_checkpoint_costs():
    env = ArcVecEnv(4, TaskConfig(curriculum=False), seed=5)
    env.reset()
    task = env.task
    for _ in range(150):
        env.step(np.zeros((4, 12), dtype=np.float32))
    # checkpoint just outside the radius, robot standing still
    task.target[:] = task.sim.root[:, :2] + [task.cfg.checkpoint_radius + 0.3, 0.0]
    task.prev_dist[:] = task.cfg.checkpoint_radius + 0.3
    rewards = [env.step(np.zeros((4, 12), dtype=np.float32))[1] for _ in range(30)]
    assert np.all(np.sum(rewards, axis=0) < 0)
    env.close()


def test_progress_reward_is_capped():
    cfg = TaskConfig(curriculum=False)
    env = ArcVecEnv(1, cfg, seed=6)
    env.reset()
    task = env.task
    # pretend the robot covered 1 m in one step: worth only max_speed * dt
    task.prev_dist[0] += 1.0
    _, rew, _, _ = env.step(np.zeros((1, 12), dtype=np.float32))
    assert rew[0] <= cfg.w_progress * cfg.max_speed * cfg.dt + 1e-6
    env.close()


def test_torso_contact_is_penalized_then_terminal():
    cfg = TaskConfig(curriculum=False)
    env = ArcVecEnv(1, cfg, seed=7)
    env.reset()
    task = env.task
    # femurs up, shins pointing outwards: the torso drops onto the ground
    fold = np.tile([0.0, 1.0, 1.0], 4)[None].astype(np.float32)
    first_contact, end, cause = None, None, None
    for t in range(120):
        _, rew, done, infos = env.step(fold)
        if first_contact is None and task.torso_steps[0] > 0:
            first_contact = t
            assert not done[0]  # touching alone does not end the episode
        if done[0]:
            end, cause = t, infos[0]["cause"]
            break
    assert first_contact is not None and end is not None
    assert cause == "beached" and end - first_contact >= cfg.beached_steps
    env.close()


# -- viewing direction ---------------------------------------------------------

def _standing_env(seed):
    env = ArcVecEnv(1, TaskConfig(curriculum=False), seed=seed)
    env.reset()
    for _ in range(150):
        env.step(np.zeros((1, 12), dtype=np.float32))
    return env, env.task


def _place_target(task, angle, dist):
    """Puts the checkpoint at `angle` (rad) from the gaze, `dist` meters away."""
    rot = task.sim.rot[0]
    heading = np.arctan2(rot[1, 0], rot[0, 0]) + angle
    task.target[0] = task.sim.root[0, :2] + dist * np.array([np.cos(heading), np.sin(heading)])
    task.prev_dist[0] = dist
    task.prev_gaze[0] = abs(angle)


def test_progress_pays_only_when_target_in_view():
    env, task = _standing_env(8)
    rewards = {}
    for angle in (0.0, np.pi):
        _place_target(task, angle, 3.0)
        task.prev_dist[0] += 0.03  # pretend the robot just walked 3 cm towards it
        rewards[angle] = env.step(np.zeros((1, 12), dtype=np.float32))[1][0]
    assert rewards[0.0] > rewards[np.pi] + 0.8 * task.cfg.w_progress * 0.03
    env.close()


def test_turning_towards_target_pays():
    env, task = _standing_env(9)
    _place_target(task, np.pi / 2, 3.0)
    task.prev_gaze[0] += 0.2  # pretend the eye just turned 0.2 rad towards it
    rew = env.step(np.zeros((1, 12), dtype=np.float32))[1][0]
    assert rew > 0.8 * task.cfg.w_turn * 0.2 - task.cfg.w_time
    env.close()


def test_checkpoint_counts_only_when_seen():
    env, task = _standing_env(10)
    _place_target(task, np.pi, 0.3)  # right behind: not seen
    env.step(np.zeros((1, 12), dtype=np.float32))
    assert task.checkpoints[0] == 0
    _place_target(task, 0.0, 0.3)  # right in front
    env.step(np.zeros((1, 12), dtype=np.float32))
    assert task.checkpoints[0] == 1
    env.close()


def test_latest_checkpoint(tmp_path):
    from arc.runs import MODEL, latest_checkpoint
    assert latest_checkpoint(tmp_path) == tmp_path  # no checkpoints: the run itself
    for steps in (2_000_000, 10_000_000, 4_000_000):
        (tmp_path / "checkpoints" / f"step_{steps}").mkdir(parents=True)
        (tmp_path / "checkpoints" / f"step_{steps}" / MODEL).touch()
    (tmp_path / "checkpoints" / "step_99000000").mkdir()  # no model: ignored
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoints" / "step_10000000"
    assert latest_checkpoint(tmp_path / "checkpoints") == tmp_path / "checkpoints" / "step_10000000"
