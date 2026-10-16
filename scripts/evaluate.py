"""Evaluate a policy: checkpoints per minute and fall rate per terrain difficulty.

    uv run python scripts/evaluate.py runs/flat1 --difficulties 0 0.3 0.6 1.0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from arc import ArcVecEnv, TaskConfig


def evaluate(run: Path, difficulty: float, envs: int, steps: int, threads: int, seed: int):
    cfg = TaskConfig(curriculum=False, difficulty=difficulty, max_steps=steps)
    env = VecNormalize.load(str(run / "vecnormalize.pkl"), ArcVecEnv(envs, cfg, threads=threads, seed=seed))
    env.training = False
    env.norm_reward = False
    model = PPO.load(run / "model.zip", device="cpu")
    obs = env.reset()
    checkpoints = np.zeros(envs)
    alive_steps = np.zeros(envs)
    fell = np.zeros(envs, dtype=bool)
    finished = np.zeros(envs, dtype=bool)
    # one episode per env: stop counting an env after its first episode ends
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        alive_steps += ~finished
        for i in np.flatnonzero(done & ~finished):
            checkpoints[i] = infos[i]["checkpoints"]
            fell[i] = infos[i]["fell"]
        finished |= done
        if finished.all():
            break
    env.close()
    minutes = alive_steps.sum() * cfg.dt / 60.0
    return checkpoints.sum() / minutes, fell.mean(), checkpoints.mean()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--difficulties", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000, help="episode length (1000 = 33 s)")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    print(f"{args.run}: {args.envs} episodes of up to {args.steps * TaskConfig.dt:.0f} s per difficulty")
    print("difficulty  checkpoints/min  checkpoints/episode  fall rate")
    for d in args.difficulties:
        rate, fall, per_ep = evaluate(args.run, d, args.envs, args.steps, args.threads, args.seed)
        print(f"{d:10.2f}  {rate:15.1f}  {per_ep:19.2f}  {fall:9.0%}")


if __name__ == "__main__":
    main()
