"""Evaluate a policy per terrain difficulty: checkpoints per minute, time
stalled near a checkpoint, falls and their causes.

    uv run python scripts/evaluate.py runs/terrain --difficulties 0 0.3 0.6 1.0
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from arc import TaskConfig
from arc.runs import default_threads, load_policy


def evaluate(run: Path, difficulty: float, envs: int, steps: int, threads: int, seed: int,
             stall_seconds: float = 3.0, stall_radius: float = 1.5):
    """One episode per env. Returns checkpoints per minute, fall rate, fall
    causes, checkpoints per episode and the fraction of time spent stalled:
    within stall_radius of the current checkpoint for over stall_seconds."""
    cfg = TaskConfig(curriculum=False, difficulty=difficulty, max_steps=steps)
    env, act = load_policy(run, envs, cfg, threads, seed)
    task = env.unwrapped.task
    obs = env.reset()
    checkpoints = np.zeros(envs)
    alive_steps = np.zeros(envs)
    near_steps = np.zeros(envs)       # consecutive steps near the current checkpoint
    stalled_steps = np.zeros(envs)
    fell = np.zeros(envs, dtype=bool)
    causes: Counter = Counter()
    finished = np.zeros(envs, dtype=bool)
    window = stall_seconds / cfg.dt
    for _ in range(steps):
        target = task.target.copy()
        obs, _, done, infos = env.step(act(obs))
        live = ~finished
        alive_steps += live
        near = np.linalg.norm(target - task.sim.root[:, :2], axis=1) < stall_radius
        near &= np.all(task.target == target, axis=1)  # not just reached
        near_steps = np.where(near & live, near_steps + 1, 0)
        stalled_steps += (near_steps > window) & live
        for i in np.flatnonzero(done & live):
            checkpoints[i] = infos[i]["checkpoints"]
            fell[i] = infos[i]["fell"]
            if fell[i]:
                causes[infos[i]["cause"]] += 1
        finished |= done
        if finished.all():
            break
    env.close()
    minutes = alive_steps.sum() * cfg.dt / 60.0
    return (checkpoints.sum() / minutes, fell.mean(), causes, checkpoints.mean(),
            stalled_steps.sum() / alive_steps.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--difficulties", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000, help="episode length (1000 = 33 s)")
    p.add_argument("--threads", type=int, default=default_threads())
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    print(f"{args.run}: {args.envs} episodes of up to {args.steps * TaskConfig.dt:.0f} s per difficulty")
    print("difficulty  checkpoints/min  per episode  stalled  falls  causes")
    for d in args.difficulties:
        rate, fall, causes, per_ep, stalled = evaluate(args.run, d, args.envs, args.steps,
                                                       args.threads, args.seed)
        detail = ", ".join(f"{k} {v}" for k, v in sorted(causes.items()))
        print(f"{d:10.2f}  {rate:15.1f}  {per_ep:11.2f}  {stalled:7.0%}  {fall:5.0%}  {detail}")


if __name__ == "__main__":
    main()
