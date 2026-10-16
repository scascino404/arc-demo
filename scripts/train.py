"""Train a PPO policy for the checkpoint task.

    uv run python scripts/train.py --name terrain --steps 40e6
    uv run python scripts/train.py --name terrain --steps 10e6 --resume runs/terrain

Outputs go to runs/<name>/: final model.zip + vecnormalize.pkl, periodic
checkpoints, and TensorBoard logs (uv run tensorboard --logdir runs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from arc import ArcVecEnv, TaskConfig
from arc.runs import default_threads, load_model, load_normalize, save_run


class TaskStats(BaseCallback):
    """Logs task statistics of finished episodes and saves checkpoints."""

    def __init__(self, out: Path, save_every: int, venv: VecNormalize):
        super().__init__()
        self.out = out
        self.save_every = save_every
        self.venv = venv
        self.next_save = save_every
        self.checkpoints: list[int] = []
        self.falls: list[bool] = []
        self.causes: list[str] = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for i in np.flatnonzero(self.locals["dones"]):
            if "checkpoints" in infos[i]:
                self.checkpoints.append(infos[i]["checkpoints"])
                self.falls.append(infos[i]["fell"])
                self.causes.append(infos[i]["cause"])
        if self.num_timesteps >= self.next_save:
            self.next_save += self.save_every
            save_run(self.out / "checkpoints" / f"step_{self.num_timesteps}", self.model, self.venv)
        return True

    def _on_rollout_end(self) -> None:
        task = self.venv.unwrapped.task
        if self.checkpoints:
            self.logger.record("task/checkpoints_per_episode", float(np.mean(self.checkpoints)))
            self.logger.record("task/fall_rate", float(np.mean(self.falls)))
            for cause in ("flip", "beached", "unstable", "void"):
                self.logger.record(f"task/fall_{cause}", float(np.mean([c == cause for c in self.causes])))
        self.logger.record("task/difficulty_mean", float(task.level.mean()))
        self.logger.record("task/difficulty_max", float(task.level.max()))
        self.checkpoints.clear()
        self.falls.clear()
        self.causes.clear()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="run")
    p.add_argument("--steps", type=float, default=20e6)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=default_threads(),
                   help="threads for both the simulation and torch (default: half the cores)")
    p.add_argument("--n-steps", type=int, default=256, help="rollout length per env")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--difficulty", type=float, default=None,
                   help="fixed terrain difficulty (disables the curriculum)")
    p.add_argument("--max-difficulty", type=float, default=1.0)
    p.add_argument("--resume", type=Path, default=None,
                   help="directory with model.zip and vecnormalize.pkl to continue from")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    out = Path("runs") / args.name

    cfg = TaskConfig(max_difficulty=args.max_difficulty)
    if args.difficulty is not None:
        cfg.curriculum = False
        cfg.difficulty = args.difficulty
    env = VecMonitor(ArcVecEnv(args.envs, cfg, threads=args.threads, seed=args.seed))
    if args.resume:
        env = load_normalize(args.resume, env, training=True)
        model = load_model(args.resume, env, tensorboard_log=str(out / "tb"))
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO(
            "MlpPolicy", env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.envs * args.n_steps // 4,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            max_grad_norm=1.0,
            target_kl=0.03,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                               activation_fn=torch.nn.ELU, log_std_init=-0.5),
            tensorboard_log=str(out / "tb"),
            device="cpu",
            seed=args.seed,
            verbose=1,
        )

    stats = TaskStats(out, save_every=2_000_000, venv=env)
    try:
        model.learn(total_timesteps=int(args.steps), callback=stats,
                    reset_num_timesteps=args.resume is None, tb_log_name="ppo")
    finally:
        save_run(out, model, env)
        print(f"saved to {out}")
        env.close()


if __name__ == "__main__":
    main()
