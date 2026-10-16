"""Watch a trained policy in the viewer, or record it to a GIF.

    uv run python scripts/enjoy.py runs/flat1
    uv run python scripts/enjoy.py runs/flat1 --difficulty 0.6
    uv run python scripts/enjoy.py runs/flat1 --record out/flat1.gif --seconds 20

Keys: R new terrain, 0-9 terrain difficulty, Space pause, C camera view,
Esc/Q quit.
Mouse: left drag orbits the camera, wheel zooms.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from arc import ArcVecEnv, TaskConfig

KEY_R, KEY_SPACE, KEY_0, KEY_9 = ord("r"), ord(" "), ord("0"), ord("9")


def load(run: Path, cfg: TaskConfig, render_mode: str, seed: int):
    env = ArcVecEnv(1, cfg, seed=seed, render_mode=render_mode)
    env = VecNormalize.load(str(run / "vecnormalize.pkl"), env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(run / "model.zip", device="cpu")
    return env, model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="directory with model.zip and vecnormalize.pkl")
    p.add_argument("--difficulty", type=float, default=0.0)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    p.add_argument("--record", type=Path, default=None, help="write a GIF instead of opening a window")
    p.add_argument("--seconds", type=float, default=20.0, help="recording length")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = TaskConfig(curriculum=False, difficulty=args.difficulty, max_steps=10**9)
    env, model = load(args.run, cfg, "rgb_array" if args.record else "human", args.seed)
    arc_env: ArcVecEnv = env.venv
    task = arc_env.task
    obs = env.reset()

    if args.record:
        frames = []
        for t in range(int(args.seconds / cfg.dt)):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, done, _ = env.step(action)
            img = env.render()
            if t % 2 == 0:  # 15 fps keeps the GIF small
                frames.append(Image.fromarray(img).quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        args.record.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.record, save_all=True, append_images=frames[1:], duration=66, loop=0)
        print(f"{args.record}: {len(frames)} frames, {task.checkpoints[0]} checkpoints reached")
        env.close()
        return

    paused = False
    next_t = time.perf_counter()
    while True:
        renderer = arc_env._view.renderer
        for key in renderer.keys() if renderer else []:
            if key == KEY_SPACE:
                paused = not paused
            elif key == KEY_R or KEY_0 <= key <= KEY_9:
                if key != KEY_R:
                    task.level[:] = (key - KEY_0) / 9.0
                obs = env.reset()
        if not paused:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, done, infos = env.step(action)
            if done[0]:
                print(f"episode end: {infos[0]['checkpoints']} checkpoints, fell={infos[0]['fell']}")
        env.render()
        if arc_env.window_closed:
            break
        next_t += cfg.dt
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.perf_counter()
    env.close()


if __name__ == "__main__":
    main()
