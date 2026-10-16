"""End-to-end VecEnv throughput (physics + Python task logic), random actions."""

import argparse
import time

import numpy as np

from arc import ArcVecEnv, TaskConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seconds", type=float, default=3.0)
    args = p.parse_args()
    for n in args.envs:
        cfg = TaskConfig(difficulty=0.7, curriculum=False)
        env = ArcVecEnv(n, cfg, threads=args.threads, seed=0)
        sim = env.task.sim
        env.reset()
        rng = np.random.default_rng(0)
        actions = rng.uniform(-1, 1, (64, n, sim.num_joints)).astype(np.float32)
        steps, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            env.step_async(actions[steps % 64])
            env.step_wait()
            steps += 1
        el = time.perf_counter() - t0
        # time of the native step alone, for the overhead split
        t1 = time.perf_counter()
        for k in range(50):
            sim.step(sim.default_angles + cfg.action_scale * actions[k % 64])
        t_sim = (time.perf_counter() - t1) / 50
        print(f"{n:4d} envs x {args.threads} threads: {steps * n / el:8.0f} steps/s "
              f"(native step {1e3 * t_sim:.2f} ms of {1e3 * el / steps:.2f} ms per batch)")
        env.close()


if __name__ == "__main__":
    main()
