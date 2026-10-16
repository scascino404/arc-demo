# arc-demo

A four-legged robot spider learns, by reinforcement learning, to walk over
rough block terrain and reach checkpoints placed at random around it.
Inspired by Embark Studios'
[Transforming animation with machine learning](https://medium.com/embarkstudios/transforming-animation-with-machine-learning-27ac694590c).

The physics engine and renderer are written from scratch in C; the task and
training are in Python, using [Stable Baselines3](https://stable-baselines3.readthedocs.io/) PPO.

## Setup

You need gcc with OpenMP, the SDL2 and OpenGL development files (the
renderer needs OpenGL 2.1 with GLSL 1.20), and
[uv](https://docs.astral.sh/uv/).

```sh
make            # build the native libraries and the viewer
uv sync         # create the Python environment (CPU-only torch)
make test       # physics and robot tests (C)
uv run pytest   # environment tests (Python)
```

`make` optimizes for the CPU it runs on. To build libraries for other
machines, use `make OPT="-O3 -g"`.

## Usage

### Watch the trained robot

```sh
uv run python scripts/enjoy.py models/leaper --difficulty 0.6
```

`models/leaper` is a trained policy that comes with the repository (see
[Results](#results)).

### Look at the physics

```sh
bin/arcview
```

Opens the simulation without any learning: the robot holds its stance, or
moves its joints randomly. If it falls off the edge, it starts over.

### Train a policy

```sh
uv run python scripts/train.py --name terrain --steps 40e6
```

This starts on flat ground and makes the terrain harder as the robot gets
better. Use `--difficulty 0` to train on flat ground only. The run is saved
in `runs/terrain/`, with checkpoints every 2M steps. To train it for
`--steps` more steps, add `--resume runs/terrain` to the same command; a
resumed run fine-tunes the policy, with a learning rate that decays to 0.
Follow training with `uv run tensorboard --logdir runs`.

Training uses half of the CPU cores by default (`--threads`); on most
machines that is one thread per physical core, which is the fastest setting.

### Watch or record a policy

```sh
uv run python scripts/enjoy.py runs/terrain --difficulty 0.6
uv run python scripts/enjoy.py runs/terrain --record out/terrain.gif
```

Given a run directory, `enjoy.py` plays its checkpoint with the most steps.

### Measure a policy

```sh
uv run python scripts/evaluate.py runs/terrain
```

Prints checkpoints reached per minute, the share of time stalled near a
checkpoint without reaching it, and how often the robot falls, for terrain
difficulties from 0 to 1.

### Viewer controls

| key | `arcview` | `enjoy.py` |
|---|---|---|
| Space | hold stance / random actions | pause |
| R | new terrain | new terrain |
| 0-9 | terrain difficulty | terrain difficulty |
| C | camera: behind / side / front | camera: behind / side / front |
| P, N, S | pause, single step, slow motion | |
| Esc, Q | quit | quit |

Drag with the left mouse button to orbit the camera; scroll to zoom.

## The task

- **The robot** is modelled on the leaper from ARC Raiders: a round,
  paneled body with a red ring eye at the front, and four long armored
  legs, each with three motorized joints (12 in total).
- **The arena** is a 25.6 m square of blocks floating in a black void.
- **Actions** are target angles for the 12 joints.
- **Observations** include the body's orientation and velocity, the joint
  angles, the direction and distance to the checkpoint, the terrain height
  around the robot, and which feet touch the ground.
- **Rewards** are for moving towards the checkpoint and for reaching it.
  Small penalties discourage wasted energy, jerky motion, bouncing, and
  scraping the body or knees on the ground.
- **The robot has to look where it goes.** Progress only counts while the
  checkpoint is within the eye's field of view (90 degrees wide), so the
  robot learns to turn towards a new checkpoint first and then walk to it.
  The checkpoint marker lights up while the robot sees it.
- **Episodes** last 33 s, or end early when the robot flips over, lies on
  its body for more than a second, or walks off the edge into the void.
- **Terrain difficulty** goes from 0 (flat) to 1 (hills and steps). During
  training, each robot's difficulty rises when it does well and drops when it
  falls early, so training keeps pace with what the robot can manage.

## Results

`models/leaper` was trained from scratch with the default settings for 60M
steps, then fine-tuned with `--resume` for 15M more at `--lr 3e-4`
(about 50 minutes in all on 8 cores of a Ryzen 7 8840U laptop):

```sh
uv run python scripts/train.py --name leaper --steps 60e6
uv run python scripts/train.py --name leaper --steps 15e6 --lr 3e-4 --resume runs/leaper
```

With `scripts/evaluate.py --envs 512` (512 episodes of 33 s per difficulty):

| difficulty | checkpoints/min | stalled | falls |
|-----------:|----------------:|--------:|------:|
| 0.00       | 24.3            | 0%      | 0.0%  |
| 0.25       | 24.4            | 0%      | 0.0%  |
| 0.50       | 24.5            | 0%      | 0.2%  |
| 0.75       | 24.4            | 0%      | 0.2%  |
| 1.00       | 24.0            | 0%      | 2.1%  |

The fine-tuning matters: after the first 60M steps, the robot fell in 4.3%
of the episodes at difficulty 1.

The default settings came out of a sweep of 3M-step runs (under 3 minutes
each), judged on how reliably and how soon training takes off, and on
checkpoints and falls at the end. Compared with the previous settings,
8 minibatches instead of 4, a learning rate of 6e-4 instead of 3e-4, and a
fall penalty of 20 instead of 10 made training take off on every seed
(after about 0.6M steps) instead of on half of them, and train about 20%
faster. More epochs, more envs per update, a smaller network, or a lower
target KL were all slower to converge.

On the same laptop, training runs at about 25k steps per second.
`make bench` and `scripts/bench_env.py` measure the raw simulation speed.

## Project layout

- `csrc/`: the C code. A small rigid-body physics engine (`arc_phys.c`), the
  robot (`arc_robot.c`), the terrain (`arc_terrain.c`), a simulation of many
  worlds stepped in parallel (`arc_sim.c`), and the renderer
  (`arc_render.c`: OpenGL 2.1 with small GLSL shaders for per-pixel
  lighting, a shadow map, fog and glow). Tests are in `csrc/tests/`, the
  viewer and benchmark in `csrc/tools/`.
- `arc/`: the Python package. Bindings to the C libraries (`native.py`),
  the task: observations, rewards and terrain difficulty (`task.py`),
  environments for Stable Baselines3 and Gymnasium (`env.py`, registered as
  `Arc-v0`), and saving and loading training runs (`runs.py`).
- `scripts/`: training, playback, evaluation, and benchmarking.
- `models/leaper/`: the trained policy described in [Results](#results).
- `tests/`: Python tests for the environment and rewards.

The physics uses substepped XPBD
([Müller et al. 2020](https://matthias-research.github.io/pages/publications/PBDBodies.pdf));
the terrain curriculum follows
[Rudin et al. 2021](https://arxiv.org/abs/2109.11978). Simulation is
deterministic: the same seed gives the same result for any number of
threads.
