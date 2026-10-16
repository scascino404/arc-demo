# arc-demo

A quadruped learns to cross rough block terrain and reach randomly placed
checkpoints, inspired by Embark Studios'
[Transforming animation with machine learning](https://medium.com/embarkstudios/transforming-animation-with-machine-learning-27ac694590c).

- `csrc/`: a small C89 physics engine (substepped XPBD), the robot, the terrain
  generator, a batched N-world simulation API parallelized with OpenMP, and an
  SDL2 + OpenGL 2.1 renderer.
- `arc/`: Python bindings (ctypes), the task logic vectorized with numpy, a
  Stable Baselines3 `VecEnv` and a Gymnasium env (`Arc-v0`).
- `scripts/`: training (PPO), playback/recording, benchmarks.

## Setup

Requires gcc with OpenMP, SDL2 and OpenGL development files, and
[uv](https://docs.astral.sh/uv/).

```sh
make            # arc/libarcsim.so, arc/libarcrender.so, bin/arcview
make test       # physics and robot regression tests (C)
uv sync         # Python environment (CPU-only torch)
uv run pytest   # environment tests
```

## Usage

```sh
bin/arcview                                   # physics viewer, no learning
uv run python scripts/train.py --name flat --steps 10e6 --difficulty 0
uv run python scripts/train.py --name terrain --steps 40e6 --resume runs/flat
uv run python scripts/enjoy.py runs/terrain --difficulty 0.6
uv run python scripts/enjoy.py runs/terrain --record out/terrain.gif
uv run tensorboard --logdir runs
```

`train.py` uses half the CPU cores by default (`--threads`) for both the
simulation and torch.

Viewer keys: Space toggles hold/random actions (arcview) or pauses (enjoy),
R new terrain, 0-9 terrain difficulty, Esc/Q quit. Left drag orbits the
camera, the wheel zooms.

## Design

### Physics (`csrc/arc_phys.c`)

Rigid bodies with diagonal inertia, simulated with substepped XPBD
(Müller et al. 2020): 20 substeps per 1/30 s control step, one constraint
iteration per substep.

- Hinge joints: axis alignment, a 3x3 block point-to-point attachment
  (a scalar solve drifts badly when an anchor is far from a body with small
  inertia), limits, and a PD drive implemented as a compliant angular
  constraint (stiffness = 1/compliance) whose impulse is capped by the motor
  torque limit, plus velocity-level damping.
- Collision geometry is spheres attached to bodies, colliding only with the
  terrain, a grid of axis-aligned columns. Contacts are re-detected every
  substep. Edges buried by an equal or higher neighbour are skipped, so
  spheres roll across seams without bumps.
- Friction is solved at position level, static or clamped kinetic
  (Macklin et al. 2014); normal contact velocity is removed (restitution 0),
  which also discards the separating velocity of penetration corrections,
  so the policy cannot pump energy out of the contact solver.
- Worlds whose state becomes non-finite or absurdly fast are flagged
  unstable; the task terminates their episode.
- A world is a fixed-capacity struct with no pointers: copyable, and
  stepping is deterministic and independent of the thread count.

### Task (`arc/env.py`)

- Actions: 12 joint targets as offsets from the default stance
  (0.6 rad per unit action), tracked by the PD drives.
- Observations (body or heading frame): gravity direction, linear and
  angular velocity, joint angles and rates, previous action, direction and
  distance to the checkpoint, a 11x7 heightmap around the torso, foot
  contacts.
- Reward: progress towards the checkpoint, a bonus per checkpoint, small
  penalties for joint work, action changes and knee/shin contacts; falling
  (torso contact, tipping over) ends the episode with a penalty.
- Curriculum: each world has a terrain level that rises after episodes
  with 3+ checkpoints and drops after early falls.
