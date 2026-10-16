# arc-demo

A leaper-like robot spider learns to cross rough block terrain and reach
randomly placed checkpoints, inspired by Embark Studios'
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
uv run python scripts/train.py --name terrain --steps 40e6   # terrain curriculum
uv run python scripts/train.py --name flat --steps 10e6 --difficulty 0
uv run python scripts/enjoy.py runs/terrain --difficulty 0.6
uv run python scripts/enjoy.py runs/terrain --record out/terrain.gif
uv run python scripts/evaluate.py runs/terrain
uv run tensorboard --logdir runs
```

`train.py` uses half the CPU cores by default (`--threads`) for both the
simulation and torch.

Viewer keys: Space toggles hold/random actions (arcview) or pauses (enjoy),
R new terrain, 0-9 terrain difficulty, C camera (behind / side / front),
Esc/Q quit. Left drag orbits the
camera, the wheel zooms.

## Results

`runs/leaper1`: PPO from scratch with the terrain curriculum, stopped at 26M
steps (about 30 minutes on 8 threads of a Ryzen 7 8840U). Evaluated with
`scripts/evaluate.py` (64 episodes of 33 s per difficulty, deterministic):

| difficulty | checkpoints/min | stalled | falls |
|-----------:|----------------:|--------:|------:|
| 0.00       | 23.0            | 0%      | 0%    |
| 0.25       | 23.6            | 0%      | 0%    |
| 0.50       | 23.7            | 0%      | 0%    |
| 0.75       | 23.8            | 0%      | 3%    |
| 1.00       | 23.5            | 0%      | 3%    |

The robot walks 91-93% of the distance with the checkpoint in its field of
view and brings a new checkpoint into view in 0.33 s (median).

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
- Rigid angular corrections (axis alignment, limits) are 3x3 block solves
  too, so slender segments do not spin up about their long axis.
- Collision geometry is spheres attached to bodies, colliding only with the
  terrain, a grid of axis-aligned columns surrounded by boundary walls. Contacts are re-detected every
  substep. Edges buried by an equal or higher neighbour are skipped, so
  spheres roll across seams without bumps.
- Friction is solved at position level against a persistent per-sphere
  anchor (the touch-down point), static or clamped kinetic
  (Macklin et al. 2014), so planted feet neither creep nor roll; normal
  contact velocity is removed (restitution 0),
  which also discards the separating velocity of penetration corrections,
  so the policy cannot pump energy out of the contact solver.
- Worlds whose state becomes non-finite or absurdly fast are flagged
  unstable; the task terminates their episode.
- A world is a fixed-capacity struct with no pointers: copyable, and
  stepping is deterministic and independent of the thread count.

### Robot (`csrc/arc_robot.c`)

A 19.6 kg spider modelled after ARC Raiders' leaper: a spherical body with a
glowing red ring eye marking the front, hanging below four long legs splayed
at 45 degrees. Each leg has hip yaw, hip pitch and knee joints (12 PD-driven
joints, 250 N m/rad, 80 N m limit). In the default stance the knees are high
above the body and the long blade-like shins come down steeply; the feet are
about 0.8 m from the center. The hip links carry the reflected inertia of
the hip motors ("armature"), which the solver needs to stay stable.

### Task (`arc/env.py`)

- Actions: 12 joint targets as offsets from the default stance
  (0.6 rad per unit action), tracked by the PD drives.
- Observations (body or heading frame): gravity direction, linear and
  angular velocity, joint angles and rates, previous action, direction and
  distance to the checkpoint, a 13x11 heightmap around the torso, foot
  contacts.
- Reward: progress towards the checkpoint (capped at 1.2 m/s, so speed
  beyond that earns nothing), a bonus per checkpoint (torso within 0.5 m,
  the drawn ring), a small time cost, and penalties for joint work, action
  changes, torso/knee contact, torso roll/pitch rate and vertical bounce.
- Viewing direction: the eye looks along the body's forward axis with a
  45 degree half-angle field of view. Progress only pays while the
  checkpoint is in view (fully when dead ahead), turning the eye towards it
  pays per radian, and a checkpoint only counts when reached in view. The
  robot therefore turns to look at a new checkpoint, then walks to it
  eye-first. The marker lights up while the robot sees it.
- Episodes end when the robot flips (tilt > 72 degrees) or rests on its
  torso for over 1 s, with a penalty. The time cost's discounted sum (5)
  stays below the fall penalty (10), so ending an episode on purpose never
  pays, and hovering or orbiting near a checkpoint loses reward.
  `tests/test_env.py` checks these properties; `scripts/evaluate.py` reports
  time spent stalled near checkpoints and fall causes.
- Curriculum: each world has its own terrain level (0 = flat to 1, in steps
  of 0.1, scaling hills, bumps and block jitter). It rises after episodes
  with 3+ checkpoints and drops after falls in the first 10 s without any
  checkpoint, so the population spreads around the edge of what the policy
  can do (as in Rudin et al. 2021, "Learning to Walk in Minutes").
