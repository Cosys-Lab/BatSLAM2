# Simulator (MATLAB)

The simulator renders the datasets BatSLAM 2.0 runs on: a 2D world of reflectors, a drive through it,
and for every sonar pulse the binaural echo that the robot's two ears receive, plus ground truth and
noisy odometry.

```matlab
cd sim
generate_dataset()                                   % the hall, into ../data/hall.mat
generate_dataset(struct('scene', 'city', 'outFile', '../data/city.mat'))
make_paper_datasets                                  % every dataset used in the paper
```

The options (scene, route seed and length, pulse spacing, sensor noise, range, odometry noise and bias)
are listed with their defaults at the top of `generate_dataset.m`. Rendering uses `parfor` when the
Parallel Computing Toolbox is available and runs serially otherwise.

## Worlds and drives

| File | What it builds |
|---|---|
| `buildHallScene.m`, `buildRoute.m` | A 16 × 10 m hall with two islands and a hand-designed multi-loop route, including loops in the reverse direction and a deliberate aliasing trap. |
| `buildIndoorScene.m` | An indoor floor of 38 × 18 m: a 3 × 3 grid of rooms with 1.5 m corridors and a pillar hall. |
| `buildCityScene.m` | A city of 46 × 26 m: blocks with 3–4 m streets, a plaza and a pillar hall. |
| `buildCityRoute.m` | A long random tour through the street or corridor graph of the city and the indoor floor, which revisits places after long excursions in both directions. |
| `followPath.m` | Samples a route at constant spacing (one pulse every 15 cm). |

All three worlds are deliberately self-similar (repeated blocks, identical pillar patterns at several
places), so that appearance alone is often ambiguous.

## Sonar model (`simulator/src/`)

| File | Role |
|---|---|
| `SonarRenderer.m` | Synthesizes the FM call and renders the binaural echo: for every reflector and each ear, emitter directivity × spherical spreading × air absorption × reflectivity × ear directivity, summed, plus sensor noise. |
| `HRTFModel.m` | Loads the head-related transfer function and interpolates the emitter and ear gains. |
| `BatEnvironment.m` | Point, circle (pillar) and wall reflectors, optionally with a spectral signature. |
| `BatRobot.m` | Pose of the robot, the head and the two ears. |
| `Geometry.m`, `atmosphericAbsorption.m` | Frame conventions, and air absorption after ISO 9613-1. |

The ears and emitter are those of the bat *Phyllostomus discolor*
(`simulator/data/HRTF_Phyllostomus_Discolor_azel.mat`, De Mey et al., 2008). The gains are stored as
`[azimuth × elevation × frequency]`, the order in which `HRTFModel` interpolates them.
