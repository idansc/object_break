# Object Break

Simulator for objects breaking apart into pieces. Generates video frame sequences showing fracture and piece separation — designed as training data for video diffusion models (LTX, Wan2.1, etc.).

---

## Quick Start — Try It Now!

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Run the demo — sphere breaks into 3 pieces
python scripts/demo.py

# 3. Try different explosion modes
python scripts/demo.py --mode explode --pieces 3 --trails
python scripts/demo.py --mode impact --pieces 5 --shape box
python scripts/demo.py --mode scatter --pieces 8 --shape cylinder

# 4. Generate ALL modes at once
python scripts/demo.py --all-modes --pieces 3 --trails

# 5. Open the interactive notebook
jupyter notebook notebooks/demo.ipynb
```

---

## Explosion Modes

Six different ways objects can break apart. Each mode produces visually distinct motion patterns:

### `explode` — Outward Explosion
All pieces fly **outward from the center** in every direction. Classic explosion effect. Each piece moves radially away with uniform speed.

```bash
python scripts/demo.py --mode explode --pieces 5
```

### `impact` — Directional Impact
Simulates a **hit from one side**. Pieces near the impact point fly faster, pieces further away move slower. Velocity is biased along the impact direction — like a ball smashing through a wall.

```bash
python scripts/demo.py --mode impact --pieces 5
```

### `split_half` — Clean Split
Object **splits along a random axis** into two groups that separate in opposite directions. Like cleanly cracking something in two. Pieces on each side move together.

```bash
python scripts/demo.py --mode split_half --pieces 4
```

### `scatter` — Random Chaos
Each piece gets a **completely random direction and speed**. Unpredictable, chaotic breakage with no spatial pattern.

```bash
python scripts/demo.py --mode scatter --pieces 6
```

### `peel` — Surface Peel
Pieces **peel off the surface** with a strong tangential component. More graceful, curving motion — like a shell or coating coming apart layer by layer.

```bash
python scripts/demo.py --mode peel --pieces 5
```

### `directional` — Shockwave
All pieces fly in **roughly the same direction** with some random spread. Like being hit by a blast wave or strong wind pushing everything one way.

```bash
python scripts/demo.py --mode directional --pieces 5
```

---

## Example Videos

Pre-generated example videos are in `examples/`. Each shows a sphere breaking into 3 pieces with trajectory trails:

```
examples/
├── explode/video.mp4
├── impact/video.mp4
├── split_half/video.mp4
├── scatter/video.mp4
├── peel/video.mp4
└── directional/video.mp4
```

---

## All Demo Options

```
python scripts/demo.py [OPTIONS]

Shape:
  --shape {sphere,box,cylinder,cone}   Object shape (default: sphere)
  --radius FLOAT                       Size of the object (default: 1.0)
  --height FLOAT                       Height for cylinder/cone (default: 1.5)

Fracture:
  --pieces INT                         Number of fragments (default: 3)
  --mode MODE                          Explosion mode (default: explode)
                                       Options: explode, impact, split_half,
                                       scatter, peel, directional

Physics:
  --force FLOAT                        Impact force (default: 3.0)
  --velocity FLOAT                     Velocity multiplier (default: 1.5)
  --gravity X Y Z                      Gravity vector (default: 0 0 0)

Video:
  --duration FLOAT                     Video length in seconds (default: 5.0)
  --hold FLOAT                         Seconds showing intact object (default: 0.3)
  --fps FLOAT                          Frames per second (default: 30)
  --resolution INT                     Image size (default: 512)
  --trails                             Show trajectory trails behind pieces

Batch:
  --all-modes                          Generate one video per explosion mode
  --seed INT                           Random seed (default: 42)
  --output DIR                         Output directory (default: demo_output)
```

---

## How It Works

### 1. Fracture (Voronoi Tessellation)
The input mesh is split into fragments using 3D Voronoi cells:
- Seed points are generated near the impact point (power-law radial distribution — more seeds near impact = smaller pieces there)
- `scipy.spatial.Voronoi` computes 3D cells from the seeds
- Each cell is intersected with the original mesh using boolean operations (`trimesh` + `manifold3d`)
- Result: a list of watertight fragment meshes

### 2. Physics Simulation
Each fragment gets initial velocity based on the chosen explosion mode, plus random angular velocity for tumbling:
- **Position**: Euler integration with configurable gravity and damping
- **Rotation**: Rodrigues' formula for smooth 3D rotation from angular velocity
- **Hold phase**: optionally show the intact object for N frames before breaking

### 3. Rendering
PyVista renders each frame offscreen:
- Each fragment gets a distinct color
- Optional trajectory trails show the path of each piece
- During hold phase, the intact mesh is rendered as a single gray object
- Output: MP4 video + individual PNG frames

### 4. Output
```
output/
├── video.mp4           # Video sequence
├── intact.png          # Render of intact object
├── metadata.json       # All parameters and fragment info
├── frames/
│   ├── frame_0000.png  # Individual frames
│   └── ...
└── fragments/
    ├── fragment_000.obj  # Fragment meshes (OBJ)
    └── ...
```

---

## Interactive Notebook

For the best experience, use the Jupyter notebook:

```bash
pip install jupyter
jupyter notebook notebooks/demo.ipynb
```

The notebook lets you tweak every parameter and see the video inline. It includes examples for all 6 explosion modes and shows how to generate training data in batch.

---

## CLI for Custom Meshes

```bash
# Single mesh
object-break generate --mesh input.obj --pieces 12 --output ./output

# Batch (all meshes in a directory)
object-break batch --mesh-dir ./meshes --output ./dataset --variations 5 --workers 4
```

---

## Python API

```python
from object_break.pipeline.config import PipelineConfig
from object_break.pipeline.generator import SampleGenerator

config = PipelineConfig()
config.fracture.num_pieces = 8
config.fracture.force_magnitude = 5.0
config.physics.mode = "explode"

gen = SampleGenerator(config, seed=42)
metadata = gen.generate("input.obj", "./output")
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
