# Object Break

Simulator for objects breaking apart into pieces. Generates video frame sequences showing fracture and piece separation — designed as training data for video diffusion models (LTX, Wan2.1, etc.).

## How it works

1. **Fracture**: Takes a 3D mesh and splits it into fragments using Voronoi tessellation with impact-biased seed placement
2. **Physics**: Simulates fragments flying apart with rigid body dynamics (velocity, rotation, gravity)
3. **Render**: Renders each frame using PyVista offscreen rendering
4. **Export**: Saves video (mp4), individual frames (png), fragment meshes (obj), and metadata (json)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Single mesh
```bash
object-break generate --mesh input.obj --pieces 12 --output ./output
```

### Batch (all meshes in a directory)
```bash
object-break batch --mesh-dir ./meshes --output ./dataset --variations 5 --workers 4
```

### Python API
```python
from object_break.pipeline.config import PipelineConfig
from object_break.pipeline.generator import SampleGenerator

config = PipelineConfig()
config.fracture.num_pieces = 12
config.fracture.force_magnitude = 8.0

gen = SampleGenerator(config, seed=42)
metadata = gen.generate("input.obj", "./output")
```

### Demo
```bash
python scripts/demo.py
```

## Configuration

See `config/default.yaml` for all options. Key parameters:

- `fracture.num_pieces` — number of fragments (2-100)
- `fracture.force_magnitude` — how hard the impact is
- `fracture.seed_spread` — higher = more uniform pieces, lower = concentrated at impact
- `physics.num_frames` — length of the video
- `physics.gravity` — gravity vector (set to [0,0,0] for zero-g)
- `render.resolution` — output image size

## Output format

```
output/
├── video.mp4           # Video of pieces separating
├── intact.png          # Render of the intact object
├── metadata.json       # All parameters and fragment info
├── frames/
│   ├── frame_0000.png
│   └── ...
└── fragments/
    ├── fragment_000.obj
    └── ...
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
