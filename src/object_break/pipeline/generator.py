"""Single-sample generation: fracture → physics → render."""

import json
from pathlib import Path

import numpy as np
import trimesh

from ..fracture.seeds import impact_biased_seeds, random_surface_point
from ..fracture.cutter import fracture_mesh
from ..physics.simulation import FragmentSimulation
from ..render.renderer import SequenceRenderer
from .config import PipelineConfig


class SampleGenerator:
    """Generates a single fracture simulation sample."""

    def __init__(self, config: PipelineConfig, seed: int | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def generate(self, mesh_path: str | Path, output_dir: str | Path) -> dict:
        """Run the full pipeline for one mesh.

        Args:
            mesh_path: Path to input mesh file (OBJ, STL, GLB, etc).
            output_dir: Directory to write output (video, frames, metadata).

        Returns:
            Metadata dict for this sample.
        """
        mesh_path = Path(mesh_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and normalize mesh
        mesh = trimesh.load(str(mesh_path), force="mesh")
        mesh = _normalize_mesh(mesh)

        # Determine impact point
        fc = self.config.fracture
        if fc.impact_point is not None:
            impact_point = np.array(fc.impact_point)
            impact_normal = np.array(fc.impact_direction or [0, 0, -1])
        else:
            impact_point, impact_normal = random_surface_point(mesh, rng=self.rng)

        impact_direction = -impact_normal  # inward
        if fc.impact_direction is not None:
            impact_direction = np.array(fc.impact_direction)

        # Generate seeds and fracture
        seeds = impact_biased_seeds(
            mesh, impact_point, fc.num_pieces,
            spread=fc.seed_spread, rng=self.rng,
        )
        fragments = fracture_mesh(mesh, seeds)

        if len(fragments) < 2:
            raise RuntimeError(
                f"Fracture produced only {len(fragments)} fragments. "
                "Try increasing num_pieces or adjusting seed_spread."
            )

        # Physics simulation
        pc = self.config.physics
        sim = FragmentSimulation(
            fragments=fragments,
            mode=pc.mode,
            impact_point=impact_point,
            impact_direction=impact_direction,
            force=fc.force_magnitude,
            gravity=np.array(pc.gravity),
            velocity_scale=pc.velocity_scale,
            angular_velocity_scale=pc.angular_velocity_scale,
            damping=pc.damping,
            rng=self.rng,
        )
        sim_result = sim.run(
            num_frames=pc.num_frames, fps=pc.fps, hold_frames=pc.hold_frames
        )

        # Render
        rc = self.config.render
        renderer = SequenceRenderer(
            resolution=tuple(rc.resolution),
            bg_color=rc.bg_color,
            camera_distance=rc.camera_distance,
            camera_elevation=rc.camera_elevation,
            camera_azimuth=rc.camera_azimuth,
        )

        # Render intact mesh
        renderer.render_intact(mesh, output_dir / "intact.png")

        # Render breaking sequence
        renderer.render_sequence(
            fragments=fragments,
            sim_result=sim_result,
            output_dir=output_dir,
            intact_mesh=mesh,
            save_video=rc.save_video,
            save_frames=rc.save_frames,
        )

        # Export fragment meshes
        fragments_dir = output_dir / "fragments"
        fragments_dir.mkdir(exist_ok=True)
        for i, frag in enumerate(fragments):
            frag.export(str(fragments_dir / f"fragment_{i:03d}.obj"))

        # Write metadata
        metadata = {
            "source_mesh": str(mesh_path.name),
            "random_seed": self.seed,
            "num_fragments": len(fragments),
            "fracture_params": {
                "num_pieces": fc.num_pieces,
                "impact_point": impact_point.tolist(),
                "impact_direction": impact_direction.tolist(),
                "force_magnitude": fc.force_magnitude,
                "seed_spread": fc.seed_spread,
            },
            "physics_params": {
                "gravity": pc.gravity,
                "num_frames": pc.num_frames,
                "fps": pc.fps,
                "velocity_scale": pc.velocity_scale,
                "angular_velocity_scale": pc.angular_velocity_scale,
            },
            "render_params": {
                "resolution": rc.resolution,
                "camera_distance": rc.camera_distance,
                "camera_elevation": rc.camera_elevation,
                "camera_azimuth": rc.camera_azimuth,
            },
            "fragments": [
                {
                    "id": i,
                    "volume": float(f.volume),
                    "centroid": f.centroid.tolist(),
                }
                for i, f in enumerate(fragments)
            ],
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata


def _normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center mesh at origin and scale to fit in unit sphere."""
    mesh.vertices -= mesh.centroid
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if scale > 0:
        mesh.vertices /= scale
    return mesh
