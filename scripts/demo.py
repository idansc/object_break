#!/usr/bin/env python3
"""Demo: fracture a primitive shape and render the breaking sequence."""

from pathlib import Path

import numpy as np
import trimesh

from object_break.fracture.seeds import impact_biased_seeds, random_surface_point
from object_break.fracture.cutter import fracture_mesh
from object_break.physics.simulation import FragmentSimulation
from object_break.render.renderer import SequenceRenderer


def main():
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    # Create a sample mesh (sphere, box, or cylinder)
    print("Creating sample mesh...")
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    # Alternative shapes:
    # mesh = trimesh.creation.box(extents=[1.5, 1.0, 0.8])
    # mesh = trimesh.creation.cylinder(radius=0.5, height=1.5)

    rng = np.random.default_rng(42)

    # Pick impact point on surface
    impact_point, impact_normal = random_surface_point(mesh, rng=rng)
    impact_direction = -impact_normal
    print(f"Impact point: {impact_point}")
    print(f"Impact direction: {impact_direction}")

    # Generate fracture seeds
    num_pieces = 12
    print(f"Generating {num_pieces} fracture seeds...")
    seeds = impact_biased_seeds(mesh, impact_point, num_pieces, spread=2.0, rng=rng)
    print(f"Seeds shape: {seeds.shape}")

    # Fracture the mesh
    print("Fracturing mesh...")
    fragments = fracture_mesh(mesh, seeds)
    print(f"Got {len(fragments)} fragments")

    for i, f in enumerate(fragments):
        vol = f.volume
        print(f"  Fragment {i}: volume={vol:.4f}, vertices={len(f.vertices)}, faces={len(f.faces)}")

    # Run physics simulation
    print("Running physics simulation...")
    sim = FragmentSimulation(
        fragments=fragments,
        impact_point=impact_point,
        impact_direction=impact_direction,
        force=5.0,
        gravity=np.array([0.0, 0.0, -9.81]),
        velocity_scale=2.0,
        angular_velocity_scale=3.0,
        rng=rng,
    )
    sim_result = sim.run(num_frames=60, fps=30.0)
    print(f"Simulated {sim_result.num_frames} frames")

    # Render
    print("Rendering frame sequence...")
    renderer = SequenceRenderer(
        resolution=(512, 512),
        camera_distance=4.0,
        camera_elevation=25.0,
        camera_azimuth=45.0,
    )
    video_path = renderer.render_sequence(
        fragments=fragments,
        sim_result=sim_result,
        output_dir=output_dir,
        save_video=True,
        save_frames=True,
    )

    print(f"\nDone! Output saved to: {output_dir}")
    if video_path:
        print(f"Video: {video_path}")
    print(f"Frames: {output_dir / 'frames'}")

    # Export fragments
    frags_dir = output_dir / "fragments"
    frags_dir.mkdir(exist_ok=True)
    for i, frag in enumerate(fragments):
        frag.export(str(frags_dir / f"fragment_{i:03d}.obj"))
    print(f"Fragments: {frags_dir}")


if __name__ == "__main__":
    main()
