#!/usr/bin/env python3
"""Demo: fracture a shape and render the breaking sequence.

Usage:
    python scripts/demo.py                                    # explode, 3 pieces, 5s
    python scripts/demo.py --mode scatter --pieces 5
    python scripts/demo.py --mode split_half --shape box
    python scripts/demo.py --all-modes                        # one video per mode
    python scripts/demo.py --shape cylinder --radius 0.4 --height 1.5 --pieces 6
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from object_break.fracture.seeds import impact_biased_seeds, random_surface_point
from object_break.fracture.cutter import fracture_mesh
from object_break.physics.simulation import FragmentSimulation, EXPLOSION_MODES
from object_break.render.renderer import SequenceRenderer


SHAPES = {
    "sphere": lambda r, h: trimesh.creation.icosphere(subdivisions=3, radius=r),
    "box": lambda r, h: trimesh.creation.box(extents=[r * 2, r * 2, r * 2]),
    "cylinder": lambda r, h: trimesh.creation.cylinder(radius=r, height=h),
    "cone": lambda r, h: trimesh.creation.cone(radius=r, height=h),
}


def generate_one(
    mesh: trimesh.Trimesh,
    fragments: list[trimesh.Trimesh],
    impact_point: np.ndarray,
    impact_direction: np.ndarray,
    mode: str,
    args,
    output_dir: Path,
    seed: int,
):
    """Generate one video for a given explosion mode."""
    rng = np.random.default_rng(seed)

    hold_frames = int(args.hold * args.fps)
    sim_frames = int(args.duration * args.fps) - hold_frames
    sim_frames = max(sim_frames, 1)

    sim = FragmentSimulation(
        fragments=fragments,
        mode=mode,
        impact_point=impact_point,
        impact_direction=impact_direction,
        force=args.force,
        gravity=np.array(args.gravity),
        velocity_scale=args.velocity,
        angular_velocity_scale=2.0,
        damping=0.005,
        rng=rng,
    )
    sim_result = sim.run(num_frames=sim_frames, fps=args.fps, hold_frames=hold_frames)

    renderer = SequenceRenderer(
        resolution=(args.resolution, args.resolution),
        camera_distance=args.radius * 5.0,
        camera_elevation=25.0,
        camera_azimuth=45.0,
        show_trails=args.trails,
        trail_length=int(args.fps * 0.5),
    )
    video_path = renderer.render_sequence(
        fragments=fragments,
        sim_result=sim_result,
        output_dir=output_dir,
        intact_mesh=mesh,
        save_video=True,
        save_frames=not args.all_modes,  # skip frames in all-modes to save time
    )

    # Export fragments
    frags_dir = output_dir / "fragments"
    frags_dir.mkdir(exist_ok=True)
    for i, frag in enumerate(fragments):
        frag.export(str(frags_dir / f"fragment_{i:03d}.obj"))

    return video_path


def main():
    parser = argparse.ArgumentParser(description="Object Break Demo")
    parser.add_argument("--shape", default="sphere", choices=SHAPES.keys())
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.5)
    parser.add_argument("--pieces", type=int, default=3)
    parser.add_argument("--mode", default="explode", choices=EXPLOSION_MODES,
                        help=f"Explosion mode (default: explode). Options: {', '.join(EXPLOSION_MODES)}")
    parser.add_argument("--all-modes", action="store_true",
                        help="Generate one video for each explosion mode")
    parser.add_argument("--force", type=float, default=3.0)
    parser.add_argument("--velocity", type=float, default=1.5)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--hold", type=float, default=0.3)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--trails", action="store_true")
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, 0],
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="demo_output")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Create shape
    print(f"Creating {args.shape} (radius={args.radius})...")
    mesh = SHAPES[args.shape](args.radius, args.height)

    # Impact point
    impact_point, impact_normal = random_surface_point(mesh, rng=rng)
    impact_direction = -impact_normal

    # Fracture
    print(f"Fracturing into {args.pieces} pieces...")
    seeds = impact_biased_seeds(mesh, impact_point, args.pieces, spread=2.0, rng=rng)
    fragments = fracture_mesh(mesh, seeds)
    print(f"Got {len(fragments)} fragments")

    if args.all_modes:
        # Generate one video per explosion mode
        print(f"\nGenerating all {len(EXPLOSION_MODES)} modes...")
        for mode in EXPLOSION_MODES:
            out_dir = Path(args.output) / mode
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- {mode} ---")
            video = generate_one(
                mesh, fragments, impact_point, impact_direction,
                mode, args, out_dir, seed=args.seed,
            )
            print(f"  {video}")
        print(f"\nDone! All videos in {args.output}/")
    else:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nMode: {args.mode}")
        video = generate_one(
            mesh, fragments, impact_point, impact_direction,
            args.mode, args, out_dir, seed=args.seed,
        )
        print(f"\nDone! {len(fragments)} fragments, {args.duration}s video")
        print(f"Video: {video}")


if __name__ == "__main__":
    main()
