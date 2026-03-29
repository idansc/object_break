"""CLI entry point for object-break."""

from pathlib import Path

import click

from .pipeline.config import PipelineConfig
from .pipeline.generator import SampleGenerator
from .pipeline.batch import batch_generate


@click.group()
def main():
    """Object Break — fracture simulation for training data generation."""
    pass


@main.command()
@click.option("--mesh", required=True, type=click.Path(exists=True), help="Input mesh file (OBJ, STL, GLB)")
@click.option("--output", required=True, type=click.Path(), help="Output directory")
@click.option("--pieces", default=8, help="Number of pieces to break into")
@click.option("--force", default=5.0, help="Impact force magnitude")
@click.option("--frames", default=60, help="Number of simulation frames")
@click.option("--fps", default=30.0, help="Frames per second")
@click.option("--resolution", default=512, help="Render resolution (square)")
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML config file")
@click.option("--seed", default=None, type=int, help="Random seed")
def generate(mesh, output, pieces, force, frames, fps, resolution, config_path, seed):
    """Generate a single fracture simulation sample."""
    if config_path:
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()

    # Override with CLI args
    config.fracture.num_pieces = pieces
    config.fracture.force_magnitude = force
    config.physics.num_frames = frames
    config.physics.fps = fps
    config.render.resolution = [resolution, resolution]

    gen = SampleGenerator(config, seed=seed)
    metadata = gen.generate(mesh, output)

    click.echo(f"Generated {metadata['num_fragments']} fragments")
    click.echo(f"Output: {output}")


@main.command()
@click.option("--mesh-dir", required=True, type=click.Path(exists=True), help="Directory of input meshes")
@click.option("--output", required=True, type=click.Path(), help="Output directory")
@click.option("--variations", default=1, help="Number of random variations per mesh")
@click.option("--workers", default=1, help="Number of parallel workers")
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML config file")
@click.option("--seed", default=0, type=int, help="Base random seed")
def batch(mesh_dir, output, variations, workers, config_path, seed):
    """Generate fracture samples for all meshes in a directory."""
    if config_path:
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()

    mesh_dir = Path(mesh_dir)
    mesh_paths = sorted(
        p for p in mesh_dir.iterdir()
        if p.suffix.lower() in {".obj", ".stl", ".glb", ".gltf", ".ply", ".off"}
    )

    if not mesh_paths:
        click.echo(f"No mesh files found in {mesh_dir}")
        return

    click.echo(f"Found {len(mesh_paths)} meshes, generating {len(mesh_paths) * variations} samples")

    results = batch_generate(
        mesh_paths=mesh_paths,
        output_dir=output,
        config=config,
        num_variations=variations,
        max_workers=workers,
        base_seed=seed,
    )

    click.echo(f"Successfully generated {len(results)} samples")


if __name__ == "__main__":
    main()
