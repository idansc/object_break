"""Batch generation with parameter sweeps."""

import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from .config import PipelineConfig
from .generator import SampleGenerator


def generate_one(args: tuple) -> dict | str:
    """Worker function for parallel generation. Returns metadata or error string."""
    mesh_path, output_dir, config_dict, seed = args
    try:
        config = PipelineConfig(**config_dict)
        gen = SampleGenerator(config, seed=seed)
        return gen.generate(mesh_path, output_dir)
    except Exception as e:
        return f"Error processing {mesh_path}: {e}"


def batch_generate(
    mesh_paths: list[str | Path],
    output_dir: str | Path,
    config: PipelineConfig,
    num_variations: int = 1,
    max_workers: int = 1,
    base_seed: int = 0,
) -> list[dict]:
    """Generate fracture samples for multiple meshes.

    Args:
        mesh_paths: List of input mesh file paths.
        output_dir: Base output directory.
        config: Pipeline configuration.
        num_variations: Number of random variations per mesh.
        max_workers: Number of parallel workers.
        base_seed: Starting random seed.

    Returns:
        List of metadata dicts for successful samples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    sample_idx = 0
    for mesh_path in mesh_paths:
        for var in range(num_variations):
            sample_dir = output_dir / f"sample_{sample_idx:06d}"
            seed = base_seed + sample_idx
            tasks.append((str(mesh_path), str(sample_dir), config.model_dump(), seed))
            sample_idx += 1

    results = []
    errors = []

    if max_workers <= 1:
        for task in tqdm(tasks, desc="Generating samples"):
            result = generate_one(task)
            if isinstance(result, str):
                errors.append(result)
            else:
                results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_one, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Generating samples"):
                result = future.result()
                if isinstance(result, str):
                    errors.append(result)
                else:
                    results.append(result)

    # Write manifest
    manifest = {
        "num_samples": len(results),
        "num_errors": len(errors),
        "errors": errors,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return results
