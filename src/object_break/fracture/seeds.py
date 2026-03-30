"""Seed point generation strategies for Voronoi fracture."""

import numpy as np
import trimesh


def impact_biased_seeds(
    mesh: trimesh.Trimesh,
    impact_point: np.ndarray,
    num_pieces: int,
    spread: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate seed points concentrated near an impact point.

    Seeds follow a power-law radial distribution from the impact point,
    creating denser fracture near the impact and larger pieces further away.

    Args:
        mesh: The mesh to fracture.
        impact_point: 3D point on or near the mesh surface.
        num_pieces: Number of fragment seeds to generate.
        spread: Controls distribution spread. Higher = more uniform, lower = more concentrated.
        rng: Random number generator.

    Returns:
        (num_pieces, 3) array of seed points inside the mesh bounding box.
    """
    if rng is None:
        rng = np.random.default_rng()

    bbox_min = mesh.bounds[0]
    bbox_max = mesh.bounds[1]
    bbox_size = bbox_max - bbox_min
    max_radius = np.linalg.norm(bbox_size) / 2

    seeds = []
    attempts = 0
    max_attempts = num_pieces * 100

    while len(seeds) < num_pieces and attempts < max_attempts:
        # Power-law radial distribution: more points near impact
        r = rng.power(spread) * max_radius
        # Random direction on unit sphere
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-10

        point = impact_point + direction * r

        # Keep points within bounding box (with small padding)
        pad = bbox_size * 0.05
        if np.all(point >= bbox_min - pad) and np.all(point <= bbox_max + pad):
            seeds.append(point)

        attempts += 1

    if len(seeds) < num_pieces:
        # Fill remaining with uniform seeds
        remaining = num_pieces - len(seeds)
        extra = uniform_seeds(mesh.bounds[0], mesh.bounds[1], remaining, rng)
        seeds.extend(extra.tolist())

    return np.array(seeds[:num_pieces])


def uniform_seeds(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    num_pieces: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate uniformly distributed seed points within a bounding box."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(bbox_min, bbox_max, size=(num_pieces, 3))


def random_surface_point(
    mesh: trimesh.Trimesh,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a random point on the mesh surface.

    Returns:
        Tuple of (point, face_normal) at the sampled location.
    """
    if rng is None:
        rng = np.random.default_rng()

    points, face_indices = trimesh.sample.sample_surface(mesh, 1)
    point = points[0]
    normal = mesh.face_normals[face_indices[0]]
    return point, normal
