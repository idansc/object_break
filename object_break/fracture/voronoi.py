"""Voronoi-based 3D fracture computation."""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull
import trimesh


def compute_voronoi_cells(
    seeds: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> list[np.ndarray]:
    """Compute bounded Voronoi cells from seed points.

    Uses mirror points to handle boundary regions, ensuring all cells
    are finite and clipped to the bounding box.

    Args:
        seeds: (N, 3) array of seed points.
        bbox_min: Minimum corner of bounding box.
        bbox_max: Maximum corner of bounding box.

    Returns:
        List of (M, 3) vertex arrays, one per cell (convex hull vertices).
    """
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min

    # Mirror seeds across each face of the bounding box to bound all Voronoi regions
    mirror_points = []
    for axis in range(3):
        for sign in [-1, 1]:
            mirrored = seeds.copy()
            if sign == -1:
                mirrored[:, axis] = 2 * bbox_min[axis] - seeds[:, axis]
            else:
                mirrored[:, axis] = 2 * bbox_max[axis] - seeds[:, axis]
            mirror_points.append(mirrored)

    all_points = np.vstack([seeds] + mirror_points)
    num_original = len(seeds)

    vor = Voronoi(all_points)

    cells = []
    for i in range(num_original):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            # Fallback: create a small box around the seed
            cells.append(_small_box(seeds[i], bbox_size * 0.01))
            continue

        vertices = vor.vertices[region]

        # Clip to bounding box
        vertices = np.clip(vertices, bbox_min, bbox_max)

        # Remove degenerate cells
        if len(vertices) < 4:
            cells.append(_small_box(seeds[i], bbox_size * 0.01))
            continue

        # Check if vertices span 3D (not coplanar)
        centered = vertices - vertices.mean(axis=0)
        if np.linalg.matrix_rank(centered, tol=1e-8) < 3:
            cells.append(_small_box(seeds[i], bbox_size * 0.01))
            continue

        cells.append(vertices)

    return cells


def cells_to_meshes(cells: list[np.ndarray]) -> list[trimesh.Trimesh]:
    """Convert Voronoi cell vertex arrays to convex hull trimesh objects.

    Args:
        cells: List of (M, 3) vertex arrays from compute_voronoi_cells.

    Returns:
        List of trimesh.Trimesh convex hulls.
    """
    meshes = []
    for vertices in cells:
        try:
            hull = ConvexHull(vertices)
            # hull.points are the input vertices, hull.simplices index into them
            cell_mesh = trimesh.Trimesh(
                vertices=hull.points,
                faces=hull.simplices,
            )
            # Fix winding so volume is positive
            cell_mesh.fix_normals()
            meshes.append(cell_mesh)
        except Exception:
            continue
    return meshes


def _small_box(center: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    """Create vertices of a small box centered at a point."""
    offsets = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=float)
    return center + offsets * half_size
