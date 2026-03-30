"""Mesh cutting: intersect original mesh with Voronoi cells to produce fragments."""

import numpy as np
import trimesh


def cut_mesh(
    mesh: trimesh.Trimesh,
    cell_meshes: list[trimesh.Trimesh],
) -> list[trimesh.Trimesh]:
    """Cut a mesh into fragments using Voronoi cell shapes.

    For each cell, computes the boolean intersection of the original mesh
    with the convex cell hull. Uses manifold3d as the boolean engine.

    Args:
        mesh: The original mesh to fracture.
        cell_meshes: List of convex hull meshes from voronoi.cells_to_meshes.

    Returns:
        List of fragment trimeshes. May contain fewer items than cell_meshes
        if some intersections are empty.
    """
    fragments = []

    for cell in cell_meshes:
        try:
            fragment = trimesh.boolean.intersection([mesh, cell], engine="manifold")

            if fragment is None:
                continue

            # Handle Scene results (multiple meshes)
            if isinstance(fragment, trimesh.Scene):
                for geom in fragment.geometry.values():
                    if isinstance(geom, trimesh.Trimesh) and geom.volume > 1e-10:
                        fragments.append(geom)
                continue

            if not isinstance(fragment, trimesh.Trimesh):
                continue

            if fragment.volume < 1e-10:
                continue

            if len(fragment.vertices) < 4 or len(fragment.faces) < 4:
                continue

            fragments.append(fragment)

        except Exception:
            continue

    return fragments


def fracture_mesh(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    padding: float = 0.05,
) -> list[trimesh.Trimesh]:
    """Full fracture pipeline: seeds → Voronoi cells → mesh cutting.

    Convenience function combining voronoi cell computation and mesh cutting.

    Args:
        mesh: The mesh to fracture.
        seeds: (N, 3) seed points for Voronoi cells.
        padding: Bounding box padding as fraction of mesh size.

    Returns:
        List of fragment trimeshes.
    """
    from .voronoi import compute_voronoi_cells, cells_to_meshes

    bbox_min = mesh.bounds[0]
    bbox_max = mesh.bounds[1]
    bbox_size = bbox_max - bbox_min
    pad = bbox_size * padding

    cells = compute_voronoi_cells(seeds, bbox_min - pad, bbox_max + pad)
    cell_meshes = cells_to_meshes(cells)
    return cut_mesh(mesh, cell_meshes)
