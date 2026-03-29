"""Tests for the fracture engine."""

import numpy as np
import pytest
import trimesh

from object_break.fracture.seeds import impact_biased_seeds, uniform_seeds, random_surface_point
from object_break.fracture.voronoi import compute_voronoi_cells, cells_to_meshes
from object_break.fracture.cutter import cut_mesh, fracture_mesh
from object_break.physics.simulation import FragmentSimulation


@pytest.fixture
def box_mesh():
    return trimesh.creation.box(extents=[2.0, 2.0, 2.0])


@pytest.fixture
def sphere_mesh():
    return trimesh.creation.icosphere(subdivisions=2, radius=1.0)


class TestSeeds:
    def test_uniform_seeds_shape(self):
        seeds = uniform_seeds(np.array([-1, -1, -1]), np.array([1, 1, 1]), 10)
        assert seeds.shape == (10, 3)

    def test_uniform_seeds_in_bounds(self):
        lo, hi = np.array([-1, -1, -1]), np.array([1, 1, 1])
        seeds = uniform_seeds(lo, hi, 100)
        assert np.all(seeds >= lo)
        assert np.all(seeds <= hi)

    def test_impact_biased_seeds_count(self, sphere_mesh):
        impact = np.array([1.0, 0.0, 0.0])
        seeds = impact_biased_seeds(sphere_mesh, impact, 10, rng=np.random.default_rng(0))
        assert seeds.shape == (10, 3)

    def test_random_surface_point(self, sphere_mesh):
        pt, normal = random_surface_point(sphere_mesh)
        assert pt.shape == (3,)
        assert normal.shape == (3,)
        # Point should be approximately on unit sphere
        assert abs(np.linalg.norm(pt) - 1.0) < 0.15


class TestVoronoi:
    def test_compute_cells(self):
        seeds = np.array([
            [-0.5, 0, 0],
            [0.5, 0, 0],
            [0, -0.5, 0],
            [0, 0.5, 0],
        ])
        cells = compute_voronoi_cells(seeds, np.array([-2, -2, -2]), np.array([2, 2, 2]))
        assert len(cells) == 4
        for cell in cells:
            assert cell.shape[1] == 3
            assert len(cell) >= 4

    def test_cells_to_meshes(self):
        seeds = np.array([
            [-0.5, 0, 0],
            [0.5, 0, 0],
            [0, -0.5, 0],
            [0, 0.5, 0],
        ])
        cells = compute_voronoi_cells(seeds, np.array([-2, -2, -2]), np.array([2, 2, 2]))
        meshes = cells_to_meshes(cells)
        assert len(meshes) == 4
        for m in meshes:
            assert isinstance(m, trimesh.Trimesh)
            assert m.volume > 0


class TestCutter:
    def test_fracture_box_produces_fragments(self, box_mesh):
        rng = np.random.default_rng(42)
        seeds = uniform_seeds(np.array([-1, -1, -1]), np.array([1, 1, 1]), 5, rng)
        fragments = fracture_mesh(box_mesh, seeds)
        assert len(fragments) >= 2
        for f in fragments:
            assert f.volume > 0

    def test_fracture_sphere(self, sphere_mesh):
        rng = np.random.default_rng(42)
        impact = np.array([1.0, 0.0, 0.0])
        seeds = impact_biased_seeds(sphere_mesh, impact, 8, rng=rng)
        fragments = fracture_mesh(sphere_mesh, seeds)
        assert len(fragments) >= 2

    def test_total_volume_approximately_preserved(self, box_mesh):
        rng = np.random.default_rng(42)
        seeds = uniform_seeds(np.array([-1, -1, -1]), np.array([1, 1, 1]), 6, rng)
        fragments = fracture_mesh(box_mesh, seeds)
        total_fragment_volume = sum(f.volume for f in fragments)
        original_volume = box_mesh.volume
        # Allow some tolerance due to boolean ops
        assert total_fragment_volume > original_volume * 0.5
        assert total_fragment_volume < original_volume * 1.5


class TestPhysics:
    def test_simulation_produces_frames(self, box_mesh):
        rng = np.random.default_rng(42)
        seeds = uniform_seeds(np.array([-1, -1, -1]), np.array([1, 1, 1]), 4, rng)
        fragments = fracture_mesh(box_mesh, seeds)

        sim = FragmentSimulation(
            fragments=fragments,
            impact_point=np.array([0, 0, 1]),
            force=5.0,
            rng=np.random.default_rng(0),
        )
        result = sim.run(num_frames=30, fps=30.0)

        assert result.num_frames == 30
        assert result.num_fragments == len(fragments)
        assert result.positions.shape == (30, len(fragments), 3)
        assert result.rotations.shape == (30, len(fragments), 3, 3)

    def test_fragments_move_apart(self, box_mesh):
        rng = np.random.default_rng(42)
        seeds = uniform_seeds(np.array([-1, -1, -1]), np.array([1, 1, 1]), 4, rng)
        fragments = fracture_mesh(box_mesh, seeds)

        sim = FragmentSimulation(
            fragments=fragments,
            impact_point=np.array([0, 0, 1]),
            force=5.0,
            gravity=np.zeros(3),  # No gravity for this test
            rng=np.random.default_rng(0),
        )
        result = sim.run(num_frames=30, fps=30.0)

        # Average distance between fragments should increase over time
        def avg_dist(frame):
            pos = result.positions[frame]
            dists = []
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    dists.append(np.linalg.norm(pos[i] - pos[j]))
            return np.mean(dists)

        assert avg_dist(29) > avg_dist(0)
