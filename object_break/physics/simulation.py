"""Simple rigid body physics simulation for fragment separation."""

from dataclasses import dataclass, field

import numpy as np
import trimesh


@dataclass
class FragmentState:
    """State of a single fragment during simulation."""

    mesh: trimesh.Trimesh
    mass: float
    centroid: np.ndarray
    position: np.ndarray  # displacement from original centroid
    velocity: np.ndarray
    rotation: np.ndarray  # rotation matrix (3x3)
    angular_velocity: np.ndarray  # axis-angle per second


@dataclass
class SimulationResult:
    """Full simulation output: transforms for every fragment at every frame."""

    num_frames: int
    num_fragments: int
    # (num_frames, num_fragments, 3) positions
    positions: np.ndarray
    # (num_frames, num_fragments, 3, 3) rotation matrices
    rotations: np.ndarray
    fps: float


class FragmentSimulation:
    """Simulates fragments flying apart from an impact point.

    Each fragment receives an initial velocity directed away from the
    impact point, with magnitude inversely proportional to distance.
    Fragments also get random angular velocity for tumbling.
    """

    def __init__(
        self,
        fragments: list[trimesh.Trimesh],
        impact_point: np.ndarray,
        impact_direction: np.ndarray | None = None,
        force: float = 5.0,
        gravity: np.ndarray | None = None,
        velocity_scale: float = 2.0,
        angular_velocity_scale: float = 3.0,
        damping: float = 0.01,
        rng: np.random.Generator | None = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])
        self.damping = damping
        self.rng = rng

        self.states: list[FragmentState] = []
        impact_point = np.asarray(impact_point, dtype=float)

        if impact_direction is not None:
            impact_direction = np.asarray(impact_direction, dtype=float)
            impact_direction /= np.linalg.norm(impact_direction) + 1e-10

        for frag in fragments:
            centroid = frag.centroid
            mass = max(frag.volume, 1e-6)

            # Direction away from impact point
            away = centroid - impact_point
            dist = np.linalg.norm(away) + 1e-6
            away_dir = away / dist

            # Velocity: inverse distance weighting (closer to impact = faster)
            # Plus a component along the impact direction
            speed = force * velocity_scale / (1.0 + dist * 5.0)
            vel = away_dir * speed

            if impact_direction is not None:
                # Add component along impact direction
                vel += impact_direction * speed * 0.5

            # Random perturbation
            vel += rng.standard_normal(3) * speed * 0.3

            # Random angular velocity for tumbling
            ang_vel = rng.standard_normal(3) * angular_velocity_scale

            self.states.append(FragmentState(
                mesh=frag,
                mass=mass,
                centroid=centroid.copy(),
                position=np.zeros(3),
                velocity=vel,
                rotation=np.eye(3),
                angular_velocity=ang_vel,
            ))

    def step(self, dt: float) -> None:
        """Advance simulation by dt seconds."""
        for state in self.states:
            # Apply gravity
            state.velocity += self.gravity * dt

            # Apply damping
            state.velocity *= (1.0 - self.damping)
            state.angular_velocity *= (1.0 - self.damping)

            # Update position
            state.position += state.velocity * dt

            # Update rotation (small angle approximation for angular velocity)
            angle = np.linalg.norm(state.angular_velocity) * dt
            if angle > 1e-8:
                axis = state.angular_velocity / (np.linalg.norm(state.angular_velocity) + 1e-10)
                dR = _rotation_matrix(axis, angle)
                state.rotation = dR @ state.rotation

    def run(self, num_frames: int, fps: float = 30.0) -> SimulationResult:
        """Run the full simulation and return all frame transforms.

        Args:
            num_frames: Number of frames to simulate.
            fps: Frames per second.

        Returns:
            SimulationResult with positions and rotations for all frames.
        """
        dt = 1.0 / fps
        n = len(self.states)

        positions = np.zeros((num_frames, n, 3))
        rotations = np.zeros((num_frames, n, 3, 3))

        for frame in range(num_frames):
            for i, state in enumerate(self.states):
                positions[frame, i] = state.centroid + state.position
                rotations[frame, i] = state.rotation

            if frame < num_frames - 1:
                self.step(dt)

        return SimulationResult(
            num_frames=num_frames,
            num_fragments=n,
            positions=positions,
            rotations=rotations,
            fps=fps,
        )


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula: rotation matrix from axis-angle."""
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
