"""Rigid body physics simulation for fragment separation."""

from dataclasses import dataclass

import numpy as np
import trimesh


EXPLOSION_MODES = [
    "explode",       # all pieces fly outward from center
    "impact",        # pieces fly away from impact point (directional)
    "split_half",    # two halves separate along a random axis
    "scatter",       # random directions per piece
    "peel",          # pieces peel outward from surface normals
    "directional",   # all pieces fly in roughly the same direction
]


@dataclass
class FragmentState:
    """State of a single fragment during simulation."""

    mesh: trimesh.Trimesh
    mass: float
    centroid: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    angular_velocity: np.ndarray


@dataclass
class SimulationResult:
    """Full simulation output: transforms for every fragment at every frame."""

    num_frames: int
    num_fragments: int
    positions: np.ndarray       # (num_frames, num_fragments, 3)
    rotations: np.ndarray       # (num_frames, num_fragments, 3, 3)
    fps: float
    hold_frames: int = 0


def compute_velocities(
    fragments: list[trimesh.Trimesh],
    mode: str,
    force: float,
    velocity_scale: float,
    impact_point: np.ndarray | None = None,
    impact_direction: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Compute initial velocity for each fragment based on explosion mode.

    Args:
        fragments: List of fragment meshes.
        mode: One of EXPLOSION_MODES.
        force: Base force magnitude.
        velocity_scale: Multiplier on velocity.
        impact_point: Point of impact (used by 'impact' mode).
        impact_direction: Direction of impact (used by 'impact', 'directional').
        rng: Random number generator.

    Returns:
        List of velocity vectors, one per fragment.
    """
    if rng is None:
        rng = np.random.default_rng()

    centroids = np.array([f.centroid for f in fragments])
    center = centroids.mean(axis=0)
    speed = force * velocity_scale

    if mode == "explode":
        # Every piece flies outward from the geometric center
        vels = []
        for c in centroids:
            away = c - center
            dist = np.linalg.norm(away) + 1e-6
            direction = away / dist
            # Uniform speed + small random variation
            v = direction * speed * (0.8 + rng.random() * 0.4)
            v += rng.standard_normal(3) * speed * 0.1
            vels.append(v)
        return vels

    elif mode == "impact":
        # Pieces fly away from impact point, biased along impact direction
        if impact_point is None:
            impact_point = center
        vels = []
        for c in centroids:
            away = c - impact_point
            dist = np.linalg.norm(away) + 1e-6
            direction = away / dist
            s = speed / (1.0 + dist * 3.0)
            v = direction * s
            if impact_direction is not None:
                v += impact_direction * s * 0.5
            v += rng.standard_normal(3) * s * 0.2
            vels.append(v)
        return vels

    elif mode == "split_half":
        # Split along a random axis — two groups go opposite ways
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis) + 1e-10
        vels = []
        for c in centroids:
            side = np.sign(np.dot(c - center, axis))
            if side == 0:
                side = 1.0
            v = axis * side * speed * 0.6
            v += rng.standard_normal(3) * speed * 0.15
            vels.append(v)
        return vels

    elif mode == "scatter":
        # Each piece gets a completely random direction
        vels = []
        for _ in centroids:
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction) + 1e-10
            s = speed * (0.5 + rng.random())
            vels.append(direction * s)
        return vels

    elif mode == "peel":
        # Pieces fly outward along the direction from center to their centroid
        # but with stronger tangential component — like peeling off a surface
        vels = []
        for c in centroids:
            radial = c - center
            dist = np.linalg.norm(radial) + 1e-6
            radial_dir = radial / dist
            # Add tangential component (perpendicular to radial)
            tangent = rng.standard_normal(3)
            tangent -= np.dot(tangent, radial_dir) * radial_dir
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-6:
                tangent /= tangent_norm
            v = (radial_dir * 0.4 + tangent * 0.6) * speed
            v += rng.standard_normal(3) * speed * 0.1
            vels.append(v)
        return vels

    elif mode == "directional":
        # All pieces fly in roughly the same direction with some spread
        if impact_direction is not None:
            base_dir = np.asarray(impact_direction, dtype=float)
            base_dir /= np.linalg.norm(base_dir) + 1e-10
        else:
            base_dir = rng.standard_normal(3)
            base_dir /= np.linalg.norm(base_dir) + 1e-10
        vels = []
        for _ in centroids:
            v = base_dir * speed * (0.7 + rng.random() * 0.6)
            v += rng.standard_normal(3) * speed * 0.25
            vels.append(v)
        return vels

    else:
        raise ValueError(f"Unknown explosion mode: {mode!r}. Choose from {EXPLOSION_MODES}")


class FragmentSimulation:
    """Simulates fragments flying apart with configurable explosion modes."""

    def __init__(
        self,
        fragments: list[trimesh.Trimesh],
        mode: str = "explode",
        impact_point: np.ndarray | None = None,
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

        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, 0.0])
        self.damping = damping
        self.rng = rng

        # Compute velocities based on explosion mode
        velocities = compute_velocities(
            fragments, mode, force, velocity_scale,
            impact_point=impact_point,
            impact_direction=impact_direction,
            rng=rng,
        )

        self.states: list[FragmentState] = []
        for frag, vel in zip(fragments, velocities):
            ang_vel = rng.standard_normal(3) * angular_velocity_scale

            self.states.append(FragmentState(
                mesh=frag,
                mass=max(frag.volume, 1e-6),
                centroid=frag.centroid.copy(),
                position=np.zeros(3),
                velocity=vel,
                rotation=np.eye(3),
                angular_velocity=ang_vel,
            ))

    def step(self, dt: float) -> None:
        """Advance simulation by dt seconds."""
        for state in self.states:
            state.velocity += self.gravity * dt
            state.velocity *= (1.0 - self.damping)
            state.angular_velocity *= (1.0 - self.damping)
            state.position += state.velocity * dt

            angle = np.linalg.norm(state.angular_velocity) * dt
            if angle > 1e-8:
                axis = state.angular_velocity / (np.linalg.norm(state.angular_velocity) + 1e-10)
                dR = _rotation_matrix(axis, angle)
                state.rotation = dR @ state.rotation

    def run(
        self, num_frames: int, fps: float = 30.0, hold_frames: int = 0
    ) -> SimulationResult:
        """Run the full simulation and return all frame transforms."""
        dt = 1.0 / fps
        n = len(self.states)

        total_frames = hold_frames + num_frames
        positions = np.zeros((total_frames, n, 3))
        rotations = np.zeros((total_frames, n, 3, 3))

        for frame in range(total_frames):
            for i, state in enumerate(self.states):
                positions[frame, i] = state.centroid + state.position
                rotations[frame, i] = state.rotation

            if frame >= hold_frames and frame < total_frames - 1:
                self.step(dt)

        return SimulationResult(
            num_frames=total_frames,
            num_fragments=n,
            positions=positions,
            rotations=rotations,
            fps=fps,
            hold_frames=hold_frames,
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
