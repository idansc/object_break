"""Offscreen rendering of fracture simulation frame sequences using PyVista."""

from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from ..physics.simulation import SimulationResult


# Distinct colors for fragments
FRAGMENT_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
]


class SequenceRenderer:
    """Renders frame sequences of fragments separating."""

    def __init__(
        self,
        resolution: tuple[int, int] = (512, 512),
        bg_color: str = "white",
        camera_distance: float = 3.0,
        camera_elevation: float = 30.0,
        camera_azimuth: float = 45.0,
    ):
        self.resolution = resolution
        self.bg_color = bg_color
        self.camera_distance = camera_distance
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth

    def render_sequence(
        self,
        fragments: list[trimesh.Trimesh],
        sim_result: SimulationResult,
        output_dir: Path | str,
        save_video: bool = True,
        save_frames: bool = True,
    ) -> Path | None:
        """Render all frames of the simulation.

        Args:
            fragments: List of fragment meshes in original position.
            sim_result: Simulation result with per-frame transforms.
            output_dir: Directory to write frames and video to.
            save_video: Whether to create an mp4 video.
            save_frames: Whether to save individual frame PNGs.

        Returns:
            Path to output video if save_video=True, else None.
        """
        output_dir = Path(output_dir)
        frames_dir = output_dir / "frames"

        if save_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)

        # Compute scene bounds from all frames for consistent camera
        all_positions = sim_result.positions  # (num_frames, num_fragments, 3)
        scene_center = all_positions.mean(axis=(0, 1))

        # Convert fragments to PyVista meshes once
        pv_meshes = []
        for frag in fragments:
            faces_pv = np.column_stack([
                np.full(len(frag.faces), 3),
                frag.faces,
            ]).ravel()
            pv_mesh = pv.PolyData(frag.vertices.copy(), faces_pv)
            pv_meshes.append(pv_mesh)

        images = []

        for frame_idx in range(sim_result.num_frames):
            img = self._render_frame(
                pv_meshes, fragments, sim_result, frame_idx, scene_center
            )
            images.append(img)

            if save_frames:
                frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
                _save_image(img, frame_path)

        video_path = None
        if save_video and len(images) > 0:
            video_path = output_dir / "video.mp4"
            _save_video(images, video_path, fps=sim_result.fps)

        return video_path

    def _render_frame(
        self,
        pv_meshes: list[pv.PolyData],
        fragments: list[trimesh.Trimesh],
        sim_result: SimulationResult,
        frame_idx: int,
        scene_center: np.ndarray,
    ) -> np.ndarray:
        """Render a single frame, returning an RGBA numpy array."""
        pl = pv.Plotter(off_screen=True, window_size=self.resolution)
        pl.set_background(self.bg_color)

        for i, (pv_mesh, frag) in enumerate(zip(pv_meshes, fragments)):
            # Get transform for this frame
            pos = sim_result.positions[frame_idx, i]
            rot = sim_result.rotations[frame_idx, i]

            # Transform mesh: rotate around original centroid, then translate
            centroid = frag.centroid
            verts = pv_mesh.points.copy()

            # Rotate around centroid
            verts_centered = verts - centroid
            verts_rotated = (rot @ verts_centered.T).T
            verts_transformed = verts_rotated + pos

            # Create transformed mesh
            transformed = pv_mesh.copy()
            transformed.points = verts_transformed

            color = FRAGMENT_COLORS[i % len(FRAGMENT_COLORS)]
            pl.add_mesh(transformed, color=color, smooth_shading=True)

        # Set up camera
        pl.camera_position = _compute_camera_position(
            scene_center,
            self.camera_distance,
            self.camera_elevation,
            self.camera_azimuth,
        )

        img = pl.screenshot(return_img=True)
        pl.close()
        return img

    def render_intact(
        self,
        mesh: trimesh.Trimesh,
        output_path: Path | str,
    ) -> np.ndarray:
        """Render the intact (unfractured) mesh."""
        pl = pv.Plotter(off_screen=True, window_size=self.resolution)
        pl.set_background(self.bg_color)

        faces_pv = np.column_stack([
            np.full(len(mesh.faces), 3),
            mesh.faces,
        ]).ravel()
        pv_mesh = pv.PolyData(mesh.vertices, faces_pv)
        pl.add_mesh(pv_mesh, color="#cccccc", smooth_shading=True)

        center = mesh.centroid
        pl.camera_position = _compute_camera_position(
            center, self.camera_distance, self.camera_elevation, self.camera_azimuth,
        )

        img = pl.screenshot(str(output_path), return_img=True)
        pl.close()
        return img


def _compute_camera_position(
    center: np.ndarray,
    distance: float,
    elevation: float,
    azimuth: float,
) -> list:
    """Compute camera position, focal point, and up vector."""
    elev_rad = np.radians(elevation)
    azim_rad = np.radians(azimuth)

    x = center[0] + distance * np.cos(elev_rad) * np.cos(azim_rad)
    y = center[1] + distance * np.cos(elev_rad) * np.sin(azim_rad)
    z = center[2] + distance * np.sin(elev_rad)

    camera_pos = [x, y, z]
    focal_point = center.tolist()
    up = [0, 0, 1]
    return [camera_pos, focal_point, up]


def _save_image(img: np.ndarray, path: Path) -> None:
    """Save numpy array as PNG image."""
    import imageio.v3 as iio
    iio.imwrite(str(path), img)


def _save_video(images: list[np.ndarray], path: Path, fps: float = 30.0) -> None:
    """Save list of image arrays as mp4 video."""
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for img in images:
        writer.append_data(img)
    writer.close()
