"""Pipeline configuration models."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class FractureConfig(BaseModel):
    num_pieces: int = Field(default=8, ge=2, le=100)
    impact_point: Optional[list[float]] = None
    impact_direction: Optional[list[float]] = None
    force_magnitude: float = Field(default=5.0, ge=0.1)
    seed_spread: float = Field(default=2.0, ge=0.1)


class PhysicsConfig(BaseModel):
    gravity: list[float] = Field(default=[0.0, 0.0, -9.81])
    num_frames: int = Field(default=60, ge=1)
    fps: float = Field(default=30.0, ge=1.0)
    angular_velocity_scale: float = 3.0
    velocity_scale: float = 2.0
    damping: float = 0.01


class RenderConfig(BaseModel):
    resolution: list[int] = Field(default=[512, 512])
    bg_color: str = "white"
    camera_distance: float = 3.0
    camera_elevation: float = 30.0
    camera_azimuth: float = 45.0
    save_frames: bool = True
    save_video: bool = True


class PipelineConfig(BaseModel):
    fracture: FractureConfig = FractureConfig()
    physics: PhysicsConfig = PhysicsConfig()
    render: RenderConfig = RenderConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
