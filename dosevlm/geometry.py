"""Image-grid geometry and resampling between DICOM coordinate frames.

Conventions
-----------
* Volumes are stored as numpy arrays indexed ``[k, j, i]`` == ``[z, y, x]``.
* ``Geometry.origin`` is the patient-coordinate (DICOM LPS, mm) position of
  voxel ``[0, 0, 0]``.
* ``Geometry.spacing`` is ``(sx, sy, sz)`` in mm (column, row, slice spacing).
* ``Geometry.direction`` is a 3x3 matrix whose columns are the unit vectors
  of increasing ``i``, ``j`` and ``k`` in patient space.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class Geometry:
    origin: np.ndarray  # (3,) x, y, z in mm
    spacing: np.ndarray  # (3,) sx, sy, sz in mm
    shape: tuple  # (nz, ny, nx)
    direction: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.spacing = np.asarray(self.spacing, dtype=np.float64)
        self.direction = np.asarray(self.direction, dtype=np.float64)
        self.shape = tuple(int(s) for s in self.shape)

    # ------------------------------------------------------------------ #
    def index_to_world(self, ijk: np.ndarray) -> np.ndarray:
        """``ijk`` (..., 3) continuous voxel indices (i, j, k) -> (..., 3) mm."""
        ijk = np.asarray(ijk, dtype=np.float64)
        return self.origin + (ijk * self.spacing) @ self.direction.T

    def world_to_index(self, xyz: np.ndarray) -> np.ndarray:
        """``xyz`` (..., 3) mm -> (..., 3) continuous (i, j, k) indices."""
        xyz = np.asarray(xyz, dtype=np.float64)
        rel = (xyz - self.origin) @ self.direction  # inverse of orthonormal dir
        return rel / self.spacing

    def world_grid(self) -> np.ndarray:
        """Patient coordinates of every voxel centre, shape (nz, ny, nx, 3)."""
        nz, ny, nx = self.shape
        k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
        ijk = np.stack([i, j, k], axis=-1).astype(np.float64)
        return self.index_to_world(ijk)

    @property
    def voxel_volume_cc(self) -> float:
        return float(np.prod(self.spacing)) / 1000.0

    def to_dict(self) -> dict:
        return {
            "origin": self.origin.tolist(),
            "spacing": self.spacing.tolist(),
            "shape": list(self.shape),
            "direction": self.direction.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        return cls(np.array(d["origin"]), np.array(d["spacing"]), tuple(d["shape"]),
                   np.array(d.get("direction", np.eye(3))))


def resample(volume: np.ndarray, src: Geometry, dst: Geometry, order: int = 1,
             cval: float = 0.0, chunk: int = 16) -> np.ndarray:
    """Resample ``volume`` defined on ``src`` onto ``dst`` (trilinear by default).

    Works slice-chunk by slice-chunk so memory stays bounded for large grids.
    """
    out = np.empty(dst.shape, dtype=np.float32)
    nz, ny, nx = dst.shape
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    vol = np.asarray(volume, dtype=np.float32)
    for k0 in range(0, nz, chunk):
        ks = np.arange(k0, min(nz, k0 + chunk))
        kk = np.broadcast_to(ks[:, None, None], (len(ks), ny, nx))
        ii = np.broadcast_to(i[None], kk.shape)
        jj = np.broadcast_to(j[None], kk.shape)
        world = dst.index_to_world(np.stack([ii, jj, kk], axis=-1).astype(np.float64))
        sidx = src.world_to_index(world)  # (..., 3) i, j, k
        coords = np.stack([sidx[..., 2], sidx[..., 1], sidx[..., 0]], axis=0)
        out[k0:k0 + len(ks)] = ndimage.map_coordinates(vol, coords, order=order,
                                                       mode="constant", cval=cval)
    return out


def make_target_grid(center_xyz: np.ndarray, spacing, shape) -> Geometry:
    """Axis-aligned grid of ``shape`` (nz, ny, nx) centred on ``center_xyz``."""
    spacing = np.asarray(spacing, dtype=np.float64)
    nz, ny, nx = shape
    extent = (np.array([nx, ny, nz]) - 1) * spacing
    origin = np.asarray(center_xyz, dtype=np.float64) - extent / 2.0
    return Geometry(origin, spacing, (nz, ny, nx), np.eye(3))
