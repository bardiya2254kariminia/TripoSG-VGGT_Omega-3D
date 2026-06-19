"""
Camera pose estimation and mesh rendering package.

Mesh-generation backends:
  hanyuan  — Hunyuan3D-2 (MeshGenerator / Renderer)
  trellis  — TRELLIS.2-4B (TrellisMeshGenerator / TrellisRenderer, optional)

Pose estimation:
  vggt_omega  — Facebook's VGGT-Omega (Omega_CameraPoseFinder)

Recommended usage:
    from camera import Omega_CameraPoseFinder, load_vggt_omega

    vggt_model = load_vggt_omega(checkpoint_path, device="cuda")
    pose_finder = Omega_CameraPoseFinder(vggt_model, image_size=512, device="cuda")
    pose_finder.set_mesh("mesh.glb")
"""

from .models.hanyuan import MeshGenerator, Renderer

try:
    from .models.trellis import TrellisMeshGenerator, TrellisRenderer
    from .utils.trellis_utils import (
        TrellisPipelineManager,
        build_trellis_mesh,
        default_glb_path,
    )
except ImportError:
    TrellisMeshGenerator = None  # type: ignore[misc, assignment]
    TrellisRenderer = None  # type: ignore[misc, assignment]
    TrellisPipelineManager = None  # type: ignore[misc, assignment]
    build_trellis_mesh = None  # type: ignore[misc, assignment]
    default_glb_path = None  # type: ignore[misc, assignment]

# Pose finder imports are optional — mesh-only workflows (hanyuan inference) do not
# need matplotlib / full pose stack at import time.
try:
    from .camera_pose_finder import (
        CameraPoseFinder,
        Omega_CameraPoseFinder,
        load_vggt_omega,
        compute_new_pose_from_relative,
    )
except ImportError:
    CameraPoseFinder = None  # type: ignore[misc, assignment]
    Omega_CameraPoseFinder = None  # type: ignore[misc, assignment]
    load_vggt_omega = None  # type: ignore[misc, assignment]
    compute_new_pose_from_relative = None  # type: ignore[misc, assignment]

__all__ = [
    # Hunyuan backend
    'MeshGenerator',
    'Renderer',
    # TRELLIS backend (None if trellis2 not installed)
    'TrellisMeshGenerator',
    'TrellisRenderer',
    'TrellisPipelineManager',
    'build_trellis_mesh',
    'default_glb_path',
    # Pose estimation
    'load_vggt_omega',
    'CameraPoseFinder',
    'Omega_CameraPoseFinder',
    'compute_new_pose_from_relative',
]
