"""
Camera pose estimation and mesh rendering package.

Mesh-generation backends:
  hanyuan  — Hunyuan3D-2 (MeshGenerator / HunyuanRenderer)
  trellis  — TRELLIS.2-4B (TrellisMeshGenerator / TrellisRenderer, optional)

Pose estimation models:
  vggt        — Facebook's VGGT-1B (CameraPoseFinder)
  vggt_omega  — Facebook's VGGT-Omega (Omega_CameraPoseFinder)

Recommended usage:
    from camera import create_pose_finder

    pose_finder = create_pose_finder(
        mesh_pth="mesh.glb",
        image_size=512,
        device="cuda",
        pose_model="vggt",      # or "vggt_omega"
        backend="hanyuan"       # or "trellis" (requires trellis2 install)
    )
"""

from .models.hanyuan import MeshGenerator, HunyuanRenderer

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
        create_pose_finder,
        compute_new_pose_from_relative,
    )
except ImportError:
    CameraPoseFinder = None  # type: ignore[misc, assignment]
    Omega_CameraPoseFinder = None  # type: ignore[misc, assignment]
    create_pose_finder = None  # type: ignore[misc, assignment]
    compute_new_pose_from_relative = None  # type: ignore[misc, assignment]

__all__ = [
    # Hunyuan backend
    'MeshGenerator',
    'HunyuanRenderer',
    # TRELLIS backend (None if trellis2 not installed)
    'TrellisMeshGenerator',
    'TrellisRenderer',
    'TrellisPipelineManager',
    'build_trellis_mesh',
    'default_glb_path',
    # Pose estimation (model-agnostic via factory)
    'create_pose_finder',
    'CameraPoseFinder',
    'Omega_CameraPoseFinder',
    'compute_new_pose_from_relative',
]
