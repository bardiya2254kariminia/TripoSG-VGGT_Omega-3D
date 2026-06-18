"""
Utility modules for camera operations.
"""

try:
    from .trellis_utils import (
        TrellisPipelineManager,
        build_trellis_mesh,
        default_glb_path,
    )
except ImportError:
    TrellisPipelineManager = None  # type: ignore[misc, assignment]
    build_trellis_mesh = None  # type: ignore[misc, assignment]
    default_glb_path = None  # type: ignore[misc, assignment]

__all__ = [
    'TrellisPipelineManager',
    'build_trellis_mesh',
    'default_glb_path',
]
