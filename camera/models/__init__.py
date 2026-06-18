"""
Model modules for 3D mesh generation and rendering.

Backends:
  hanyuan  — Hunyuan3D-based mesh generation + PyTorch3D rendering
  trellis  — TRELLIS.2-4B mesh generation + PyTorch3D rendering (optional)
"""

from .hanyuan import MeshGenerator, HunyuanRenderer

try:
    from .trellis import TrellisMeshGenerator, TrellisRenderer
except ImportError:
    TrellisMeshGenerator = None  # type: ignore[misc, assignment]
    TrellisRenderer = None  # type: ignore[misc, assignment]

__all__ = [
    'MeshGenerator',
    'HunyuanRenderer',
    'TrellisMeshGenerator',
    'TrellisRenderer',
]
